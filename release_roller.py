#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
long2wide_fixed_v1.py

Fix principali rispetto a long2wide.py / merge_observations_safe_v13:

1) CODICE_SCUOLA canonicalizzato SEMPRE (estrae il primo codice meccanografico valido a 10 caratteri)
   -> evita righe duplicate per "stessa scuola" con caratteri sporchi/spazi/etc.
2) Ogni blocco che entra nel merge viene reso 1-riga-per-scuola (groupby first) prima della merge.
3) Dedup finale: se, nonostante tutto, restano duplicati per CODICE_SCUOLA, collassa in 1 riga
   prendendo il primo valore non-vuoto / non-NaN per colonna.
4) Riordino colonne: tutta l'ANAGRAFICA all'inizio, poi le variabili (ordinate per prefisso numerico).
5) Non modifica i valori numerici già presenti nel LONG (salvo parsing in costruzione LONG).

Input/Output:
- --inputs ... (dir che contengono observations_semantic.csv + anagrafica*.csv)
- --out-long-dir ... (scrive part-*.parquet)
- --out-wide-csv ...
- --out-wide-parquet ...
- --snai-xlsx ... (opzionale; file CAP + SNAI.xlsx)
"""

import argparse
import os
import csv
import io
import re
import unicodedata
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Dict

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore
    _HAVE_PYARROW = False

def _require_pyarrow() -> None:
    if not _HAVE_PYARROW:
        raise RuntimeError("pyarrow non disponibile. Installa pyarrow per usare output parquet (wide.parquet e long/*.parquet).")



# ----------------------------
# Discovery
# ----------------------------

def find_obs_csv_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for x in inputs:
        p = Path(x)
        if p.is_dir():
            cand = p / "observations_semantic.csv"
            if cand.exists():
                paths.append(cand)
            else:
                paths.extend(list(p.glob("**/observations_semantic*.csv")))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"Input non trovato: {x}")
    seen = set()
    out: List[Path] = []
    for fp in paths:
        s = str(fp.resolve())
        if s not in seen:
            seen.add(s)
            out.append(Path(s))
    return out


def find_anagrafica_csv_paths(inputs: List[str], pattern: str = "anagrafica*.csv") -> List[Path]:
    paths: List[Path] = []
    for x in inputs:
        p = Path(x)
        if p.is_dir():
            paths.extend(sorted(p.glob(pattern)))
            paths.extend(sorted(p.glob(f"**/{pattern}")))
    seen = set()
    out: List[Path] = []
    for fp in paths:
        s = str(fp.resolve())
        if s not in seen:
            seen.add(s)
            out.append(Path(s))
    return out


# ----------------------------
# Canonicalizzazione CODICE_SCUOLA
# ----------------------------

_MECC_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{8}\b")


# ----------------------------
# Release / Area packaging
# ----------------------------

AREE_GEOGRAFICHE = ("SUD", "NORDEST", "NORDOVEST", "CENTRO", "ISOLE")

def detect_area_from_path(fp: Path) -> Optional[str]:
    """Rileva l'area geografica cercando una substring nelle directory antenate del file."""
    parts = [p.name.upper() for p in fp.resolve().parents]
    for name in parts:
        for a in AREE_GEOGRAFICHE:
            if a in name:
                return a
    return None

def group_paths_by_area(paths: List[Path]) -> dict:
    out = {a: [] for a in AREE_GEOGRAFICHE}
    for fp in paths:
        a = detect_area_from_path(fp)
        if a:
            out[a].append(fp)
    # drop empties
    return {k: v for k, v in out.items() if v}

def _first_nonempty(series: pd.Series):
    for v in series:
        if pd.isna(v):
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return np.nan


def first_non_empty_series(series: pd.Series):
    """Aggregatore per groupby. Ritorna il primo valore non vuoto/non-NaN."""
    return _first_nonempty(series)

def load_sic_studenti_csv(fp: Path) -> pd.DataFrame:
    """Carica il dataset sic-studenti in modo robusto.

    Il CSV può contenere virgole non quotate in campi testuali (tipico: denominazioni),
    causando ParserError. In quel caso, prova una ricostruzione conservativa:
    - se l'header ha 3 colonne, spezza ogni riga in massimo 3 campi (split con maxsplit=2)
      e accorpa le virgole residue nell'ultimo campo.
    - altrimenti, ripiega su engine='python' e scarta solo le righe irrimediabilmente rotte.
    """
    try:
        df = pd.read_csv(fp, sep=",", encoding="utf-8", dtype=str, keep_default_na=False)
    except Exception as e:
        # ParserError o simili
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            raise

        if not raw:
            raise ValueError("sic-studenti: file vuoto")

        header = raw[0].split(",")
        n = len(header)

        if n == 3:
            rows = []
            for line in raw[1:]:
                if not line:
                    continue
                parts = line.split(",", 2)
                if len(parts) < 3:
                    parts = parts + [""] * (3 - len(parts))
                rows.append(parts[:3])
            df = pd.DataFrame(rows, columns=header)
        else:
            # Ripiego: parsing più permissivo (alcune righe possono essere scartate)
            df = pd.read_csv(
                fp,
                sep=",",
                encoding="utf-8",
                dtype=str,
                keep_default_na=False,
                engine="python",
                on_bad_lines="skip",
            )

    if "CODICE_SCUOLA" not in df.columns:
        raise ValueError("sic-studenti: colonna CODICE_SCUOLA non trovata")

    df["CODICE_SCUOLA"] = df["CODICE_SCUOLA"].map(canon_mecc)
    df = df[df["CODICE_SCUOLA"].ne("")].copy()

    # dedup: 1 riga per scuola, primo valore non vuoto per colonna
    df = df.groupby("CODICE_SCUOLA", as_index=False).agg(first_non_empty_series)

    return df

def _norm_comune_key(x: str) -> str:
    s = "" if x is None else str(x)
    s = s.strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s

def _strip_accents(s: str) -> str:
    if s is None:
        return ""
    # NFKD: separa diacritici
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def _norm_place_key(x: str) -> str:
    """Normalizza denominazioni di comuni per match robusto:
    - maiuscole, senza accenti
    - spazi compressi
    - rimozione di punteggiatura comune
    """
    s = "" if x is None else str(x)
    s = _strip_accents(s)
    s = s.strip().upper()
    s = s.replace("’", "'")
    # In molte fonti i caratteri accentati compaiono come apostrofo finale (es. CITTA')
    # Rimuovo l'apostrofo SOLO se segue una vocale (A/E/I/O/U): Citta' -> CITTA
    s = re.sub(r"([AEIOU])'", r"\1", s)
    s = re.sub(r"[^A-Z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_cap(x: object) -> str:
    """Normalizza CAP a 5 cifre con padding a sinistra.

    Regole:
    - vuoto/NaN -> ""
    - numerico 1..4 cifre -> zfill(5)
    - numerico 5 cifre -> invariato
    - non numerico -> "" (evita filtri semanticamente falsi)
    """
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    if not s.isdigit():
        return ""
    if len(s) > 5:
        # CAP italiani sono 5 cifre; se arriva altro (es. codici esteri) lo scartiamo
        return ""
    return s.zfill(5)


# Mappatura "tipoDiIstruzione" (sigle) -> etichetta leggibile.
# Fonti: nota MIUR/Edscuola sulle sigle delle istituzioni scolastiche (PC, PM, ...)
# e documentazione su "Infanzia (AA)".
TIPO_ISTRUZIONE_MAP: Dict[str, str] = {
    # === I CICLO ===
    "AA": "Scuola dell'infanzia",
    "EE": "Scuola primaria",
    "MM": "Scuola secondaria di I grado",
    "SD": "Scuola secondaria di I grado (annessa convitto)",

    # === LICEI ===
    "PC": "Liceo classico",
    "PS": "Liceo scientifico",
    "PM": "Liceo scientifico opz. scienze applicate",
    "PL": "Liceo linguistico",
    "PP": "Liceo delle scienze umane",
    "PQ": "Liceo delle scienze umane opz. economico-sociale",
    "PR": "Liceo musicale",
    "PT": "Liceo coreutico",
    "PB": "Liceo artistico",
    "RS": "Liceo scientifico europeo",
    "RV": "Liceo scientifico (annesso convitto)",
    "SL": "Liceo (annesso convitto)",

    # === ISTITUTI TECNICI (storici e ordinamento vigente) ===
    "TA": "Istituto tecnico agrario",
    "TB": "Istituto tecnico per geometri (Costruzioni, ambiente e territorio)",
    "TD": "Istituto tecnico commerciale (Amministrazione, finanza e marketing)",
    "TE": "Istituto tecnico industriale",
    "TF": "Istituto tecnico (indirizzi storici femminili / economia domestica)",
    "TH": "Istituto tecnico nautico (Trasporti e logistica)",
    "TL": "Istituto tecnico per il turismo",
    "TM": "Istituto tecnico (Meccanica, meccatronica ed energia)",
    "TN": "Istituto tecnico (Elettronica ed elettrotecnica)",
    "TG": "Istituto tecnico (Grafica e comunicazione)",
    "TI": "Istituto tecnico (Sistema moda)",
    "TJ": "Istituto tecnico (Agraria, agroalimentare e agroindustria)",
    "TK": "Istituto tecnico (Costruzioni, ambiente e territorio)",
    "TF": "Istituto tecnico (Informatica e telecomunicazioni)",

    # === ISTITUTI PROFESSIONALI ===
    "RA": "Istituto professionale (Servizi per l'agricoltura e lo sviluppo rurale)",
    "RB": "Istituto professionale (Servizi socio-sanitari)",
    "RC": "Istituto professionale (Servizi commerciali)",
    "RD": "Istituto professionale (Manutenzione e assistenza tecnica)",
    "RE": "Istituto professionale (Produzioni industriali e artigianali)",
    "RF": "Istituto professionale (Pesca commerciale e produzioni ittiche)",
    "RH": "Istituto professionale (Enogastronomia e ospitalità alberghiera)",
    "RI": "Istituto professionale (Servizi culturali e dello spettacolo)",
    "RJ": "Istituto professionale (Arti ausiliarie delle professioni sanitarie)",

    # === ALTRE ISTITUZIONI ===
    "RM": "Corso serale",
    "VC": "Convitto nazionale",
    "VE": "Educandato statale",
}



def canon_mecc(x: str) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\u00a0", " ").strip().upper()
    m = _MECC_RE.search(s)
    if m:
        return m.group(0)
    # fallback: togli spazi e caratteri strani
    s2 = re.sub(r"\s+", "", s)
    s2 = re.sub(r"[^A-Z0-9]", "", s2)
    return s2


# ----------------------------
# Robust CSV streaming
# ----------------------------

def _iter_records_balanced_quotes(fp: Path, encoding: str = "utf-8") -> Iterable[str]:
    max_join_lines = 10
    with fp.open("rb") as f:
        buf: List[str] = []
        q = 0
        n = 0
        for raw in f:
            line = raw.decode(encoding, errors="replace")
            buf.append(line)
            q += line.count('"')
            n += 1
            if q % 2 == 0:
                yield "".join(buf)
                buf, q, n = [], 0, 0
            elif n >= max_join_lines:
                # linea probabilmente rotta: "normalizza" togliendo quote
                yield "".join(buf).replace('"', "")
                buf, q, n = [], 0, 0
        if buf:
            rec = "".join(buf)
            if q % 2 == 1:
                rec = rec.replace('"', "")
            yield rec


def safe_read_csv_chunks(fp: Path, chunksize: int = 100_000) -> Iterable[pd.DataFrame]:
    it = _iter_records_balanced_quotes(fp, encoding="utf-8")
    header = next(it, None)
    if header is None:
        return
    batch: List[str] = []
    rows = 0

    def _parse(text: str) -> pd.DataFrame:
        try:
            return pd.read_csv(
                io.StringIO(text),
                dtype=str,
                keep_default_na=False,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception:
            return pd.read_csv(
                io.StringIO(text.replace('"', "")),
                dtype=str,
                keep_default_na=False,
                engine="python",
                on_bad_lines="skip",
            )

    for rec in it:
        batch.append(rec)
        rows += 1
        if rows >= chunksize:
            yield _parse(header + "".join(batch))
            batch, rows = [], 0
    if batch:
        yield _parse(header + "".join(batch))


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def to_float_or_nan(s: pd.Series) -> pd.Series:
    """
    Parsing numerico conservativo (fix della vecchia regola sbagliata "togli sempre i punti"):
    - se contiene sia '.' che ',' -> assume formato europeo 1.234,56
    - se contiene solo ',' -> virgola decimale
    - se contiene solo '.' -> punto decimale
    """
    x = s.astype(str).str.replace("\u00a0", " ").str.strip()
    x = x.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
    x = x.str.replace("%", "", regex=False)

    has_comma = x.str.contains(",", regex=False, na=False)
    has_dot = x.str.contains(".", regex=False, na=False)

    both = has_comma & has_dot
    if both.any():
        xb = x.where(~both, x)
        xb = xb.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        x = x.where(~both, xb)

    comma_only = has_comma & ~has_dot
    if comma_only.any():
        xc = x.where(~comma_only, x).str.replace(",", ".", regex=False)
        x = x.where(~comma_only, xc)

    x = x.str.replace(" ", "", regex=False)
    return pd.to_numeric(x, errors="coerce")



_STAR_COMPUTED_SUFFIXES: tuple[str, ...] = (
    "_mean_y",
    "_mean_y1y2",
    "_slope_y",
    "_sum_scuola",
    "_tasso_non_ammessi_pct",
    "_share_top_pct",
    "_share_low_pct",
    "_share_mean",
)

_STAR_COMPUTED_EXACT: set[str] = {
    # SNV/RAV derived means
    "media_punteggiomedio_superiori_classiseconde",
    "media_punteggiomedio_superiori_classiquinteultimoanno",
}


def _star_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix '*' ONLY to columns created by this script via explicit calculations.

    We do not star raw scraped variables or reshaped pivots.
    Idempotent.
    """
    ren: dict[str, str] = {}
    for c in df.columns:
        if c.startswith("*"):
            continue
        if c in _STAR_COMPUTED_EXACT or any(c.endswith(sfx) for sfx in _STAR_COMPUTED_SUFFIXES):
            ren[c] = "*" + c
    if ren:
        df = df.rename(columns=ren)
    return df


def _add_snv_means(df: pd.DataFrame) -> pd.DataFrame:
    """Add SNV-derived mean columns (already starred) from 80_rav_22a1_invalsipunteggiomedio_* variables."""
    base = "80_rav_22a1_invalsipunteggiomedio_"

    cols_seconde = [
        base + "superiori_classiseconde_italiano_punteggio_medio_",
        base + "superiori_classiseconde_matematica_punteggio_medio_",
    ]

    cols_quinte = [
        base + "superiori_classiquinteultimoanno_italiano_punteggio_medio_",
        base + "superiori_classiquinteultimoanno_matematica_punteggio_medio_",
        base + "superiori_classiquinteultimoanno_inglesereading_punteggio_medio_",
        base + "superiori_classiquinteultimoanno_ingleselistening_punteggio_medio_",
    ]

    def row_mean(cols: list[str]) -> pd.Series:
        ex = [c for c in cols if c in df.columns]
        if not ex:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
        mat = pd.concat([to_float_or_nan(df[c]) for c in ex], axis=1)
        out = mat.mean(axis=1, skipna=True)
        out = out.where(mat.notna().any(axis=1))
        return out

    df["*media_punteggiomedio_superiori_classiseconde"] = row_mean(cols_seconde)
    df["*media_punteggiomedio_superiori_classiquinteultimoanno"] = row_mean(cols_quinte)
    return df

def slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    c_school = pick_col(df, ["CODICE_SCUOLA", "codice_scuola", "school_code"])
    c_name = pick_col(df, ["NOME_SCUOLA", "nome_scuola", "school_name"])
    c_ep = pick_col(df, ["endpoint_key", "ENDPOINT_KEY", "endpoint"])
    c_cat = pick_col(df, ["category", "CATEGORY"])
    c_val = pick_col(df, ["value", "VALUE"])
    c_level = pick_col(df, ["level", "LEVEL", "scope", "SCOPE"])
    if any(x is None for x in [c_school, c_ep, c_cat, c_val]):
        raise ValueError(f"Colonne minime non trovate. Colonne viste: {list(df.columns)[:60]}")

    out = pd.DataFrame({
        "CODICE_SCUOLA": df[c_school].map(canon_mecc),
        "NOME_SCUOLA": df[c_name].astype(str).str.strip() if c_name else "",
        "endpoint_key": df[c_ep].astype(str).str.strip(),
        "category": df[c_cat].astype(str).str.strip(),
        "level": df[c_level].astype(str).str.strip() if c_level else "",
        "value_raw": df[c_val].astype(str).str.strip(),
    })
    out = out[out["CODICE_SCUOLA"].ne("")]
    out["value_num"] = to_float_or_nan(out["value_raw"])
    out["value_text"] = out["value_raw"].where(out["value_num"].isna(), "")
    return out[["CODICE_SCUOLA", "NOME_SCUOLA", "endpoint_key", "category", "level", "value_num", "value_text"]]


def write_parquet_part(df: pd.DataFrame, out_dir: Path, part_idx: int) -> None:
    _require_pyarrow()
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"part-{part_idx:06d}.parquet"
    _require_pyarrow()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), fp, compression="zstd")


def read_long_dataset(parquet_dir: Path) -> pd.DataFrame:
    _require_pyarrow()
    return pd.read_parquet(parquet_dir, engine="pyarrow")


def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype).startswith("string"):
            s = out[c].astype(str)
            s = s.str.replace("\r", " ", regex=False).str.replace("\n", " ", regex=False).str.replace("\t", " ", regex=False)
            s = s.str.replace(r"\s{2,}", " ", regex=True)
            out[c] = s
    return out


# ----------------------------
# Anagrafica
# ----------------------------

def _first_nonempty(series):
    for v in series:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "na"}:
            return v
    return ""


def load_anagrafica_df(paths: List[Path], chunksize: int = 100_000) -> Optional[pd.DataFrame]:
    if not paths:
        return None
    frames = []
    for fp in paths:
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                it = _iter_records_balanced_quotes(fp, encoding=enc)
                header = next(it, None)
                if header is None:
                    break
                batch = [header]
                for line in it:
                    batch.append(line)
                    if len(batch) >= chunksize:
                        df = pd.read_csv(
                            io.StringIO("".join(batch)),
                            sep=None,
                            dtype=str,
                            keep_default_na=False,
                            engine="python",
                            on_bad_lines="skip",
                        )
                        batch = [header]
                        c_school = pick_col(df, ["CODICE_SCUOLA", "codice_scuola", "CODICE_MECCANOGRAFICO", "codice_meccanografico", "CODICE", "codice"])
                        if c_school is None:
                            continue
                        tmp = df.rename(columns={c_school: "CODICE_SCUOLA"})
                        tmp["CODICE_SCUOLA"] = tmp["CODICE_SCUOLA"].map(canon_mecc)
                        tmp = tmp[tmp["CODICE_SCUOLA"].ne("")]
                        frames.append(tmp)
                if len(batch) > 1:
                    df = pd.read_csv(
                        io.StringIO("".join(batch)),
                        sep=None,
                        dtype=str,
                        keep_default_na=False,
                        engine="python",
                        on_bad_lines="skip",
                    )
                    c_school = pick_col(df, ["CODICE_SCUOLA", "codice_scuola", "CODICE_MECCANOGRAFICO", "codice_meccanografico", "CODICE", "codice"])
                    if c_school is not None:
                        tmp = df.rename(columns={c_school: "CODICE_SCUOLA"})
                        tmp["CODICE_SCUOLA"] = tmp["CODICE_SCUOLA"].map(canon_mecc)
                        tmp = tmp[tmp["CODICE_SCUOLA"].ne("")]
                        frames.append(tmp)
                break
            except UnicodeDecodeError:
                continue
    if not frames:
        return None
    a = pd.concat(frames, ignore_index=True)
    for c in a.columns:
        a[c] = a[c].astype(str)

    # Normalizzazioni anagrafiche (prima del dedup):
    # - CAP come stringa a 5 cifre (padding zeri a sinistra)
    # - tipoDiIstruzione in etichetta leggibile, quando disponibile
    c_cap = pick_col(a, ["cap", "CAP", "Cap"])
    if c_cap is not None:
        a[c_cap] = a[c_cap].map(_normalize_cap)
        if c_cap != "cap":
            a = a.rename(columns={c_cap: "cap"})

    c_tipo = pick_col(a, [
        "tipoDiIstruzione",
        "TIPODIISTRUZIONE",
        "tipo_di_istruzione",
        "TIPO_DI_ISTRUZIONE",
        "tipoIstruzione",
        "TIPOISTRUZIONE",
    ])
    if c_tipo is not None:
        tt = a[c_tipo].astype(str).str.strip().str.upper()
        a[c_tipo] = tt.map(lambda k: TIPO_ISTRUZIONE_MAP.get(k, k))
        if c_tipo != "tipoDiIstruzione":
            a = a.rename(columns={c_tipo: "tipoDiIstruzione"})
    out = a.groupby("CODICE_SCUOLA", sort=False, as_index=False).aggregate(_first_nonempty)
    # dedup hard
    out = out.drop_duplicates("CODICE_SCUOLA", keep="first")
    return out


# ----------------------------
# WIDE helpers
# ----------------------------

def _is_scuola_scope(d: pd.DataFrame) -> pd.Series:
    cat = d["category"].astype(str).str.strip().str.lower()
    lvl = d["level"].astype(str).str.strip().str.lower()
    return (cat == "scuola") | (lvl == "scuola")


def _year_from_category(cat: str) -> Optional[int]:
    '''Estrae un indice temporale per le serie SIC.

    Supporta:
      - y1..y6 / anno1..anno6 / year1..year6 / a1..a6
      - categorie numeriche 1..6
      - anni scolastici tipo 2020/21 o 2020-21 (usa il primo anno)

    Ritorna None se non ricava un valore affidabile.
    '''
    if cat is None:
        return None
    c = str(cat).strip()
    if not c:
        return None

    s = slugify(c)

    # y1..y6 / anno1..anno6 / year1..year6 / a1..a6
    m = re.search(r"(?:^|_)(?:y|anno|year|a)([1-6])(?:_|$)", s)
    if m:
        return int(m.group(1))

    # categoria già numerica (o contiene un singolo numero 1..6 ben delimitato)
    m = re.search(r"(?:^|_)([1-6])(?:_|$)", s)
    if m:
        return int(m.group(1))

    # anni scolastici 2020/21, 2020-21, 2020_21
    m = re.search(r"(19\d{2}|20\d{2})", c)
    if m:
        return int(m.group(1))

    return None


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Calcola la pendenza lineare (y ~ a + b*x) in modo robusto.

    - Rimuove NaN
    - Se i punti utili sono <2 o x è degenerato, restituisce NaN
    - Sopprime RankWarning di numpy (mal condizionato)
    """
    import warnings

    # filtra NaN
    m = (~np.isnan(x)) & (~np.isnan(y))
    x = x[m]
    y = y[m]

    if len(x) < 2:
        return np.nan
    # x costante -> slope non identificata
    if np.unique(x.astype(float)).size < 2:
        return np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.RankWarning)
            return float(np.polyfit(x.astype(float), y.astype(float), 1)[0])
    except Exception:
        return np.nan



def _linear_slope_indexed(yvals):
    """
    Stima della pendenza (retta ai minimi quadrati) su indici 1..n,
    ignorando valori mancanti/non numerici.
    Ritorna np.nan se <2 punti validi.
    """
    # normalizza: pd.NA / None / stringhe -> np.nan o float
    cleaned = []
    for v in yvals:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cleaned.append(np.nan)
            elif pd.isna(v):
                cleaned.append(np.nan)
            else:
                cleaned.append(float(v))
        except Exception:
            cleaned.append(np.nan)

    y = np.array(cleaned, dtype="float64")
    x = np.arange(1, len(y) + 1, dtype="float64")
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan

    x = x[mask]
    y = y[mask]

    # slope di y = a + b*x
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    b = ((x - x_mean) * (y - y_mean)).sum() / denom
    return float(b)


def _add_forced_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Forza il calcolo delle slope per le 3 serie SIC su y1..y5.

    Motivo: le categorie/anni in input possono essere non uniformi, quindi la slope
    derivata dal long può risultare mancante o incoerente. Qui ricalcoliamo la slope
    direttamente sulle colonne wide y1..y5, quando presenti.

    - 10_abbandoni_slope_y su 10_abbandoni_y1..y5
    - 11_ripetenti_slope_y su 11_ripetenti_y1..y5
    - 12_trasferimenti_slope_y su 12_trasferimenti_y1..y5
    """

    if df is None or df.empty:
        return df

    def _coerce_float_series(s: pd.Series) -> pd.Series:
        out = pd.to_numeric(
            s.astype("string")
             .str.replace(" ", " ")
             .str.strip()
             .str.replace("%", "", regex=False)
             .str.replace(",", ".", regex=False)
             .replace({
                 "": pd.NA,
                 "<NA>": pd.NA,
                 "n.d.": pd.NA, "nd": pd.NA, "N.D.": pd.NA,
                 "N/A": pd.NA, "NA": pd.NA,
                 "nan": pd.NA, "NaN": pd.NA, "None": pd.NA,
             }),
            errors="coerce",
        )
        return out.astype("float64")

    specs = [
        ("10_abbandoni", "10_abbandoni_slope_y"),
        ("11_ripetenti", "11_ripetenti_slope_y"),
        ("12_trasferimenti", "12_trasferimenti_slope_y"),
    ]

    out = df.copy()

    for base, outcol in specs:
        ycols = [f"{base}_y{i}" for i in range(1, 6) if f"{base}_y{i}" in out.columns]
        if not ycols:
            continue

        mat = pd.concat([_coerce_float_series(out[c]) for c in ycols], axis=1)

        def _row_slope(row):
            return _linear_slope_indexed(row.values.tolist())

        out[outcol] = mat.apply(_row_slope, axis=1)

    return out


def _add_prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df


def _dedup_block_one_row_per_school(block: pd.DataFrame) -> pd.DataFrame:
    if block is None or block.empty:
        return block
    if "CODICE_SCUOLA" not in block.columns:
        return block
    # canonicalizza ancora per sicurezza
    block = block.copy()
    block["CODICE_SCUOLA"] = block["CODICE_SCUOLA"].map(canon_mecc)
    block = block[block["CODICE_SCUOLA"].ne("")]
    # se ci sono duplicati, tieni la prima riga (dopo sorting stabile)
    return block.drop_duplicates("CODICE_SCUOLA", keep="first")


def _collapse_duplicates(wide: pd.DataFrame) -> pd.DataFrame:
    if wide["CODICE_SCUOLA"].duplicated().any():
        def pick_first(series):
            # preferisce valori non vuoti e non NaN
            for v in series:
                if pd.isna(v):
                    continue
                if isinstance(v, str) and v.strip() == "":
                    continue
                return v
            return series.iloc[0]
        wide = wide.groupby("CODICE_SCUOLA", sort=False, as_index=False).agg(pick_first)
    return wide


# Regole voti (memoria):
# ESAME: LOW={60,61-70}; TOP={91-99,100,100 e lode}; mean midpoint, Lode=101
EXAM_MIDS = {"60": 60.0, "61_70": 65.5, "71_80": 75.5, "81_90": 85.5, "91_99": 95.0, "100": 100.0, "100_e_lode": 101.0}
EXAM_LOW = {"60", "61_70"}
EXAM_TOP = {"91_99", "100", "100_e_lode"}

# MEDIE: LOW={6,7}; TOP={10,10 e lode}; mean midpoint, Lode=11
MID_MIDS = {"6": 6.0, "7": 7.0, "8": 8.0, "9": 9.0, "10": 10.0, "10_e_lode": 11.0}
MID_LOW = {"6", "7"}
MID_TOP = {"10", "10_e_lode"}


def _vote_features(d: pd.DataFrame, kind: str) -> pd.DataFrame:
    dd = d.copy()
    dd = dd[dd["value_num"].notna()]
    if dd.empty:
        return pd.DataFrame()
    dd["bin"] = dd["category"].map(lambda x: slugify(x)[:40])

    if kind == "exam":
        mids, low_bins, top_bins = EXAM_MIDS, EXAM_LOW, EXAM_TOP
    else:
        mids, low_bins, top_bins = MID_MIDS, MID_LOW, MID_TOP

    dd = dd[dd["bin"].isin(set(mids) | low_bins | top_bins)]
    if dd.empty:
        return pd.DataFrame()

    def _agg(g):
        w = g["value_num"].astype(float)
        s = float(w.sum())
        if s == 0.0:
            return pd.Series({"share_top_pct": np.nan, "share_low_pct": np.nan, "mean": np.nan})
        top = float(w[g["bin"].isin(top_bins)].sum() / s * 100.0)
        low = float(w[g["bin"].isin(low_bins)].sum() / s * 100.0)
        gm = g[g["bin"].isin(mids)].copy()
        if gm.empty:
            mean = np.nan
        else:
            ww = gm["value_num"].astype(float)
            xx = gm["bin"].map(mids).astype(float)
            denom = float(ww.sum())
            mean = float((ww * xx).sum() / denom) if denom else np.nan
        return pd.Series({"share_top_pct": top, "share_low_pct": low, "mean": mean})

    out = dd.groupby("CODICE_SCUOLA", sort=False).apply(_agg).reset_index()
    return out


def load_snai_xlsx(fp: Path) -> pd.DataFrame:
    sn = pd.read_excel(fp)
    if "COMUNE" not in sn.columns or "SNAI_2020" not in sn.columns:
        raise ValueError("SNAI: colonne richieste non trovate (COMUNE, SNAI_2020)")
    out = pd.DataFrame()
    out["__comune_key"] = sn["COMUNE"].map(_norm_comune_key)
    out["snai_classificazione"] = sn["SNAI_2020"].astype(str).str.strip()
    out = out[out["__comune_key"].ne("")].drop_duplicates("__comune_key", keep="first")
    return out[["__comune_key", "snai_classificazione"]]




def load_istat_comuni_xlsx(fp: Path) -> pd.DataFrame:
    """Carica elenco comuni ISTAT e prepara chiave di join.
    Richiede colonne:
      - Denominazione in italiano
      - Denominazione Regione
      - Ripartizione geografica
    Se presente, usa anche 'Sigla automobilistica' per disambiguare omonimi.
    """
    df = pd.read_excel(fp)

    # colonne attese (tolleranza su maiuscole/minuscole)
    cols = {c.lower(): c for c in df.columns}
    c_com = cols.get("denominazione in italiano") or cols.get("denominazione in italiano ")
    c_reg = cols.get("denominazione regione")
    c_rip = cols.get("ripartizione geografica")
    c_sig = cols.get("sigla automobilistica") or cols.get("sigla")

    missing = [x for x, c in [("Denominazione in italiano", c_com), ("Denominazione Regione", c_reg), ("Ripartizione geografica", c_rip)] if c is None]
    if missing:
        raise ValueError(f"ISTAT comuni: colonne richieste mancanti: {', '.join(missing)}")

    out = pd.DataFrame()
    out["__comune_key"] = df[c_com].map(_norm_place_key)
    out["__prov_key"] = df[c_sig].astype(str).str.strip().str.upper() if c_sig else ""
    out["regione"] = df[c_reg].astype(str).str.strip()
    out["macroarea"] = df[c_rip].astype(str).str.strip()

    out = out[out["__comune_key"].ne("")].copy()

    # dedup: se esiste sigla provincia, preferisci le righe con sigla valorizzata
    if c_sig:
        out["__has_prov"] = out["__prov_key"].ne("")
        out = out.sort_values(["__comune_key", "__has_prov"], ascending=[True, False]).drop_duplicates(
            ["__comune_key", "__prov_key"], keep="first"
        )
        out = out.drop(columns=["__has_prov"])
    else:
        out = out.drop_duplicates(["__comune_key"], keep="first")

    return out


def merge_regione_macroarea(wide: pd.DataFrame, istat_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggiunge 'regione' e 'macroarea' al wide.
    Match primario: (provincia, comune) se entrambe disponibili.
    Fallback: solo comune.
    Ritorna anche un dataframe di righe non matchate (per debug).
    """
    w = wide.copy()

    if "comune" not in w.columns:
        # niente da fare
        w["regione"] = ""
        w["macroarea"] = ""
        return w, w.iloc[0:0].copy()

    w["__comune_key"] = w["comune"].map(_norm_place_key)
    w["__prov_key"] = w["provincia"].astype(str).str.strip().str.upper() if "provincia" in w.columns else ""

    ist = istat_df.copy()
    ist["__comune_key"] = ist["__comune_key"].map(_norm_place_key)
    ist["__prov_key"] = ist["__prov_key"].astype(str).str.strip().str.upper()

    # 1) join su (comune, provincia) quando provincia valorizzata sia in wide che in ISTAT
    j1 = w.merge(
        ist[ist["__prov_key"].ne("")][["__comune_key", "__prov_key", "regione", "macroarea"]],
        on=["__comune_key", "__prov_key"],
        how="left",
        suffixes=("", "_istat"),
    )

    # 2) fallback: join su comune solo
    miss = j1["regione"].isna() | (j1["regione"].astype(str).str.strip() == "")
    if miss.any():
        j2 = w[miss].merge(
            ist[["__comune_key", "regione", "macroarea"]].drop_duplicates(["__comune_key"], keep="first"),
            on="__comune_key",
            how="left",
        )
        j1.loc[miss, "regione"] = j2["regione"].values
        j1.loc[miss, "macroarea"] = j2["macroarea"].values

    j1["regione"] = j1["regione"].fillna("").astype(str)
    j1["macroarea"] = j1["macroarea"].fillna("").astype(str)

    unmatched = j1[(j1["__comune_key"].ne("")) & (j1["regione"].astype(str).str.strip() == "")][
        ["CODICE_SCUOLA", "comune"] + (["provincia"] if "provincia" in j1.columns else [])
    ].drop_duplicates()

    # cleanup helper cols
    j1 = j1.drop(columns=["__comune_key", "__prov_key"], errors="ignore")

    # posizionamento: dopo provincia/cap se presenti
    cols = list(j1.columns)
    for c in ["regione", "macroarea"]:
        if c in cols:
            cols.remove(c)
    insert_after = None
    for anchor in ["provincia", "cap", "comune"]:
        if anchor in cols:
            insert_after = anchor
            break
    if insert_after:
        i = cols.index(insert_after) + 1
        cols = cols[:i] + ["regione", "macroarea"] + cols[i:]
    else:
        cols = ["regione", "macroarea"] + cols

    j1 = j1[cols]
    return j1, unmatched


def _snv_make_colname_22a1(prefix: str, a1: str, a2: str, a3: str, misura: str) -> str:
    # misura: "punteggio_medio" or "diff_escs"
    parts = [prefix.rstrip("_"), slugify(a1), slugify(a2), slugify(a3), slugify(misura)]
    parts = [p for p in parts if p]
    return "_".join(parts) + "_"


def _snv_make_colname_22b2(prefix: str, b1: str, b2: str, b3: str, b4: str, b5: str) -> str:
    parts = [prefix.rstrip("_"), slugify(b1), slugify(b2), slugify(b3), slugify(b4), slugify(b5)]
    parts = [p for p in parts if p]
    return "_".join(parts) + "_"



def _snv_rename_columns_post(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Compat: normalizza intestazioni colonne SNV wide (solo sostituzioni stringa)."""
    if wide_df is None or wide_df.empty:
        return wide_df
    repl = [
        ("scuola_secondaria_di_ii_grado", "superiori"),
        ("scuola_secondaria_di_i_grado", "medie"),
        ("dentro_le_classi_", "dentroleclassi"),
        ("tra_le_classi_", "traleclassi"),
        ("classi_seconde", "classiseconde"),
        ("classi_quinte_ultimo_anno", "classiquinteultimoanno"),
        ("inglese_reading", "inglesereading"),
        ("inglese_listening", "ingleselistening"),
        ("licei_scientifici_classici_e_linguistici", "liceiscientificiclassicilinguistici"),
        ("istituto_nel_suo_complesso", "istituto"),
        ("scuola_primaria", "primaria"),
        ("altri_licei_diversi_da_scientifici_classici_e_linguistici", "altrilicei"),
    ]
    new_cols = []
    for c in wide_df.columns:
        nc = c
        for old, new in repl:
            nc = nc.replace(old, new)
        new_cols.append(nc)
    out = wide_df.copy()
    out.columns = new_cols
    return out

def build_snv_wide_from_clean_csv(
    snv_clean_csv: Path,
    chunksize: int,
    prefix_22a1: str = "80_rav_22a1_invalsipunteggiomedio_",
    prefix_22b2: str = "80_rav_22b2_variabilitapunteggi_",
) -> pd.DataFrame:
    """Build a school-level wide table from the already-cleaned SNV long CSV.

    Expected columns (strings):
    - CODICE_SCUOLA, DESCRITTORE, COLONNA, VALORE
    - SPLIT_A1..SPLIT_A3 for 2.2.a.1
    - SPLIT_B1..SPLIT_B5 for 2.2.b.2
    """
    if not snv_clean_csv.exists():
        raise FileNotFoundError(str(snv_clean_csv))

    def _parse_val_to_float(x: object) -> float:
        """Parse SNV values like '230,5', '12,5%', 'n.d.' into float (NaN if missing)."""
        if x is None:
            return float("nan")
        s = str(x).strip()
        if not s:
            return float("nan")
        sl = s.lower()
        if sl in {"n.d.", "nd", "nan", "none"}:
            return float("nan")
        s = s.replace("\u00a0", " ").strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return float("nan")

    def _combine_wide(acc_df: Optional[pd.DataFrame], new_df: pd.DataFrame) -> pd.DataFrame:
        """Combine chunk-level wide frames without creating _x/_y duplicate columns.

        Uses CODICE_SCUOLA as index and fills missing values from new_df.
        If both have a non-null value, keeps the existing (acc) value.
        """
        if acc_df is None:
            return new_df
        a = acc_df.set_index("CODICE_SCUOLA")
        b = new_df.set_index("CODICE_SCUOLA")
        out = a.combine_first(b).reset_index()
        return out

    acc: Optional[pd.DataFrame] = None

    for chunk in safe_read_csv_chunks(snv_clean_csv, chunksize=chunksize):
        if chunk is None or chunk.empty:
            continue

        # force string
        for c in chunk.columns:
            chunk[c] = chunk[c].astype(str)

        # canonical key + minimal checks
        for needed in ["CODICE_SCUOLA", "DESCRITTORE", "VALORE"]:
            if needed not in chunk.columns:
                raise ValueError(f"SNV cleaned CSV missing {needed}")

        chunk["CODICE_SCUOLA"] = chunk["CODICE_SCUOLA"].map(canon_mecc)
        chunk = chunk[chunk["CODICE_SCUOLA"].ne("")]

        # --- 2.2.a.1 ---
        if "COLONNA" not in chunk.columns:
            raise ValueError("SNV cleaned CSV missing COLONNA (required for 2.2.a.1)")
        a = chunk[chunk["DESCRITTORE"].eq("2.2.a.1")].copy()
        if not a.empty:
            for needed in ["SPLIT_A1", "SPLIT_A2", "SPLIT_A3"]:
                if needed not in a.columns:
                    raise ValueError(f"SNV cleaned CSV missing {needed} (required for 2.2.a.1)")
            col_norm = a["COLONNA"].astype(str).str.strip().str.lower()

            misura = pd.Series([""] * len(a), index=a.index, dtype="object")
            misura.loc[col_norm.eq("punteggio medio (1)")] = "punteggio_medio"
            misura.loc[col_norm.eq("diff. escs (2)")] = "diff_escs"

            a = a[misura.ne("")]
            if not a.empty:
                misura = misura.loc[a.index]
                a["__colname"] = [
                    _snv_make_colname_22a1(prefix_22a1, x1, x2, x3, m_)
                    for x1, x2, x3, m_ in zip(a["SPLIT_A1"], a["SPLIT_A2"], a["SPLIT_A3"], misura)
                ]
                a2 = a[["CODICE_SCUOLA", "__colname", "VALORE"]].copy()
                a2["VALORE"] = a2["VALORE"].map(_parse_val_to_float)

                a_piv = a2.pivot_table(
                    index="CODICE_SCUOLA",
                    columns="__colname",
                    values="VALORE",
                    aggfunc="first",
                ).reset_index()

                acc = _combine_wide(acc, a_piv)

        # --- 2.2.b.2 ---
        b = chunk[chunk["DESCRITTORE"].eq("2.2.b.2")].copy()
        if not b.empty:
            for needed in ["SPLIT_B1", "SPLIT_B2", "SPLIT_B3", "SPLIT_B4", "SPLIT_B5"]:
                if needed not in b.columns:
                    raise ValueError(f"SNV cleaned CSV missing {needed} (required for 2.2.b.2)")
            b["__colname"] = [
                _snv_make_colname_22b2(prefix_22b2, x1, x2, x3, x4, x5)
                for x1, x2, x3, x4, x5 in zip(
                    b["SPLIT_B1"], b["SPLIT_B2"], b["SPLIT_B3"], b["SPLIT_B4"], b["SPLIT_B5"]
                )
            ]
            b2 = b[["CODICE_SCUOLA", "__colname", "VALORE"]].copy()
            b2["VALORE"] = b2["VALORE"].map(_parse_val_to_float)

            b_piv = b2.pivot_table(
                index="CODICE_SCUOLA",
                columns="__colname",
                values="VALORE",
                aggfunc="first",
            ).reset_index()

            acc = _combine_wide(acc, b_piv)

    if acc is None:
        return pd.DataFrame({"CODICE_SCUOLA": []})

    # ensure unique columns
    acc = acc.loc[:, ~acc.columns.duplicated()]

    acc = _snv_rename_columns_post(acc)

    return acc

def build_wide(long_df: pd.DataFrame, anagrafica_df: Optional[pd.DataFrame], sic_studenti_df: Optional[pd.DataFrame], snai_df: Optional[pd.DataFrame], snv_wide_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = long_df.copy()
    df["CODICE_SCUOLA"] = df["CODICE_SCUOLA"].map(canon_mecc)
    df = df[df["CODICE_SCUOLA"].ne("")]
    for c in ["NOME_SCUOLA", "endpoint_key", "category", "level"]:
        df[c] = df[c].astype(str).str.strip()

    base = df[["CODICE_SCUOLA", "NOME_SCUOLA"]].drop_duplicates("CODICE_SCUOLA")
    wide = base.copy()

    # prepara anagrafica indicizzata
    ana_cols_order: List[str] = []
    sic_cols_order: List[str] = []
    if anagrafica_df is not None and not anagrafica_df.empty:
        ana = anagrafica_df.copy()
        ana["CODICE_SCUOLA"] = ana["CODICE_SCUOLA"].map(canon_mecc)
        ana = ana[ana["CODICE_SCUOLA"].ne("")].drop_duplicates("CODICE_SCUOLA", keep="first")
        # rinomina eventuali collisioni
        ren = {}
        for c in ana.columns:
            if c != "CODICE_SCUOLA" and c in wide.columns:
                ren[c] = f"{c}_ana"
        if ren:
            ana = ana.rename(columns=ren)
        ana_cols_order = [c for c in ana.columns if c != "CODICE_SCUOLA"]
        wide = wide.merge(ana, on="CODICE_SCUOLA", how="left")
        wide = _dedup_block_one_row_per_school(wide)


    # merge sic-studenti (dopo anagrafica, prima delle variabili)
    if sic_studenti_df is not None and not sic_studenti_df.empty:
        sic = sic_studenti_df.copy()
        sic["CODICE_SCUOLA"] = sic["CODICE_SCUOLA"].map(canon_mecc)
        sic = sic[sic["CODICE_SCUOLA"].ne("")].drop_duplicates("CODICE_SCUOLA", keep="first")

        # rimuovi sempre la colonna 'esito' (se presente), come da specifica
        drop_cols = [c for c in sic.columns if c.strip().lower() == "esito"]
        if drop_cols:
            sic = sic.drop(columns=drop_cols)

        ren = {}
        for c in sic.columns:
            if c != "CODICE_SCUOLA" and c in wide.columns:
                ren[c] = f"{c}_sicstu"
        if ren:
            sic = sic.rename(columns=ren)

        # ordine colonne sic-studenti (post-rinominazione), da posizionare subito dopo l'anagrafica
        sic_cols_order = [c for c in sic.columns if c != "CODICE_SCUOLA"]

        wide = wide.merge(sic, on="CODICE_SCUOLA", how="left")
        wide = _dedup_block_one_row_per_school(wide)

    # blocco anni + mean/slope
    def years_block(ep: str, label: str, prefix: str) -> pd.DataFrame:
        d = df[(df["endpoint_key"] == ep) & (df["value_num"].notna())].copy()
        if d.empty:
            return pd.DataFrame()
        d["y"] = d["category"].map(_year_from_category)
        d = d[d["y"].notna()]
        if d.empty:
            return pd.DataFrame()

        pv = (
            d.groupby(["CODICE_SCUOLA", "y"], as_index=False)["value_num"].first()
             .pivot(index="CODICE_SCUOLA", columns="y", values="value_num")
        )
        pv.columns = [f"{label}_y{int(c)}" for c in pv.columns]

        def _derive(g):
            yy = g["y"].astype(int).to_numpy()
            vv = g["value_num"].astype(float).to_numpy()
            m_all = float(np.nanmean(vv)) if len(vv) else np.nan
            m_12 = float(np.nanmean(vv[(yy == 1) | (yy == 2)])) if np.any((yy == 1) | (yy == 2)) else np.nan
            yy_s = yy[(yy >= 1) & (yy <= 5)]
            vv_s = vv[(yy >= 1) & (yy <= 5)]
            sl = _linear_slope(yy_s, vv_s)
            return pd.Series({f"{label}_mean_y": m_all, f"{label}_mean_y1y2": m_12, f"{label}_slope_y": sl})

        der = d.groupby("CODICE_SCUOLA", sort=False).apply(_derive)
        out = pv.join(der, how="outer")
        out = _add_prefix_cols(out, prefix).reset_index()
        return _dedup_block_one_row_per_school(out)

    for block in [
        years_block("abbandoni_24_25", "abbandoni", "10"),
        years_block("studenti_ripetenti", "ripetenti", "11"),
        years_block("trasferimenti_24_25", "trasferimenti", "12"),
    ]:
        if block is not None and not block.empty:
            wide = wide.merge(block, on="CODICE_SCUOLA", how="left")
            wide = _dedup_block_one_row_per_school(wide)

    # assenze: somma su scope "Scuola"
    def sum_scuola(ep: str, outname: str, prefix: str) -> pd.DataFrame:
        d = df[(df["endpoint_key"] == ep) & (df["value_num"].notna())].copy()
        if d.empty:
            return pd.DataFrame()
        d = d[_is_scuola_scope(d)]
        if d.empty:
            return pd.DataFrame()
        s = d.groupby("CODICE_SCUOLA", sort=False)["value_num"].sum().to_frame(outname)
        s = _add_prefix_cols(s, prefix).reset_index()
        return _dedup_block_one_row_per_school(s)

    for block in [
        sum_scuola("assenze_docenti", "assenze_docenti_sum", "13"),
        sum_scuola("assenze_ata", "assenze_ata_sum", "14"),
    ]:
        if block is not None and not block.empty:
            wide = wide.merge(block, on="CODICE_SCUOLA", how="left")
            wide = _dedup_block_one_row_per_school(wide)

    # entrate_fonti_finanziamento: tutte categorie, tutte _pct
    d = df[(df["endpoint_key"] == "entrate_fonti_finanziamento") & (df["value_num"].notna())].copy()
    if not d.empty:
        d = d[_is_scuola_scope(d)] if _is_scuola_scope(d).any() else d
        d["cat"] = d["category"].map(lambda x: slugify(x)[:50] if str(x).strip() else "cat_empty")
        pv = (
            d.groupby(["CODICE_SCUOLA", "cat"], as_index=False)["value_num"].first()
             .pivot(index="CODICE_SCUOLA", columns="cat", values="value_num")
        )
        pv.columns = [f"entrate_{c}_pct" for c in pv.columns]
        pv = _add_prefix_cols(pv, "20").reset_index()
        pv = _dedup_block_one_row_per_school(pv)
        wide = wide.merge(pv, on="CODICE_SCUOLA", how="left")
        wide = _dedup_block_one_row_per_school(wide)

    # esiti giugno/settembre: _pct + tasso_non_ammessi_pct
    def esiti(ep: str, label: str, prefix: str) -> pd.DataFrame:
        d = df[(df["endpoint_key"] == ep) & (df["value_num"].notna())].copy()
        if d.empty:
            return pd.DataFrame()
        d = d[_is_scuola_scope(d)] if _is_scuola_scope(d).any() else d
        d["cat"] = d["category"].map(lambda x: slugify(x)[:50] if str(x).strip() else "cat_empty")
        d = d[d["cat"].isin({"ammessi", "non_ammessi", "sospesi"})]
        if d.empty:
            return pd.DataFrame()
        pv = (
            d.groupby(["CODICE_SCUOLA", "cat"], as_index=False)["value_num"].first()
             .pivot(index="CODICE_SCUOLA", columns="cat", values="value_num")
        )
        pv = pv.rename(columns={
            "ammessi": f"{label}_ammessi_pct",
            "non_ammessi": f"{label}_non_ammessi_pct",
            "sospesi": f"{label}_sospesi_pct",
        })
        if f"{label}_ammessi_pct" in pv.columns:
            pv[f"{label}_tasso_non_ammessi_pct"] = 100.0 - pd.to_numeric(pv[f"{label}_ammessi_pct"], errors="coerce")
        pv = _add_prefix_cols(pv, prefix).reset_index()
        return _dedup_block_one_row_per_school(pv)

    for block in [
        esiti("esiti_giugno_24_25", "esiti_giugno", "21"),
        esiti("esiti_giugno_settembre_24_25", "esiti_settembre", "22"),
    ]:
        if block is not None and not block.empty:
            wide = wide.merge(block, on="CODICE_SCUOLA", how="left")
            wide = _dedup_block_one_row_per_school(wide)

    # docenti_pensionati/docenti_trasferiti: *_pct
    def single(ep: str, outname: str) -> pd.DataFrame:
        d = df[(df["endpoint_key"] == ep) & (df["value_num"].notna())].copy()
        if d.empty:
            return pd.DataFrame()
        d = d[_is_scuola_scope(d)] if _is_scuola_scope(d).any() else d
        s = d.groupby("CODICE_SCUOLA", sort=False)["value_num"].first().to_frame(outname)
        s = _add_prefix_cols(s, "30").reset_index()
        return _dedup_block_one_row_per_school(s)

    for block in [
        single("docenti_pensionati", "docenti_pensionati_pct"),
        single("docenti_trasferiti", "docenti_trasferiti_pct"),
    ]:
        if block is not None and not block.empty:
            wide = wide.merge(block, on="CODICE_SCUOLA", how="left")
            wide = _dedup_block_one_row_per_school(wide)

    # voti esame: top/low/mean (mean senza _pct)
    d = df[(df["endpoint_key"] == "distribuzione_votazioni_esame_24_25") & (df["value_num"].notna())].copy()
    if not d.empty:
        d = d[_is_scuola_scope(d)] if _is_scuola_scope(d).any() else d
        f = _vote_features(d[["CODICE_SCUOLA", "category", "value_num"]], kind="exam")
        if not f.empty:
            f = f.rename(columns={
                "share_top_pct": "esame_voto_share_top_pct",
                "share_low_pct": "esame_voto_share_low_pct",
                "mean": "esame_voto_share_mean",
            }).set_index("CODICE_SCUOLA")
            f = _add_prefix_cols(f, "40").reset_index()
            f = _dedup_block_one_row_per_school(f)
            wide = wide.merge(f, on="CODICE_SCUOLA", how="left")
            wide = _dedup_block_one_row_per_school(wide)

    # voti medie (rav_24c5): top/low/mean (mean senza _pct)
    d = df[(df["endpoint_key"] == "rav_24c5") & (df["value_num"].notna())].copy()
    if not d.empty:
        d = d[_is_scuola_scope(d)] if _is_scuola_scope(d).any() else d
        f = _vote_features(d[["CODICE_SCUOLA", "category", "value_num"]], kind="middle")
        if not f.empty:
            f = f.rename(columns={
                "share_top_pct": "medie_voto_share_top_pct",
                "share_low_pct": "medie_voto_share_low_pct",
                "mean": "medie_voto_share_mean",
            }).set_index("CODICE_SCUOLA")
            f = _add_prefix_cols(f, "50").reset_index()
            f = _dedup_block_one_row_per_school(f)
            wide = wide.merge(f, on="CODICE_SCUOLA", how="left")
            wide = _dedup_block_one_row_per_school(wide)

    # SNAI: match COMUNE (file SNAI) con wide.comune
    if snai_df is not None and not snai_df.empty and "comune" in wide.columns:
        key = pd.DataFrame({"CODICE_SCUOLA": wide["CODICE_SCUOLA"], "__comune_key": wide["comune"].map(_norm_comune_key)})
        m = key.merge(snai_df, on="__comune_key", how="left")
        m["snai_classificazione"] = m["snai_classificazione"].fillna("A - Polo o B - Polo intercomunale")
        sn = m[["CODICE_SCUOLA", "snai_classificazione"]].set_index("CODICE_SCUOLA")
        sn = _add_prefix_cols(sn, "70").reset_index()
        sn = _dedup_block_one_row_per_school(sn)
        wide = wide.merge(sn, on="CODICE_SCUOLA", how="left")
        wide = _dedup_block_one_row_per_school(wide)
    # merge SNV/RAV (already-wide)
    if snv_wide_df is not None and not snv_wide_df.empty:
        snv = snv_wide_df.copy()
        snv["CODICE_SCUOLA"] = snv["CODICE_SCUOLA"].map(canon_mecc)
        snv = snv[snv["CODICE_SCUOLA"].ne("")]
        snv = _dedup_block_one_row_per_school(snv)

        wide = wide.merge(snv, on="CODICE_SCUOLA", how="left")
        wide = _dedup_block_one_row_per_school(wide)

    
        # SNV derived elaborations
        wide = _add_snv_means(wide)
    # dedup finale
    wide["CODICE_SCUOLA"] = wide["CODICE_SCUOLA"].map(canon_mecc)
    wide = _collapse_duplicates(wide)

    # forza slope serie SIC su y1..y5
    wide = _add_forced_slopes(wide)

    # mark elaborazioni columns with leading '*'
    wide = _star_computed_columns(wide)

    # ordering: CODICE + anagrafica cols + NOME + resto (per prefisso)
    head = ["CODICE_SCUOLA"]
    ana_cols = [c for c in ana_cols_order if c in wide.columns]
    sic_cols = [c for c in sic_cols_order if c in wide.columns]
    mid = ["NOME_SCUOLA"] if "NOME_SCUOLA" in wide.columns else []

    rest = [c for c in wide.columns if c not in set(head + ana_cols + sic_cols + mid)]

    def _k(c):
        m = re.match(r"^\*?(\d{2})_", c)
        if m:
            return (0, int(m.group(1)), c)
        return (1, 999, c)

    wide = wide[head + ana_cols + sic_cols + mid + sorted(rest, key=_k)]
    return wide



# ----------------------------
# Main
# ----------------------------

import json
from datetime import datetime
def _norm_comune(x: str) -> str:
    """Normalize comune names for joins (uppercase, no accents/punct, compact spaces)."""
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    # normalize unicode accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    # standardize apostrophes and separators
    s = s.replace("’", "'").replace("`", "'")
    s = s.replace("-", " ").replace("/", " ")
    # remove punctuation except spaces
    s = re.sub(r"[^A-Z0-9 ']+", " ", s)
    # collapse apostrophes as separators (D'IMPERIA -> D IMPERIA)
    s = s.replace("'", " ")
    # normalize common prefixes
    s = re.sub(r"\bSANT\s+", "SANT ", s)
    s = re.sub(r"\bSAN\s+", "SAN ", s)
    s = re.sub(r"\bSANTA\s+", "SANTA ", s)
    # compact whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _write_manifest(manifest_path: Path, data: dict) -> None:
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_latest_symlink(releases_root: Path, latest_target: Path) -> None:
    """Create/replace a symlink 'latest' inside releases_root pointing to latest_target.
    Uses a relative target (directory name) for portability.
    """
    releases_root = Path(releases_root)
    latest_target = Path(latest_target)
    link = releases_root / "latest"

    # Safety: do not overwrite a real directory named 'latest'.
    if link.exists() and not link.is_symlink() and link.is_dir():
        raise RuntimeError(f"Refusing to overwrite directory: {link}")

    if link.is_symlink() or link.exists():
        link.unlink(missing_ok=True)

    # Relative symlink from releases_root -> latest_target.name
    os.symlink(latest_target.name, link)


def run_pipeline(obs_paths: List[Path], anagrafica_df: Optional[pd.DataFrame], sic_studenti_df: Optional[pd.DataFrame], snai_df: Optional[pd.DataFrame], istat_df: Optional[pd.DataFrame], out_base: Path, args: argparse.Namespace, package_name: str) -> None:
    out_base.mkdir(parents=True, exist_ok=True)
    out_long = out_base / "long"
    out_long.mkdir(parents=True, exist_ok=True)
    # pulizia
    for p in out_long.glob("part-*.parquet"):
        p.unlink()

    part = 0
    chunks = 0
    for fp in obs_paths:
        for chunk in safe_read_csv_chunks(fp, chunksize=args.chunksize):
            chunks += 1
            norm = normalize_chunk(chunk)
            write_parquet_part(norm, out_long, part)
            part += 1
            if args.progress_every and chunks % args.progress_every == 0:
                print(f"[{package_name}] PROGRESS chunks={chunks} parts={part}")
    snv_wide_df: Optional[pd.DataFrame] = None
    if args.merge_snv:
        snv_clean_csv = Path(args.snv_clean_csv)
        print(f"[{package_name}] merge_snv=1 snv_clean_csv={snv_clean_csv}")
        snv_wide_df = build_snv_wide_from_clean_csv(snv_clean_csv, chunksize=args.chunksize)


    long_df = read_long_dataset(out_long)
    wide = build_wide(long_df, anagrafica_df=anagrafica_df, sic_studenti_df=sic_studenti_df, snai_df=snai_df, snv_wide_df=snv_wide_df)
    if args.merge_istat_comuni and istat_df is not None:
        wide, unmatched = merge_regione_macroarea(wide, istat_df)
        # debug: salva elenco comuni non matchati
        try:
            unmatched.to_csv(out_base / "unmatched_comuni_istat.csv", index=False, encoding="utf-8")
        except Exception:
            pass
        wide = sanitize_for_csv(wide)

    out_csv = out_base / "wide.csv"
    out_parq = out_base / "wide.parquet"
    wide.to_csv(out_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL, escapechar="\\", lineterminator="\n")
    _require_pyarrow()
    pq.write_table(pa.Table.from_pandas(wide, preserve_index=False), out_parq, compression="zstd")

    manifest = {
        "package": package_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": [str(x) for x in args.inputs],
        "obs_files": [str(p) for p in obs_paths],
        "rows": int(len(wide)),
        "cols": int(len(wide.columns)),
        "long_parts": int(part),
        "chunks": int(chunks),
        "merge_anagrafica": (anagrafica_df is not None and not anagrafica_df.empty),
        "merge_sic_studenti": (sic_studenti_df is not None and not sic_studenti_df.empty),
        "merge_snai": (snai_df is not None and not snai_df.empty),
        "merge_istat_comuni": (istat_df is not None and not istat_df.empty),
        "istat_comuni_xlsx": str(args.istat_comuni_xlsx) if getattr(args, "merge_istat_comuni", False) else "",
        "istat_unmatched_rows": int((out_base / "unmatched_comuni_istat.csv").exists() and pd.read_csv(out_base / "unmatched_comuni_istat.csv", dtype=str).shape[0] or 0),
    }
    _write_manifest(out_base / "manifest.json", manifest)
    print(f"[{package_name}] DONE rows={len(wide)} cols={len(wide.columns)} parts={part}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--no-anagrafica", action="store_true")
    ap.add_argument("--anagrafica-glob", default="anagrafica*.csv")

    # Release mode
    ap.add_argument("--release", action="store_true", help="Scrive una release in releases/<timestamp>/ e NON usa --out-*")
    ap.add_argument("--release-root", default="releases")
    ap.add_argument("--by-area", action="store_true", help="Crea anche areas/<AREA>/... oltre ad ALL")

    # Optional merges
    ap.add_argument("--merge-snai", action="store_true")
    ap.add_argument("--merge-istat-comuni", action="store_true")
    ap.add_argument("--istat-comuni-xlsx", default=str(Path("./utils/Elenco-comuni-italiani.xlsx")))
    ap.add_argument("--snai-xlsx", default=str(Path("./utils/SNAI.xlsx")))
    ap.add_argument("--merge-sic-studenti", action="store_true")
    ap.add_argument("--sic-studenti-csv", default=str(Path("./apps/_sic_scraper_vw/sic_scraper_vw_out.csv")))
    ap.add_argument("--merge-snv", action="store_true",
                help="Merge SNV/RAV (INVALSI) wide computed from an already-cleaned long CSV.")
    ap.add_argument("--snv-clean-csv", default=str(Path("./apps/_rav_cleaner/observations_clean.csv")),
                help="Path to cleaned SNV long CSV (output of rav_cleaner).")

    # Classic outputs (only when not --release)
    ap.add_argument("--out-long-dir", default="")
    ap.add_argument("--out-wide-csv", default="")
    ap.add_argument("--out-wide-parquet", default="")

    args = ap.parse_args()

    obs_paths = find_obs_csv_paths(args.inputs)
    if not obs_paths:
        print("Nessun observations_semantic.csv trovato", file=sys.stderr)
        sys.exit(2)

    ana_paths = [] if args.no_anagrafica else find_anagrafica_csv_paths(args.inputs, args.anagrafica_glob)
    anagrafica_df = load_anagrafica_df(ana_paths) if ana_paths else None

    sic_studenti_df = None
    if args.merge_sic_studenti:
        fp = Path(args.sic_studenti_csv)
        if fp.exists():
            sic_studenti_df = load_sic_studenti_csv(fp)
        else:
            print(f"sic-studenti: file non trovato: {fp}", file=sys.stderr)

    snai_df = None
    if args.merge_snai:
        fp = Path(args.snai_xlsx)
        if fp.exists():
            snai_df = load_snai_xlsx(fp)
        else:
            print(f"SNAI: file non trovato: {fp}", file=sys.stderr)

    istat_df = None
    if args.merge_istat_comuni:
        fp = Path(args.istat_comuni_xlsx)
        if fp.exists():
            istat_df = load_istat_comuni_xlsx(fp)
        else:
            print(f"ISTAT comuni: file non trovato: {fp}", file=sys.stderr)

    if args.release:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(args.release_root) / ts
        # ALL sempre
        run_pipeline(obs_paths, anagrafica_df, sic_studenti_df, snai_df, istat_df, base / "ALL", args, "ALL")
        if args.by_area:
            by_area = group_paths_by_area(obs_paths)
            for area, fps in sorted(by_area.items()):
                run_pipeline(fps, anagrafica_df, sic_studenti_df, snai_df, istat_df, base / "areas" / area, args, area)
        # manifest root
        root_manifest = {
            "release_id": ts,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "packages": ["ALL"] + ([f"areas/{a}" for a in sorted(group_paths_by_area(obs_paths).keys())] if args.by_area else []),
        }
        _write_manifest(base / "manifest.json", root_manifest)
        print(f"[RELEASE] {base}")

        # Create/replace '.../latest' symlink.
        _update_latest_symlink(Path(args.release_root), base)
        return

    # classic mode
    if not (args.out_long_dir and args.out_wide_csv and args.out_wide_parquet):
        print("Modalità non-release: servono --out-long-dir, --out-wide-csv, --out-wide-parquet", file=sys.stderr)
        sys.exit(2)
    out_base = Path(args.out_wide_csv).resolve().parent
    # mantieni compatibilità: scrive nei percorsi indicati
    out_long = Path(args.out_long_dir)
    out_long.mkdir(parents=True, exist_ok=True)
    for p in out_long.glob("part-*.parquet"):
        p.unlink()
    part = 0
    for fp in obs_paths:
        for chunk in safe_read_csv_chunks(fp, chunksize=args.chunksize):
            norm = normalize_chunk(chunk)
            write_parquet_part(norm, out_long, part)
            part += 1
    snv_wide_df: Optional[pd.DataFrame] = None
    if args.merge_snv:
        snv_clean_csv = Path(args.snv_clean_csv)
        print(f"[{package_name}] merge_snv=1 snv_clean_csv={snv_clean_csv}")
        snv_wide_df = build_snv_wide_from_clean_csv(snv_clean_csv, chunksize=args.chunksize)


    long_df = read_long_dataset(out_long)
    wide = build_wide(long_df, anagrafica_df=anagrafica_df, sic_studenti_df=sic_studenti_df, snai_df=snai_df, snv_wide_df=snv_wide_df)
    if args.merge_istat_comuni and istat_df is not None:
        wide, unmatched = merge_regione_macroarea(wide, istat_df)
        # debug: salva elenco comuni non matchati
        try:
            unmatched.to_csv(out_base / "unmatched_comuni_istat.csv", index=False, encoding="utf-8")
        except Exception:
            pass
        wide = sanitize_for_csv(wide)
    out_csv = Path(args.out_wide_csv)
    out_parq = Path(args.out_wide_parquet)
    wide.to_csv(out_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL, escapechar="\\", lineterminator="\n")
    _require_pyarrow()
    pq.write_table(pa.Table.from_pandas(wide, preserve_index=False), out_parq, compression="zstd")
    print(f"[DONE WIDE] csv={out_csv} parquet={out_parq} rows={len(wide)} cols={len(wide.columns)}")


if __name__ == "__main__":
    main()




def _merge_snai(df_wide: pd.DataFrame, snai_path: Path, comune_col_wide: str = "comune") -> pd.DataFrame:
    """Merge SNAI classification onto wide dataframe using COMUNE name match (normalized).

    Adds column: snai_classificazione (from SNAI_2020). If no match, defaults to 'A - Polo o B - Polo intercomunale'.
    """
    if snai_path is None or (not Path(snai_path).exists()):
        # still ensure column exists
        out = df_wide.copy()
        if "snai_classificazione" not in out.columns:
            out["snai_classificazione"] = "A - Polo o B - Polo intercomunale"
        return out

    snai_path = Path(snai_path)
    try:
        snai = pd.read_excel(snai_path, dtype=str)
    except Exception:
        # cannot read: fallback default
        out = df_wide.copy()
        if "snai_classificazione" not in out.columns:
            out["snai_classificazione"] = "A - Polo o B - Polo intercomunale"
        return out

    snai = snai.copy()
    # detect columns
    cols_l = {str(c).strip().lower(): c for c in snai.columns}
    c_com = cols_l.get("comune") or cols_l.get("denominazione_ita") or cols_l.get("comune_ita") or cols_l.get("nome_comune")
    c_cls = cols_l.get("snai_2020") or cols_l.get("classificazione snai") or cols_l.get("classificazione_snai") or cols_l.get("snai")
    if c_com is None or c_cls is None:
        out = df_wide.copy()
        if "snai_classificazione" not in out.columns:
            out["snai_classificazione"] = "A - Polo o B - Polo intercomunale"
        return out

    snai[c_com] = snai[c_com].astype(str)
    snai[c_cls] = snai[c_cls].astype(str)
    snai["_comune_norm"] = snai[c_com].map(_norm_comune)

    wide = df_wide.copy()
    if comune_col_wide not in wide.columns:
        # cannot match; just default
        wide["snai_classificazione"] = "A - Polo o B - Polo intercomunale"
        return wide

    wide["_comune_norm"] = wide[comune_col_wide].map(_norm_comune)
    # Deduplicate SNAI by normalized comune (keep first non-empty classification)
    snai_d = (
        snai.sort_values(by=[c_cls], na_position="last")
            .dropna(subset=["_comune_norm"])
            .drop_duplicates(subset=["_comune_norm"], keep="first")
        [["_comune_norm", c_cls]]
        .rename(columns={c_cls: "snai_classificazione"})
    )

    wide = wide.merge(snai_d, how="left", on="_comune_norm")
    wide["snai_classificazione"] = wide["snai_classificazione"].fillna("A - Polo o B - Polo intercomunale")
    wide = wide.drop(columns=["_comune_norm"], errors="ignore")
    return wide
