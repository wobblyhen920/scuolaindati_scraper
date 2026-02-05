#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SNV scraper asincrono (RAV/Scuola in Chiaro) con:
- output long atomico per INDEX + merge finale
- fallback (scuola -> istituto) + waitlist + backup codes
- parsing tabelle con header multi-riga
- estrazione "spunte"/checkbox/icone -> 1/0
- normalizzazioni "leggibili" (niente "_" al posto degli spazi, niente "|" nelle etichette)
- lettura robusta di CSV "sporchi" (delimiter variabile, quoting rotto, encoding non-utf8)

NOTE:
- NON fa regressioni né analisi: solo scraping e pulizia/normalizzazione label.

FIX IMPORTANTI (2026-01-31):
- Endpoint keys: supporto sia colonne uppercase che lowercase (CODICE_EP/url).
- Rimozione mismatch che azzerava gli endpoint (tutte scuole "fallite").
- Definizione MAX_RETRIES/RETRY_BACKOFF (o uso retry interno).
- salva_html_async: signature coerente con la chiamata e path per ep.
- Uniformato campo INDEX (non IDX) in tutte le righe long.
- Log su parse error (prima veniva silenziato).
"""

import asyncio
import aiohttp
import pandas as pd
import os
import csv
import re
import unidecode
import traceback
import argparse
import json
import random
import shutil
import hashlib
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Optional, Tuple, Any, Dict
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import logging
from io import StringIO

# ----------------------------
# Parsing "truth" for 3 descrittori (SNV/INVALSI)
# ----------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def _clean_text(el) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True))

def _bold_headings_with_classi_before(table, limit: int = 6) -> List[str]:
    """
    Raccoglie testi <p><b>...classi...</b></p> subito prima della tabella.
    Mantiene l'ordine naturale (dall'alto verso la tabella).
    """
    out = []
    steps = 0
    node = table
    while steps < limit:
        node = node.find_previous()
        if node is None:
            break
        steps += 1
        if getattr(node, "name", None) == "p":
            b = node.find("b")
            if b:
                txt = _clean_text(b)
                if "classi" in _norm(txt):
                    out.append(txt)
        if len(out) >= 4:
            break
    # fallback: alcuni HTML SNV/INVALSI mettono il titolo '...classi...' dentro la tabella (prima riga)
    if not out:
        for tr in table.find_all("tr", limit=3):
            cells = tr.find_all(["th","td"])
            if not cells:
                continue
            txt = _clean_text(cells[0])
            if "classi" in _norm(txt):
                out.append(txt)
                break
    return list(reversed(out))

def _has_img(cell) -> bool:
    """Ritorna True quando la cella segnala una 'spunta' (non solo quando contiene un <img> esplicito).
    SNV usa talvolta <img>, talvolta CSS background-image o svg.
    """
    if cell is None:
        return False

    # 1) img esplicito
    if cell.find("img") is not None:
        return True

    # 2) svg (alcune skin)
    if cell.find("svg") is not None:
        return True

    # 3) background-image in style
    style = (cell.get("style") or "").lower()
    if "url(" in style and ("background" in style or "background-image" in style):
        return True

    # 4) classi indicative
    cls = " ".join(cell.get("class") or []).lower()
    if any(k in cls for k in ("check", "spunta", "selected", "active", "on", "img")):
        return True

    # 5) input checked
    inp = cell.find("input")
    if inp is not None and (inp.get("checked") is not None):
        return True

    return False


# --- Parser robusto per 2.2.b.2 (variabilità)
def var_clean(s: str) -> str:
    return " ".join((s or "").split())

def var_has_classi(s: str) -> bool:
    return "classi" in (s or "").lower()

def var_int_attr(tag, name: str, default: int = 1) -> int:
    try:
        return int(tag.get(name) or default)
    except Exception:
        return default

def var_bold_p_headings_before(table, max_steps: int = 200) -> List[str]:
    """
    Raccoglie i <p> precedenti al table che contengono 'classi' e includono <b>/<strong>.
    """
    headings: List[str] = []
    sib = table.previous_sibling
    steps = 0
    while sib is not None and steps < max_steps:
        steps += 1
        if getattr(sib, "name", None) == "table":
            break
        if getattr(sib, "name", None) == "p":
            txt = var_clean(sib.get_text(" ", strip=True))
            if var_has_classi(txt) and (sib.find("b") is not None or sib.find("strong") is not None):
                headings.append(txt)
        sib = sib.previous_sibling

    headings = list(reversed(headings))
    seen = set()
    out: List[str] = []
    for h in headings:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out

def var_build_header_grid(table) -> Optional[Tuple[List[str], List[Any]]]:
    """
    Costruisce nomi colonne dai th, rispettando colspan/rowspan.
    Restituisce (colnames, header_trs).
    """
    header_trs: List[Any] = []
    for tr in table.find_all("tr"):
        if tr.find("td") is not None:
            break
        if tr.find("th") is not None:
            header_trs.append(tr)

    if not header_trs:
        return None

    ncols = 0
    for tr in header_trs:
        ths = tr.find_all("th", recursive=False) or tr.find_all("th")
        ncols = max(ncols, sum(var_int_attr(th, "colspan", 1) for th in ths))

    depth = len(header_trs)
    grid: List[List[List[str]]] = [[[] for _ in range(ncols)] for _ in range(depth)]
    occupied: List[List[int]] = [[0] * ncols for _ in range(depth)]

    for r, tr in enumerate(header_trs):
        ths = tr.find_all("th", recursive=False) or tr.find_all("th")
        c = 0
        for th in ths:
            while c < ncols and occupied[r][c]:
                c += 1
            txt = var_clean(th.get_text(" ", strip=True))
            cs = var_int_attr(th, "colspan", 1)
            rs = var_int_attr(th, "rowspan", 1)

            for rr in range(r, min(depth, r + rs)):
                for cc in range(c, min(ncols, c + cs)):
                    if txt:
                        grid[rr][cc].append(txt)
                    if rr != r:
                        occupied[rr][cc] = 1
            c += cs

    colnames: List[str] = []
    for c in range(ncols):
        parts: List[str] = []
        seen = set()
        for r in range(depth):
            for t in grid[r][c]:
                t = var_clean(t)
                if t and t not in seen:
                    parts.append(t)
                    seen.add(t)
        colnames.append(" | ".join(parts) if parts else f"col{c+1}")

    return colnames, header_trs

def var_extract_tables(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Estrae SOLO:
    - intestazioni col 'classi' (da <p> bold + blocchi di <th>)
    - la prima riga dati che contiene 'Situazione' (case-insensitive)
    per ogni tabella .tableDsc.
    """
    out: List[Dict[str, Any]] = []
    for table in soup.find_all("table", class_="tableDsc"):
        first_data = None
        for tr in table.find_all("tr"):
            tds = tr.find_all("td", recursive=False) or tr.find_all("td")
            if not tds:
                continue
            label = var_clean(tds[0].get_text(" ", strip=True))
            if re.search(r"\bsituazione\b", label, flags=re.I):
                first_data = tr
                break
        if first_data is None:
            continue

        built = var_build_header_grid(table)
        if not built:
            continue
        colnames, header_trs = built

        tds = first_data.find_all("td", recursive=False) or first_data.find_all("td")
        values = [var_clean(td.get_text(" ", strip=True)) for td in tds]

        if len(values) < len(colnames):
            values += [""] * (len(colnames) - len(values))
        elif len(values) > len(colnames):
            for i in range(len(colnames), len(values)):
                colnames.append(f"extra_{i+1}")

        row = dict(zip(colnames, values))

        header_texts: List[str] = []
        for tr in header_trs:
            txt = var_clean(tr.get_text(" ", strip=True))
            if var_has_classi(txt):
                header_texts.append(txt)

        headings: List[str] = []
        for h in var_bold_p_headings_before(table) + header_texts:
            if var_has_classi(h) and h not in headings:
                headings.append(h)

        out.append({"headings_classi": headings, "row_label": values[0] if values else "", "row": row})
    return out

def parse_ep_2_4_a_4(soup: BeautifulSoup) -> List[dict]:
    """
    INVALSI (endpoint 2.4.a.4):
    - per ogni tabella <table class="tableDsc"> estrae SOLO la prima riga dati
      in cui la prima colonna contiene 'classi' (case-insensitive)
    - colonne: Istituto/Plesso/Indirizzo/, Punteggio medio (1), Diff. ESCS (2)
    - conserva intestazioni generali in grassetto che contengono 'classi'
    """
    records: List[dict] = []
    tables = soup.select("table.tableDsc")
    for t in tables:
        heading_parts = _bold_headings_with_classi_before(t)
        heading = " | ".join([h for h in heading_parts if h])

        rows = t.find_all("tr")
        if not rows:
            continue

        header_idx = None
        header_texts = None
        for i, r in enumerate(rows):
            cells = r.find_all(["th", "td"])
            if not cells:
                continue
            txts = [_norm(_clean_text(c)) for c in cells]
            if any("istituto/plesso/indirizzo" in x for x in txts) and any("punteggio medio" in x for x in txts) and any("diff" in x and "escs" in x for x in txts):
                header_idx = i
                header_texts = txts
                break

        if header_idx is None or header_texts is None:
            continue

        def idx_of(sub: str):
            for j, h in enumerate(header_texts):
                if sub in h:
                    return j
            return None

        i_istituto = idx_of("istituto/plesso/indirizzo")
        i_punteggio = idx_of("punteggio medio")
        i_diff = None
        for j, h in enumerate(header_texts):
            if "diff" in h and "escs" in h:
                i_diff = j
                break

        if i_punteggio is None or i_diff is None:
            continue
        if i_istituto is None:
            i_istituto = 0

        for r in rows[header_idx + 1:]:
            cells = r.find_all(["td", "th"])
            if not cells:
                continue
            first = _clean_text(cells[0])
            if "classi" not in _norm(first):
                continue

            def safe(ix):
                if ix is None or ix < 0 or ix >= len(cells):
                    return ""
                return _clean_text(cells[ix])

            records.append({
                "heading": heading,
                "istituto_plesso_indirizzo": safe(i_istituto),
                "punteggio_medio_1": safe(i_punteggio),
                "diff_escs_2": safe(i_diff),
            })
            break

    return records

def _parse_snv_table_with_spunte(table, expected_cols: List[str]) -> List[dict]:
    """
    SNV: estrae SOLO le righe "Situazione..." e le colonne attese.
    Le colonne possono contenere un'immagine: in tal caso VALORE=True.
    """
    heading_parts = _bold_headings_with_classi_before(table)
    heading = " | ".join([h for h in heading_parts if h])

    rows = table.find_all("tr")
    if not rows:
        return []
    header_r = None
    header_cells = []
    for r in rows:
        cells = r.find_all(["th","td"])
        if not cells:
            continue
        txts = [_norm(_clean_text(c)) for c in cells]
        hits = sum(1 for c in expected_cols if any(_norm(c) in t for t in txts))
        if hits >= 2:
            header_r = r
            header_cells = cells
            break
    if header_r is None:
        return []

    headers = [_norm(_clean_text(c)) for c in header_cells]

    col_idx = {}
    for col in expected_cols:
        ncol = _norm(col)
        for i,h in enumerate(headers):
            if ncol in h:
                col_idx[ncol] = i
                break

    out=[]
    data_rows = rows[rows.index(header_r)+1:]
    for r in data_rows:
        cells = r.find_all(["td","th"])
        if not cells:
            continue
        rowlabel = _clean_text(cells[0])
        if "situazione" not in _norm(rowlabel):
            continue
        rec = {"heading": heading, "row_label": rowlabel}
        for col in expected_cols:
            ncol=_norm(col)
            ix = col_idx.get(ncol, None)
            if ix is None:
                rec[ncol] = False
                continue

            adj = ix
            if len(cells) != len(headers):
                adj = (len(cells) - len(headers)) + ix

            if adj < 0 or adj >= len(cells):
                rec[ncol] = False
            else:
                rec[ncol] = _has_img(cells[adj])

        # autoescludente: una sola categoria deve risultare selezionata.
        # Se non troviamo alcuna spunta, lasciamo vuoto (stringa) per distinguere da 0.
        trues = [k for k in ["basso","medio basso","medio alto","alto"] if rec.get(k) is True]
        if len(trues) == 0:
            for k in ["basso","medio basso","medio alto","alto"]:
                rec[k] = ""
        elif len(trues) > 1:
            # Se più colonne risultano 'true' (skin/placeholder), rendiamo deterministico:
            # manteniamo solo la prima nell'ordine previsto.
            keep = trues[0]
            for k in ["basso","medio basso","medio alto","alto"]:
                rec[k] = (k == keep)
        out.append(rec)
    return out

def parse_ep_snv_situazione_spunte(soup: BeautifulSoup) -> List[dict]:
    expected = ["Basso","Medio Basso","Medio Alto","Alto"]
    tables = soup.find_all("table")
    records=[]
    for t in tables:
        if not _bold_headings_with_classi_before(t):
            continue
        recs=_parse_snv_table_with_spunte(t, expected)
        records.extend(recs)
    return records

def parse_ep_snv_variabilita_spunte(soup: BeautifulSoup) -> List[dict]:
    raw = var_extract_tables(soup)
    out: List[dict] = []
    for r in raw:
        headings = r.get("headings_classi", []) or []
        heading = " | ".join([h for h in headings if "classi" in _norm(h)])
        if "classi" not in _norm(heading):
            continue
        row_label = r.get("row_label", "")
        if "situazione" not in _norm(row_label):
            continue
        row = r.get("row", {}) or {}
        rec = {"heading": heading, "row_label": row_label}
        for k, v in row.items():
            if _norm(k) in ("istituto/raggruppamento geografico", "istituto/plesso/indirizzo"):
                continue
            rec[_norm(k)] = v
        out.append(rec)
    return out


def parse_ep_2_2_a_1(soup: BeautifulSoup) -> List[dict]:
    """
    Endpoint 2.2.a.1 (Punteggio nelle prove e differenze rispetto a scuole con ESCS simile).

    Parser basato su logica INVALSI (robusto, "first classi"):
    - individua le tabelle .tableDsc
    - usa i <th class="titleTableDsc"> come heading (0=generale, 1=sotto-tema/materia se presente)
    - trova la riga header (quella che contiene una colonna tipo "Istituto/Plesso/Indirizzo" oppure "Istituto/Raggruppamento geografico")
    - seleziona la prima riga dati in cui la prima cella contiene "classi"
    - esporta tutte le colonne della riga (header->valore) come campi del record
    """
    def _clean(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    def _find_header_row(table) -> Tuple[List[str], Any]:
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            texts = [_clean(c.get_text(" ", strip=True)) for c in cells]
            joined = " ".join(t.lower() for t in texts)
            if ("istituto/plesso/indirizzo" in joined) or ("istituto/raggruppamento geografico" in joined):
                return texts, tr
        return [], None

    out: List[dict] = []

    for table in soup.select("table.tableDsc"):
        # headings
        ths = table.select("th.titleTableDsc")
        heading_general = _clean(ths[0].get_text(" ", strip=True)) if len(ths) > 0 else ""
        heading_subject = _clean(ths[1].get_text(" ", strip=True)) if len(ths) > 1 else ""
        heading = " | ".join([h for h in [heading_general, heading_subject] if h])

        headers, header_tr = _find_header_row(table)
        if not headers or header_tr is None:
            continue

        # prima riga "classi ..."
        for tr in header_tr.find_all_next("tr"):
            if tr.find("th", class_="titleTableDsc"):
                break
            tds = tr.find_all("td", recursive=False)
            if not tds:
                continue
            first = _clean(tds[0].get_text(" ", strip=True))
            if "classi" not in first.lower():
                continue

            rec = {"heading": heading, "row_label": first}
            # mappa header->valore per quanto disponibile
            for i, h in enumerate(headers):
                if i >= len(tds):
                    v = ""
                else:
                    v = _clean(tds[i].get_text(" ", strip=True))
                nk = _norm(h)
                # evita colonne identificative ridondanti
                if nk in ("istituto/plesso/indirizzo", "istituto/raggruppamento geografico"):
                    continue
                rec[nk] = v
            out.append(rec)
            break  # una riga per tabella

    return out

def records_to_long_rows(
    idx: int,
    codice_utilizzato: str,
    codice_scuola: str,
    codice_istituto: str,
    tipo: int,
    codice_ep: str,
    nome_ep: str,
    records: List[dict],
) -> List[dict]:
    """
    Converte records (heading + row_label + colonne) nel formato long.
    """
    out=[]
    for t_idx, rec in enumerate(records, start=1):
        modalita = rec.get("heading","")
        row_label = rec.get("row_label","")
        base = {
            "INDEX": idx,
            "CODICE_UTILIZZATO": codice_utilizzato,
            "CODICE_SCUOLA": codice_scuola,
            "CODICE_ISTITUTO": codice_istituto,
            "TIPO": tipo,
            "DESCRITTORE": codice_ep,
            "NOME_DESCRITTORE": nome_ep,
            "MODALITA": modalita,
            "TABELLA_IDX": t_idx,
        }
        out.append({**base, "COLONNA": "label", "VALORE": "" if row_label is None else str(row_label)})
        for k,v in rec.items():
            if k in ("heading","row_label"):
                continue
            # CSV robusto: spunte come 1/0, e VALORE sempre stringa
            if isinstance(v, bool):
                vv = "1" if v else "0"
            elif v is None:
                vv = ""
            else:
                vv = str(v)
            out.append({**base, "COLONNA": str(k), "VALORE": vv})
    return out


# ------------------------------------------------
# Wide export helpers (per rendere l'output usabile)
# ------------------------------------------------
RE_COHORT = re.compile(r"classi\s+(seconde|quinte(?:/ultimo anno)?|quarte|terze|prime)", re.I)
RE_GRADE = re.compile(r"scuola\s+(primaria|secondaria\s+di\s+i\s+grado|secondaria\s+di\s+ii\s+grado)", re.I)

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("/", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def make_var_slug(colonna: str) -> str:
    if not colonna:
        return ""
    c = colonna.strip()
    if c == "label":
        return "label"
    if "|" not in c:
        return _slug(c)

    parts = [p.strip() for p in c.split("|")]
    if len(parts) < 3:
        return _slug(c)

    dim = parts[-1]
    materia = parts[-2]
    prefix = parts[0]

    prefix2 = prefix.lower().replace("variabilità dei punteggi -", "").strip()
    seg = prefix2
    m = re.search(r"\s-\s*scuola\s", seg)
    if m:
        seg = seg[:m.start()].strip()

    grade = ""
    mg = RE_GRADE.search(prefix)
    if mg:
        grade = mg.group(1)

    cohort = ""
    mc = RE_COHORT.search(prefix)
    if mc:
        cohort = mc.group(1)

    return "_".join([x for x in [
        "var",
        _slug(seg),
        _slug(grade),
        _slug(cohort),
        _slug(materia),
        _slug(dim),
    ] if x])

def write_wide_csv_from_long(long_csv: str, wide_csv: str) -> None:
    dfl = pd.read_csv(long_csv, sep=";", dtype=str, encoding="utf-8")
    if dfl.empty:
        Path(wide_csv).write_text("", encoding="utf-8")
        return

    dfl["VALORE"] = dfl["VALORE"].fillna("").astype(str)

    def _cw(row):
        if row.get("DESCRITTORE") == "2.2.b.2":
            return make_var_slug(row.get("COLONNA", ""))
        return _slug(row.get("COLONNA", ""))

    dfl["COLONNA_WIDE"] = dfl.apply(_cw, axis=1)

    idx_cols = ["CODICE_SCUOLA", "CODICE_ISTITUTO", "CODICE_UTILIZZATO", "DESCRITTORE", "MODALITA", "TABELLA_IDX"]
    wide = dfl.pivot_table(index=idx_cols, columns="COLONNA_WIDE", values="VALORE", aggfunc="first")
    wide = wide.reset_index()
    wide.columns = [c if isinstance(c, str) else str(c) for c in wide.columns]
    wide.to_csv(wide_csv, sep=";", encoding="utf-8", index=False, quoting=csv.QUOTE_ALL)


# ------------------------------------------------
# Logging
# ------------------------------------------------
logging.basicConfig(
    filename="error_log.txt",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

VERBOSE = False
NORMALIZATION_MODE = 'readable'

# ------------------------------------------------
# Config
# ------------------------------------------------
INPUT_CSV = "input.csv"
ENDPOINTS_CSV = "endpoints.csv"

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_HTML_DIR = f"html/html_raw_{RUN_TS}"
OUTPUT_CSV = f"output_{RUN_TS}.csv"
WIDE_CSV = f"wide_{RUN_TS}.csv"
FAILED_SCHOOLS_CSV = f"failed_schools_{RUN_TS}.csv"

MAX_CONCURRENT = 3
WAIT_TIME_RANGE = (0.7, 2.2)
HEADERS = {"User-Agent": "Mozilla/5.0"}
SESSION_TIMEOUT = aiohttp.ClientTimeout(total=60)

# Retry fallback (se vuoi mantenere loop locale)
MAX_RETRIES = 3
RETRY_BACKOFF = 0.8

# Output atomico per INDEX
BLOCKS_DIR = "blocks_partial"
BLOCKS_META = str(Path(BLOCKS_DIR) / "meta.json")

FIELDNAMES = [
    "INDEX",
    "TIPO",
    "CODICE_UTILIZZATO",
    "CODICE_SCUOLA",
    "CODICE_ISTITUTO",
    "TABELLA_IDX",
    "DESCRITTORE",
    "NOME_DESCRITTORE",
    "MODALITA",
    "COLONNA",
    "ORDINE_GRADO",
    "VALORE",
]

_TEXT_FIELDS_TO_NORMALIZE = {
    "NOME_DESCRITTORE",
    "MODALITA",
    "COLONNA",
    "ORDINE_GRADO",
}

def _clean_text_no_regex(s: str, lower: bool = False) -> str:
    if s is None:
        return ""
    s = str(s).replace("\xa0", " ").strip()
    s = " ".join(s.split())
    return s.lower() if lower else s

# ------------------------------------------------
# Regex
# ------------------------------------------------
RE_PLESSO_SEZ = re.compile(r"plesso_[a-z0-9]+_sezione_([a-z0-9]+)", re.I)
RE_SEZIONE = re.compile(r"sezione_([a-z0-9]+)", re.I)
RE_MEC = re.compile(r"rmi[cps]\d{6,}", re.I)
RE_MECGEN = re.compile(r"[A-Z]{3,}[0-9]{5,}", re.I)
RE_SITUAZ = re.compile(r"situazione_della_scuola_.*", re.I)
RE_CLASSE_VAL = re.compile(r"classe", re.I)
RE_VALORE_SCARTO = re.compile(r"^situazione della scuola.*", re.I)
RE_PRIMARIA = re.compile(r"primaria|primarie|infanzia", re.I)

PATTERN_RIF = re.compile(
    r"riferimento provinciale|riferimento regionale|riferimento nazionale|provincia|regione|italia|nazionale|"
    r"centro|nord|sud|lazio|roma|punteggio|percentuale di copertura|riferimenti", re.I
)

# ------------------------------------------------
# Meta / resume guard
# ------------------------------------------------
def file_sig(path: str) -> Dict[str, object]:
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"size": float(st.st_size), "mtime": float(st.st_mtime), "sha256": h.hexdigest()}

def ensure_blocks_dir(fresh: bool) -> None:
    p = Path(BLOCKS_DIR)
    if fresh and p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def write_blocks_meta() -> None:
    meta = {
        "created_at": RUN_TS,
        "normalization_mode": NORMALIZATION_MODE,
        "input_csv": file_sig(INPUT_CSV),
        "endpoints_csv": file_sig(ENDPOINTS_CSV),
    }
    Path(BLOCKS_DIR).mkdir(parents=True, exist_ok=True)
    Path(BLOCKS_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")

def check_blocks_meta_or_isolate() -> str:
    if not Path(BLOCKS_META).exists():
        write_blocks_meta()
        return BLOCKS_DIR

    try:
        meta = json.loads(Path(BLOCKS_META).read_text(encoding="utf-8"))
        cur = {
            "normalization_mode": NORMALIZATION_MODE,
            "input_csv": file_sig(INPUT_CSV),
            "endpoints_csv": file_sig(ENDPOINTS_CSV),
        }
        if (meta.get("normalization_mode") == cur["normalization_mode"]
            and meta.get("input_csv") == cur["input_csv"]
            and meta.get("endpoints_csv") == cur["endpoints_csv"]):
            return BLOCKS_DIR

        alt = f"{BLOCKS_DIR}_{RUN_TS}"
        Path(alt).mkdir(parents=True, exist_ok=True)
        alt_meta = str(Path(alt) / "meta.json")
        Path(alt_meta).write_text(
            json.dumps(
                {
                    "created_at": RUN_TS,
                    "normalization_mode": cur["normalization_mode"],
                    "input_csv": cur["input_csv"],
                    "endpoints_csv": cur["endpoints_csv"],
                    "note": "auto-isolated due to meta mismatch",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return alt
    except Exception:
        logger.error(f"[META_ERROR] {traceback.format_exc()}")
        write_blocks_meta()
        return BLOCKS_DIR

def done_indices_from_blocks(blocks_dir: str) -> set:
    done = set()
    p = Path(blocks_dir)
    if not p.exists():
        return done
    for fp in p.glob("*.csv"):
        stem = fp.stem
        if stem.isdigit():
            done.add(int(stem))
    return done

# ------------------------------------------------
# Lettura CSV robusta
# ------------------------------------------------
def _sniff_delimiter(sample: str) -> str:
    candidates = ["\t", ";", ","]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidates))
        if dialect.delimiter in candidates:
            return dialect.delimiter
    except Exception:
        pass
    scores = {d: sample.count(d) for d in candidates}
    return max(scores, key=scores.get)

def read_csv_flexible(path: str) -> pd.DataFrame:
    raw = Path(path).read_bytes()

    decoded: Optional[str] = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            decoded = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        decoded = raw.decode("utf-8", errors="replace")

    sample = decoded[:20000]
    delim_guess = _sniff_delimiter(sample)

    seps = []
    if delim_guess in (";", ",", "\t"):
        seps.append(delim_guess)
    for s in (";", ",", "\t"):
        if s not in seps:
            seps.append(s)

    last_err: Optional[Exception] = None

    for sep in seps:
        try:
            df = pd.read_csv(
                StringIO(decoded),
                sep=sep,
                engine="python",
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
            )
            df.columns = (
                df.columns.astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
                .str.upper()
            )
            return df
        except Exception as e:
            last_err = e

        try:
            df = pd.read_csv(
                StringIO(decoded),
                sep=sep,
                engine="python",
                dtype=str,
                keep_default_na=False,
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
                on_bad_lines="skip",
            )
            df.columns = (
                df.columns.astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
                .str.upper()
            )
            return df
        except Exception as e:
            last_err = e

    raise last_err if last_err else RuntimeError(f"Impossibile leggere CSV: {path}")

# ------------------------------------------------
# Normalizzazioni (leggibili)
# ------------------------------------------------
def _cleanup_label_human(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("|", " ")
    s = s.replace("_", " ")
    s = s.replace("\\", " ")
    if NORMALIZATION_MODE == "legacy":
        s = s.replace("/", " ")
    s = re.sub(r"[-–—]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_strings(row: dict) -> dict:
    for field in list(row.keys()):
        if field not in _TEXT_FIELDS_TO_NORMALIZE:
            continue

        val = str(row.get(field, "") or "")
        val = val.replace("_|_|", "").replace("_|_", "")
        val = re.sub(r"situazione[_ ]+della[_ ]+scuola", "", val, flags=re.I)

        if NORMALIZATION_MODE == "legacy":
            val = re.sub(r"scuola[_ ]+secondaria[_ ]+di[_ ]+i[_ ]+grado", "medie", val, flags=re.I)
            val = re.sub(r"scuola[_ ]+secondaria[_ ]+di[_ ]+ii[_ ]+grado", "superiori", val, flags=re.I)
            val = re.sub(r"istituti[_ ]+tecnici", "tec", val, flags=re.I)
            val = re.sub(r"istituti[_ ]+professionali", "prof", val, flags=re.I)
            val = re.sub(r"licei[_ ]+scientifici[_ ]+classici[_ ]+e[_ ]+linguistici", "licei", val, flags=re.I)

            val = re.sub(r"classi[_ ]+prime($|_)", "i", val, flags=re.I)
            val = re.sub(r"classi[_ ]+seconde", "ii", val, flags=re.I)
            val = re.sub(r"classi[_ ]+terze", "iii", val, flags=re.I)
            val = re.sub(r"classi[_ ]+quarte", "iv", val, flags=re.I)
            val = re.sub(r"classi[_ ]+quinte", "v", val, flags=re.I)

            val = val.replace("matematica", "mat")
            val = val.replace("italiano", "ita")
            val = val.replace("inglese", "ing")
            val = val.replace("reading", "r")
            val = val.replace("listening", "l")
            val = val.replace("percentuale di studenti", "%")

        val = _cleanup_label_human(val)
        row[field] = val

    descr = row.get("DESCRITTORE", "")
    if descr in ("1.1.a.2_std_dis", "1.1.a.3_std_dsa"):
        row["MODALITA"] = ""

    if descr == "2.4.c.1" and "COLONNA" in row:
        col = row.get("COLONNA", "")
        if col.count("|") >= 2:
            row["COLONNA"] = _cleanup_label_human(col.split("|", 2)[-1].strip())

    return row

def normalizza_colonna(val):
    if pd.isnull(val):
        return ""
    v = str(val)
    v = re.sub(r"\s+", " ", v).strip().lower()
    v = RE_MEC.sub("", v)
    v = re.sub(r"^situazione della scuola\b\s*[:\-–—]?\s*", "", v)
    if not v:
        v = "scuola"
    v = v.replace(", ,", ",").replace(",,", ",")
    v = re.sub(r"_+", "_", v).strip("_")
    return v

# ------------------------------------------------
# Patch e filtri
# ------------------------------------------------
def patch_valori_speciali(row):
    return row

def azzera_valori_speciali(row):
    mod = str(row.get("MODALITA", "")).lower()
    if mod.startswith("sezione_2") or mod.startswith("sezione_5"):
        row["VALORE"] = ""
    return row

# ------------------------------------------------
# Fetch with retry
# ------------------------------------------------
async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
) -> Tuple[int, str]:
    for attempt in range(retries):
        if VERBOSE:
            print(f"[REQUEST] {attempt+1}/{retries} {url}")
        try:
            async with session.get(url) as resp:
                status = resp.status
                text = await resp.text()

                if status in (200, 404):
                    return status, text

                if status == 429:
                    ra = resp.headers.get("Retry-After", "")
                    if ra.isdigit():
                        sleep_s = min(max_delay, float(ra))
                    else:
                        sleep_s = min(max_delay, base_delay * (2 ** attempt))
                    sleep_s *= random.uniform(0.8, 1.3)
                    if attempt < retries - 1:
                        await asyncio.sleep(sleep_s)
                        continue
                    return status, text

                if status in (502, 503, 504):
                    sleep_s = min(max_delay, base_delay * (2 ** attempt)) * random.uniform(0.8, 1.3)
                    if attempt < retries - 1:
                        await asyncio.sleep(sleep_s)
                        continue
                    return status, text

                return status, text

        except (asyncio.TimeoutError, aiohttp.ClientError):
            sleep_s = min(max_delay, base_delay * (2 ** attempt)) * random.uniform(0.8, 1.3)
            if attempt < retries - 1:
                await asyncio.sleep(sleep_s)
                continue
            return 0, ""
        except Exception:
            logger.error(f"[FETCH_ERROR] {url}: {traceback.format_exc()}")
            return 0, ""

    return 0, ""

# ------------------------------------------------
# Session init
# ------------------------------------------------
async def inizializza_sessione(session: aiohttp.ClientSession, codice_scuola: str) -> Tuple[bool, int, str]:
    code = (codice_scuola or "").strip().upper()
    q = urllib.parse.quote_plus(code)

    url_login = (
        "https://snv.pubblica.istruzione.it/"
        "SistemaNazionaleValutazione/scuolaInChiaro.do"
        f"?dispatch=indicatori&scuolainserita={q}"
    )

    status, _ = await fetch_with_retry(session, url_login, retries=3, base_delay=1.0)

    if status == 200:
        return True, status, "OK"
    if status == 404:
        return False, status, "LOGIN_404"
    if status == 429:
        return False, status, "LOGIN_429"
    if status == 0:
        return False, status, "LOGIN_NETERR"
    return False, status, f"LOGIN_{status}"

async def salva_html_async(idx: int, codice_utilizzato: str, codice_ep: str, html: str) -> None:
    out_dir = Path(OUTPUT_HTML_DIR) / str(codice_utilizzato)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{idx}__{codice_ep}.html"
    await asyncio.to_thread(path.write_text, html or "", "utf-8")

def _build_row_out(
    idx: int,
    tipo: int,
    codice_utilizzato: str,
    codice_scuola: str,
    codice_istituto: str,
    tabella_idx: int,
    descrittore: str,
    nome_descrittore: str,
    modalita: str,
    colonna: str,
    valore: str,
) -> dict:
    if tabella_idx == -1 and not modalita:
        modalita = "valore"

    row_out = {
        "INDEX": idx,
        "TIPO": tipo,
        "CODICE_UTILIZZATO": codice_utilizzato,
        "CODICE_SCUOLA": codice_scuola,
        "CODICE_ISTITUTO": codice_istituto,
        "TABELLA_IDX": tabella_idx,
        "DESCRITTORE": descrittore,
        "NOME_DESCRITTORE": (nome_descrittore or "").lower(),
        "MODALITA": modalita,
        "COLONNA": colonna,
        "ORDINE_GRADO": "",
        "VALORE": valore,
    }
    row_out = azzera_valori_speciali(row_out)
    row_out = patch_valori_speciali(row_out)
    row_out = normalize_strings(row_out)
    return row_out

# ------------------------------------------------
# Endpoint helpers (FIX: uppercase/lowercase)
# ------------------------------------------------
def _ep_get(ep: dict, key: str, default: str = "") -> str:
    v = ep.get(key)
    if v is None:
        v = ep.get(key.lower())
    if v is None:
        return default
    return str(v)

# ------------------------------------------------
# Parser per endpoint (solo i 3 descrittori)
# ------------------------------------------------
async def scarica_long_safe_async(
    session: aiohttp.ClientSession,
    idx: int,
    codice_utilizzato: str,
    codice_scuola: str,
    codice_istituto: str,
    tipo: int,
    endpoints: list
) -> List[dict]:
    """
    Scarica e PARSA SOLO tre descrittori SNV/INVALSI:
      - 1.1.b.1  (ESCS: Situazione della scuola con spunte su Basso/Medio Basso/Medio Alto/Alto)
      - 2.2.b.2  (ESCS: Variabilità - righe 'Situazione ...' con spunte)
      - 2.4.a.4  (INVALSI: riga '...classi...' con Punteggio medio (1) e Diff. ESCS (2))
    """
    rows: List[dict] = []
    if not endpoints:
        return []

    wanted = {"1.1.b.1", "2.2.b.2", "2.2.a.1"}
    endpoints = [ep for ep in endpoints if _ep_get(ep, "CODICE_EP", "").strip() in wanted]
    if not endpoints:
        return []

    for endpoint in endpoints:
        codice_ep = _ep_get(endpoint, "CODICE_EP", "").strip()
        url = _ep_get(endpoint, "URL", "").strip()
        nome_ep = (
            _ep_get(endpoint, "NOME_DESCRITTORE", "").strip()
            or _ep_get(endpoint, "NOME", "").strip()
            or _ep_get(endpoint, "DESCRITTORE", "").strip()
        )
        if not codice_ep or not url:
            continue

        ok = False
        html = ""
        status = None

        for _attempt in range(MAX_RETRIES + 1):
            status, html = await fetch_with_retry(session, url)
            if status == 200 and html and len(html) > 200:
                ok = True
                break
            await asyncio.sleep(RETRY_BACKOFF * (1.0 + random.random()))

        await salva_html_async(idx, codice_utilizzato, codice_ep, html)

        if not ok:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for s in soup("script"):
            s.extract()

        try:
            if codice_ep == "2.4.a.4":
                records = parse_ep_2_4_a_4(soup)
                for t_idx, rec in enumerate(records, start=1):
                    modalita = rec.get("heading", "")
                    base = {
                        "INDEX": idx,
                        "CODICE_UTILIZZATO": codice_utilizzato,
                        "CODICE_SCUOLA": codice_scuola,
                        "CODICE_ISTITUTO": codice_istituto,
                        "TIPO": tipo,
                        "DESCRITTORE": codice_ep,
                        "NOME_DESCRITTORE": nome_ep,
                        "MODALITA": modalita,
                        "TABELLA_IDX": t_idx,
                    }
                    for col_key in ("istituto_plesso_indirizzo", "punteggio_medio_1", "diff_escs_2"):
                        rows.append({**base, "COLONNA": col_key, "VALORE": rec.get(col_key, "")})
            elif codice_ep == "1.1.b.1":
                records = parse_ep_snv_situazione_spunte(soup)
                rows.extend(records_to_long_rows(idx, codice_utilizzato, codice_scuola, codice_istituto, tipo, codice_ep, nome_ep, records))
            elif codice_ep == "2.2.a.1":
                records = parse_ep_2_2_a_1(soup)
                rows.extend(records_to_long_rows(idx, codice_utilizzato, codice_scuola, codice_istituto, tipo, codice_ep, nome_ep, records))
            elif codice_ep == "2.2.b.2":
                records = parse_ep_snv_variabilita_spunte(soup)
                rows.extend(records_to_long_rows(idx, codice_utilizzato, codice_scuola, codice_istituto, tipo, codice_ep, nome_ep, records))
        except Exception:
            logger.error(f"[PARSE_ERROR] INDEX={idx} EP={codice_ep} URL={url}\n{traceback.format_exc()}")
            continue

    return rows

async def elabora_scuola_long_async(
    row: pd.Series,
    endpoints: list,
    connector: aiohttp.TCPConnector
) -> Tuple[List[dict], int, str, str, pd.Series]:
    idx = int(row["INDEX"])
    codice_scuola = str(row["CODICE_SCUOLA"])
    codice_istituto = str(row["CODICE_ISTITUTO"])

    async with SEM:
        try:
            cookie_jar = aiohttp.CookieJar()
            async with aiohttp.ClientSession(
                timeout=SESSION_TIMEOUT,
                connector=connector,
                connector_owner=False,
                headers=HEADERS,
                cookie_jar=cookie_jar
            ) as session:

                ok, st, rsn = await inizializza_sessione(session, codice_scuola)
                if not ok:
                    row["_FAIL_STAGE"] = "login_scuola"
                    row["_FAIL_STATUS"] = st
                    row["_FAIL_REASON"] = rsn
                    return [], idx, codice_scuola, codice_istituto, row

                righe = await scarica_long_safe_async(
                    session, idx, codice_scuola, codice_scuola, codice_istituto, 0, endpoints
                )
                if righe:
                    return righe, idx, codice_scuola, codice_istituto, row

                if codice_scuola != codice_istituto:
                    session.cookie_jar.clear()
                    ok2, st2, rsn2 = await inizializza_sessione(session, codice_istituto)
                    if not ok2:
                        row["_FAIL_STAGE"] = "login_istituto"
                        row["_FAIL_STATUS"] = st2
                        row["_FAIL_REASON"] = rsn2
                        return [], idx, codice_scuola, codice_istituto, row

                    righe2 = await scarica_long_safe_async(
                        session, idx, codice_istituto, codice_scuola, codice_istituto, 1, endpoints
                    )
                    if righe2:
                        return righe2, idx, codice_scuola, codice_istituto, row

                if "_FAIL_STAGE" not in row:
                    row["_FAIL_STAGE"] = "no_rows"
                    row["_FAIL_STATUS"] = ""
                    row["_FAIL_REASON"] = "NO_ROWS"
                return [], idx, codice_scuola, codice_istituto, row

        except Exception:
            logger.error(f"[ERROR_TASK] INDEX={idx} SCUOLA={codice_scuola}: {traceback.format_exc()}")
            return [], idx, codice_scuola, codice_istituto, row

# ------------------------------------------------
# I/O atomico per INDEX
# ------------------------------------------------
def write_block_atomic(blocks_dir: str, idx: int, rows: List[dict], text_only_no_regex: bool = False) -> None:
    p = Path(blocks_dir)
    p.mkdir(parents=True, exist_ok=True)
    final_path = p / f"{idx}.csv"
    tmp_path = p / f"{idx}.tmp"

    fixed_rows: List[dict] = []
    for r in rows:
        rr = {k: r.get(k, "") for k in FIELDNAMES}

        # CSV robusto: forzature sui valori
        v = rr.get("VALORE", "")
        if isinstance(v, bool):
            rr["VALORE"] = "1" if v else "0"
        elif v is None:
            rr["VALORE"] = ""
        else:
            rr["VALORE"] = str(v)

        if text_only_no_regex:
            for k in _TEXT_FIELDS_TO_NORMALIZE:
                if k in rr:
                    if k in {"MODALITA", "COLONNA"}:
                        rr[k] = _clean_text_no_regex(rr[k], lower=True)
                    else:
                        rr[k] = _clean_text_no_regex(rr[k], lower=False)
        fixed_rows.append(rr)

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";", quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(fixed_rows)

    os.replace(tmp_path, final_path)

def merge_blocks_to_csv(blocks_dir: str, out_csv: str) -> int:
    p = Path(blocks_dir)
    files = [fp for fp in p.glob("*.csv") if fp.stem.isdigit()]
    files.sort(key=lambda x: int(x.stem))

    if not files:
        return 0

    with open(out_csv, "w", encoding="utf-8", newline="") as fout:
        wrote_header = False
        for fp in files:
            with open(fp, "r", encoding="utf-8", newline="") as fin:
                for i, line in enumerate(fin):
                    if i == 0 and wrote_header:
                        continue
                    fout.write(line)
            wrote_header = True

    return len(files)

# ------------------------------------------------
# Release builder (snapshot versionato)
# ------------------------------------------------
def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _safe_symlink_latest(latest_path: Path, target: Path) -> None:
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(target.name)
    except Exception:
        pass

def build_release_snapshot(
    release_root: Path,
    release_id: str,
    df_schools: pd.DataFrame,
    df_endpoints_raw: pd.DataFrame,
    merged_csv_path: Path,
    normalization_mode: str,
) -> Path:
    release_dir = release_root / release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    schools_out = release_dir / "schools.csv"
    df_schools.to_csv(schools_out, index=False, encoding="utf-8")

    df_ep = df_endpoints_raw.copy()
    df_ep = df_ep.rename(columns={c: c.lower() for c in df_ep.columns})
    out_cols = {}
    if "codice_ep" in df_ep.columns:
        out_cols["codice_ep"] = "endpoint_key"
    if "nome" in df_ep.columns:
        out_cols["nome"] = "title"
    if "url" in df_ep.columns:
        out_cols["url"] = "url"
    df_ep = df_ep.rename(columns=out_cols)
    if "title" not in df_ep.columns:
        df_ep["title"] = df_ep.get("endpoint_key", "")
    endpoints_out = release_dir / "endpoints.csv"
    df_ep.to_csv(endpoints_out, index=False, encoding="utf-8")

    obs_out = release_dir / "observations_semantic.csv"
    if merged_csv_path.resolve() != obs_out.resolve():
        shutil.copy2(merged_csv_path, obs_out)

    def _count_rows(fp: Path) -> int:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)

    manifest = {
        "release_id": release_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "normalization_mode": normalization_mode,
        "counts": {
            "schools": int(len(df_schools)),
            "endpoints": int(len(df_ep)),
            "observations_rows": int(_count_rows(obs_out)),
        },
        "files": {}
    }

    for fp in [schools_out, endpoints_out, obs_out]:
        manifest["files"][fp.name] = {"bytes": fp.stat().st_size, "sha256": _sha256_file(fp)}

    (release_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _safe_symlink_latest(release_root / "latest", release_dir)

    parquet_engine = None
    try:
        import pyarrow  # noqa: F401
        parquet_engine = "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            parquet_engine = "fastparquet"
        except Exception:
            parquet_engine = None

    if parquet_engine:
        try:
            df_schools.to_parquet(release_dir / "schools.parquet", index=False, engine=parquet_engine)
            df_ep.to_parquet(release_dir / "endpoints.parquet", index=False, engine=parquet_engine)
            try:
                df_obs = pd.read_csv(obs_out, dtype=str)
                df_obs.to_parquet(release_dir / "observations_semantic.parquet", index=False, engine=parquet_engine)
            except Exception:
                pass
        except Exception:
            pass

    return release_dir

# ------------------------------------------------
# Main async
# ------------------------------------------------
async def main_async(limit: int, school: str, verbose: bool, fresh: bool, text_only_no_regex: bool = False,
                     release_root: str = 'releases', release_id: str = '', no_release: bool = False,
                     concurrency: int = 3, timeout_s: int = 60):
    global VERBOSE
    VERBOSE = verbose

    global MAX_CONCURRENT, SEM, SESSION_TIMEOUT
    MAX_CONCURRENT = max(1, int(concurrency))
    SEM = asyncio.Semaphore(MAX_CONCURRENT)
    SESSION_TIMEOUT = aiohttp.ClientTimeout(total=max(10, int(timeout_s)))

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] File {INPUT_CSV} non trovato.")
        return
    if not os.path.exists(ENDPOINTS_CSV):
        print(f"[ERROR] File {ENDPOINTS_CSV} non trovato.")
        return

    ensure_blocks_dir(fresh=fresh)
    blocks_dir = check_blocks_meta_or_isolate()

    df_input = read_csv_flexible(INPUT_CSV)

    required = {"CODICE_SCUOLA", "CODICE_ISTITUTO"}
    missing = required - set(df_input.columns)
    if missing:
        raise SystemExit(
            f"[ERROR] Colonne mancanti in {INPUT_CSV}: {sorted(missing)}. "
            f"Colonne lette: {list(df_input.columns)}"
        )

    df_input = df_input.reset_index(drop=True)
    df_input.insert(0, "INDEX", range(0, len(df_input)))

    df_ep = read_csv_flexible(ENDPOINTS_CSV)
    df_ep_raw = df_ep.copy()

    required_ep = {"CODICE_EP", "URL"}
    missing_ep = required_ep - set(df_ep.columns)
    if missing_ep:
        raise SystemExit(
            f"[ERROR] Colonne mancanti in {ENDPOINTS_CSV}: {sorted(missing_ep)}. "
            f"Colonne lette: {list(df_ep.columns)}"
        )

    # Mantieni anche versione "records" con chiavi lowercase per comodità,
    # ma il codice legge entrambe (FIX).
    df_ep_lc = df_ep.rename(columns={c: c.lower() for c in df_ep.columns})
    endpoints = df_ep_lc.to_dict("records")

    if not release_id:
        release_id = datetime.now().date().isoformat()
    release_root_path = Path(release_root)
    release_root_path.mkdir(parents=True, exist_ok=True)

    if school:
        df_to_process = df_input[df_input["CODICE_SCUOLA"] == school].copy()
        if df_to_process.empty:
            print(f"[ERROR] Codice '{school}' non trovato.")
            return
    else:
        done = done_indices_from_blocks(blocks_dir)
        if done and VERBOSE:
            print(f"[RESUME] blocchi già presenti: {len(done)}")
        df_to_process = df_input[~df_input["INDEX"].isin(done)].copy()
        if limit and limit > 0:
            df_to_process = df_to_process.head(limit)

    totale = len(df_to_process)
    merged = 0
    if totale == 0:
        print("[INFO] Nessuna scuola da processare.")
        merged = merge_blocks_to_csv(blocks_dir, OUTPUT_CSV)
        # Wide export (più usabile per pivot)
        if merged:
            try:
                write_wide_csv_from_long(OUTPUT_CSV, WIDE_CSV)
                print(f"[OK] Wide export: {WIDE_CSV}")
            except Exception as e:
                print(f"[WARNING] Wide export failed: {type(e).__name__}: {e}")
            print(f"[OK] Merge completato: {OUTPUT_CSV} (blocchi: {merged})")
        return

    failed_schools: List[dict] = []
    esiti: Dict[int, str] = {}

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, limit_per_host=MAX_CONCURRENT)

    try:
        async def _run_pass(rows_in: list, desc: str) -> list:
            if not rows_in:
                return []
            failed_local: list = []
            pbar = tqdm(total=len(rows_in), desc=desc)
            q: asyncio.Queue = asyncio.Queue()
            for r in rows_in:
                q.put_nowait(r)
            for _ in range(MAX_CONCURRENT):
                q.put_nowait(None)

            lock = asyncio.Lock()

            async def worker():
                while True:
                    row_item = await q.get()
                    if row_item is None:
                        q.task_done()
                        break

                    righe, idx, cod_scuola, cod_ist, row_orig = await elabora_scuola_long_async(row_item, endpoints, connector)

                    if righe:
                        await asyncio.to_thread(write_block_atomic, blocks_dir, idx, righe, text_only_no_regex)
                        esito = "fallback_istituto" if any(int(x.get("TIPO", 0)) == 1 for x in righe) else "main"
                        esiti[idx] = esito
                    else:
                        async with lock:
                            failed_local.append(row_orig)
                        esiti[int(row_orig["INDEX"])] = "failed_pass"

                    pbar.update(1)
                    q.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(MAX_CONCURRENT)]
            await q.join()
            for w in workers:
                await w
            pbar.close()
            return failed_local

        rows_pass1 = [r for _, r in df_to_process.iterrows()]
        waitlist_rows = await _run_pass(rows_pass1, desc="Elaborazione scuole")

        waitlist_backup = await _run_pass(waitlist_rows, desc="Retry scuole fallite")

        if waitlist_backup:
            rows_backup = []
            for row_orig in waitlist_backup:
                idx = int(row_orig["INDEX"])
                scuola_bkp = str(row_orig.get("CODICE_SCUOLA_BACKUP", "")).strip()
                ist_bkp = str(row_orig.get("CODICE_ISTITUTO_BACKUP", "")).strip()

                if (scuola_bkp and ist_bkp and scuola_bkp.lower() not in ("nan", "none") and ist_bkp.lower() not in ("nan", "none")):
                    row_copy = row_orig.copy()
                    row_copy["CODICE_SCUOLA"] = scuola_bkp
                    row_copy["CODICE_ISTITUTO"] = ist_bkp
                    rows_backup.append(row_copy)
                else:
                    failed_schools.append({
                        "INDEX": idx,
                        "CODICE_SCUOLA": row_orig.get("CODICE_SCUOLA", ""),
                        "CODICE_ISTITUTO": row_orig.get("CODICE_ISTITUTO", ""),
                        "FAIL_STAGE": row_orig.get("_FAIL_STAGE", ""),
                        "FAIL_STATUS": row_orig.get("_FAIL_STATUS", ""),
                        "REASON": row_orig.get("_FAIL_REASON", "BACKUP_FIELDS_MISSING"),
                        "DETAIL": "backup_columns_missing_or_invalid",
                    })
                    esiti[idx] = "fail"

            still_failed = await _run_pass(rows_backup, desc="Retry codici backup")
            for row_orig in still_failed:
                idx = int(row_orig["INDEX"])
                failed_schools.append({
                    "INDEX": idx,
                    "CODICE_SCUOLA": row_orig.get("CODICE_SCUOLA", ""),
                    "CODICE_ISTITUTO": row_orig.get("CODICE_ISTITUTO", ""),
                    "FAIL_STAGE": row_orig.get("_FAIL_STAGE", ""),
                    "FAIL_STATUS": row_orig.get("_FAIL_STATUS", ""),
                    "REASON": row_orig.get("_FAIL_REASON", "BACKUP_FAIL"),
                    "DETAIL": "backup_codes_attempted",
                })
                esiti[idx] = "fail"
    finally:
        await connector.close()

    merged = merge_blocks_to_csv(blocks_dir, OUTPUT_CSV)
    if merged:
        print(f"[OK] Merge completato: {OUTPUT_CSV} (blocchi: {merged})")
    else:
        print("[WARNING] Nessun blocco da unire.")

    if (not no_release) and merged:
        try:
            release_dir = build_release_snapshot(
                release_root=release_root_path,
                release_id=release_id,
                df_schools=df_input,
                df_endpoints_raw=df_ep_raw,
                merged_csv_path=Path(OUTPUT_CSV),
                normalization_mode=NORMALIZATION_MODE,
            )
            print(f"[OK] Release creata: {release_dir}")
        except Exception as e:
            print(f"[WARNING] Release non creata: {e}")

    if failed_schools:
        df_failed = pd.DataFrame(failed_schools)
        df_failed.to_csv(FAILED_SCHOOLS_CSV, index=False, sep=";", quoting=csv.QUOTE_ALL)
        print(f"[INFO] Fallite: {FAILED_SCHOOLS_CSV}")

    stat_main = sum(1 for v in esiti.values() if v == "main")
    stat_fallback = sum(1 for v in esiti.values() if v == "fallback_istituto")
    stat_fail = sum(1 for v in esiti.values() if v == "fail")

    print("\n========== STATISTICA ESTRAZIONE ==========")
    print(f"Totale scuole in run:           {totale}")
    print(f"  - Successo immediato:         {stat_main}")
    print(f"  - Solo fallback istituto:     {stat_fallback}")
    print(f"  - Fallite:                    {stat_fail}")
    print("===========================================\n")

# ------------------------------------------------
# CLI
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scraping SNV asincrono con output atomico per INDEX e merge finale.")
    parser.add_argument("--release-root", type=str, default="releases", help="Cartella radice per snapshot release (default: releases)")
    parser.add_argument("--release-id", type=str, default="", help="ID release (es. 2026-07-01). Default: data odierna.")
    parser.add_argument("--no-release", action="store_true", help="Non costruire snapshot release, genera solo output CSV merged")
    parser.add_argument("--concurrency", type=int, default=3, help="Numero massimo di scuole in parallelo")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout totale per richiesta (secondi)")
    parser.add_argument("-n", "--limit", type=int, default=0, help="Numero massimo scuole (0=tutte)")
    parser.add_argument("-s", "--school", type=str, default="", help="Codice meccanografico singola scuola")
    parser.add_argument("--verbose", action="store_true", help="Log su console")
    parser.add_argument("--fresh", action="store_true", help="Cancella blocks_partial e riparte da zero")
    parser.add_argument("--legacy-normalization", action="store_true", help="Normalizzazioni compatte stile v5 (rimuove /, abbreviazioni tec/prof, ecc.)")
    parser.add_argument("--textnorm", action="store_true", help="Normalizza SOLO i campi in _TEXT_FIELDS_TO_NORMALIZE e senza regex (split/join).")
    args = parser.parse_args()

    global NORMALIZATION_MODE
    NORMALIZATION_MODE = "legacy" if args.legacy_normalization else "readable"

    school_code = args.school.strip() if args.school else ""
    try:
        asyncio.run(main_async(
            limit=args.limit,
            school=school_code,
            verbose=args.verbose,
            fresh=args.fresh,
            text_only_no_regex=args.textnorm,
            release_root=args.release_root,
            release_id=args.release_id,
            no_release=args.no_release,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
        ))
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Interruzione manuale.")

if __name__ == "__main__":
    main()
