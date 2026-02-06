#!/usr/bin/env python3
"""
sic_scraper_async.py — Batch scraper per ScuolaInChiaro / UNICA (API).

Salva sempre il raw su disco (outdir/raw/<codice>/<endpoint>.json|.txt),
checkpoint per scuola, resume automatico.
Worker pool async con coda, backpressure reale, retry su 429/5xx/timeout.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import aiohttp
import pandas as pd


# --- endpoints ---

ENDPOINTS: List[Tuple[str, str]] = [
    (
        "anagrafica_base",
        "https://unica.istruzione.gov.it/services/sic/api/v1.0/ricerche/ricercaRapida?chiaviDiRicerca={CODICE_SCUOLA}&numeroElementiPagina=5000&numeroPagina=1",
    ),
#    ("numero_alunni_23_24", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/andamento-alunni/"),

    ("esiti_giugno_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/esiti-giugno"),
    ("esiti_giugno_settembre_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/esiti-giugno-settembre"),
    ("sospesi_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/sospesi/"),

    ("diplomati_esaminati_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/diplomati-esaminati/"),
    ("distribuzione_votazioni_esame_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/distribuzione-votazioni-esame/"),

    ("abbandoni_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/abbandoni"),
    ("trasferimenti_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/trasferimenti"),
    ("studenti_ripetenti", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/studenti-ripetenti"),

    ("rav_24c5", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/rav-24c5"),
    ("rav_24c1", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/rav-24c1"),
    ("rav_24c2_II", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/rav-24c2?i=II"),
    ("rav_24c3_II", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/rav-24c3?i=II"),

    ("immatricolati_universita", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/immatricolati-universita/"),
    ("immatricolati_universita_area", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/immatricolati-universita-area-didattica/"),

    ("docenti_fasce_eta", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/docenti-fasce-eta"),
    ("docenti_trasferiti", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/docenti-trasferiti"),
    ("docenti_pensionati", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/docenti-pensionati"),
    ("assenze_docenti", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/assenze-docenti"),
    ("assenze_ata", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/assenze-ata"),

    ("entrate_fonti_finanziamento", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/finanza/grafici/entrate-fonti-finanziamento"),
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Python aiohttp",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "it,en-US;q=0.7,en;q=0.3",
}


# --- input ---

def read_schools_csv(path: Path, sep: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")

    # normalizza header
    cols = {c.strip().strip('"'): c for c in df.columns}
    df = df.rename(columns={v: k for k, v in cols.items()})

    required = {"CODICESCUOLA", "DENOMINAZIONESCUOLA"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Header senza {required}. Trovate: {list(df.columns)}")

    want = ["CODICESCUOLA", "DENOMINAZIONESCUOLA", "AREAGEOGRAFICA", "REGIONE", "PROVINCIA"]
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise SystemExit(f"Header mancanti: {missing}. Trovate: {list(df.columns)}")

    out = df[want].copy()
    out = out.rename(columns={"CODICESCUOLA": "CODICE_SCUOLA", "DENOMINAZIONESCUOLA": "NOME_SCUOLA"})

    out["CODICE_SCUOLA"] = out["CODICE_SCUOLA"].astype(str).str.strip()
    out["NOME_SCUOLA"] = out["NOME_SCUOLA"].astype(str).str.strip()
    out["AREAGEOGRAFICA"] = out["AREAGEOGRAFICA"].astype(str).str.strip()
    out["REGIONE"] = out["REGIONE"].astype(str).str.strip()
    out["PROVINCIA"] = out["PROVINCIA"].astype(str).str.strip()

    out = out[out["CODICE_SCUOLA"] != ""]
    out = out.drop_duplicates(subset=["CODICE_SCUOLA"])
    return out


# --- helpers ---

def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (s or "").strip())
    return s[:180] if s else "x"

def build_url(template: str, codice: str, nome_scuola: str) -> str:
    if "{NOME_SCUOLA}" not in template:
        return template.format(CODICE_SCUOLA=codice)
    nome_enc = quote((nome_scuola or "").strip(), safe="")
    return template.format(CODICE_SCUOLA=codice, NOME_SCUOLA=nome_enc)

def looks_like_json(content_type: str, text: str) -> bool:
    ct = (content_type or "").lower()
    if "json" in ct:
        return True
    t = (text or "").lstrip()
    return t.startswith("{") or t.startswith("[")

def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def _row_to_dict(r: Any) -> Dict[str, Any]:
    if isinstance(r, dict):
        return r
    if is_dataclass(r):
        return asdict(r)
    return vars(r)

def jitter(base: float) -> float:
    return base * (0.7 + random.random() * 0.6)


# --- result model ---

@dataclass
class ResultRow:
    CODICE_SCUOLA: str
    NOME_SCUOLA: str
    endpoint_key: str
    url: str
    final_url: str
    status: int
    ok: bool
    content_type: str
    is_json: bool
    raw_len: int
    saved_path: str
    error: str
    from_cache: bool


# --- checkpoint ---

def write_school_checkpoint(blocks_dir: Path, codice_scuola: str, rows: List[ResultRow]) -> None:
    blocks_dir.mkdir(parents=True, exist_ok=True)
    out_path = blocks_dir / f"{codice_scuola}.csv"
    tmp = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")

    rows = rows or []
    rows_dict = [_row_to_dict(r) for r in rows]

    if not rows_dict:
        fieldnames = ["CODICE_SCUOLA"]
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({"CODICE_SCUOLA": codice_scuola})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
        return

    fieldnames = list(rows_dict[0].keys())
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_dict)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)

def get_done_schools(blocks_dir: Path) -> set[str]:
    if not blocks_dir.exists():
        return set()
    return {p.stem for p in blocks_dir.glob("*.csv")}


# --- http fetch ---

TRANSIENT_STATUSES = {429, 500, 502, 503, 504}

def _timeout(args: argparse.Namespace) -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(
        total=args.timeout_total,
        connect=args.timeout_connect,
        sock_read=args.timeout_read,
    )

async def fetch_with_retry(
    session: aiohttp.ClientSession,
    inflight_sem: asyncio.Semaphore,
    url: str,
    *,
    retries: int,
    base_backoff: float,
    retry_4xx: bool,
) -> Tuple[int, str, str, str, str]:
    """Ritorna (status, final_url, content_type, text, err). status=-1 per eccezioni."""
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            async with inflight_sem:
                async with session.get(url, allow_redirects=True) as resp:
                    text = await resp.text(errors="replace")
                    status = resp.status
                    final_url = str(resp.url)
                    content_type = resp.headers.get("Content-Type", "")
                    retry_after = resp.headers.get("Retry-After")

            if 200 <= status < 300:
                return status, final_url, content_type, text, ""

            if status in TRANSIENT_STATUSES or (retry_4xx and 400 <= status < 500):
                if attempt < retries:
                    if status == 429 and retry_after:
                        try:
                            wait = float(retry_after)
                        except Exception:
                            wait = jitter(base_backoff * attempt)
                    else:
                        wait = jitter(base_backoff * attempt)
                    await asyncio.sleep(wait)
                    continue

            return status, final_url, content_type, text, f"http_{status}"

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                await asyncio.sleep(jitter(base_backoff * attempt))
                continue
            return -1, url, "", f'{{"_error":"failed","_last_exception":{json.dumps(last_err)}}}', "exception"

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                await asyncio.sleep(jitter(base_backoff * attempt))
                continue
            return -1, url, "", f'{{"_error":"failed","_last_exception":{json.dumps(last_err)}}}', "exception"

    return -1, url, "", f'{{"_error":"failed","_last_exception":{json.dumps(last_err)}}}', "exception"


async def fetch_one_endpoint(
    session: aiohttp.ClientSession,
    inflight_sem: asyncio.Semaphore,
    raw_dir: Path,
    codice: str,
    nome: str,
    endpoint_key: str,
    template: str,
    *,
    skip_existing: bool,
    fresh: bool,
    retries: int,
    base_backoff: float,
    retry_4xx: bool,
) -> ResultRow:
    url = build_url(template, codice, nome)

    school_dir = raw_dir / safe_filename(codice)
    school_dir.mkdir(parents=True, exist_ok=True)

    cached_json = school_dir / f"{safe_filename(endpoint_key)}.json"
    cached_txt = school_dir / f"{safe_filename(endpoint_key)}.txt"
    cached_path = cached_json if cached_json.exists() else (cached_txt if cached_txt.exists() else None)

    if (not fresh) and skip_existing and cached_path is not None and cached_path.stat().st_size > 0:
        raw = cached_path.read_text(encoding="utf-8", errors="replace")
        is_json = cached_path.suffix.lower() == ".json" or looks_like_json("application/json", raw)
        if is_json:
            try:
                obj = json.loads(raw)
                raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                cached_path = None

        if cached_path is not None:
            return ResultRow(
                CODICE_SCUOLA=codice,
                NOME_SCUOLA=nome,
                endpoint_key=endpoint_key,
                url=url,
                final_url=url,
                status=200,
                ok=True,
                content_type="application/json" if is_json else "text/plain",
                is_json=is_json,
                raw_len=len(raw),
                saved_path=str(cached_path),
                error="cached",
                from_cache=True,
            )

    status, final_url, content_type, text, err = await fetch_with_retry(
        session,
        inflight_sem,
        url,
        retries=retries,
        base_backoff=base_backoff,
        retry_4xx=retry_4xx,
    )

    ok = 200 <= status < 300
    is_json = looks_like_json(content_type, text)
    parsed_error = "" if not err else err
    raw_to_save = text

    if is_json:
        try:
            obj = json.loads(text)
            raw_to_save = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            parsed_error = (parsed_error + "; " if parsed_error else "") + f"json_parse_error: {e}"

    ext = "json" if is_json else "txt"
    fpath = school_dir / f"{safe_filename(endpoint_key)}.{ext}"
    atomic_write_text(fpath, raw_to_save)

    return ResultRow(
        CODICE_SCUOLA=codice,
        NOME_SCUOLA=nome,
        endpoint_key=endpoint_key,
        url=url,
        final_url=final_url,
        status=status,
        ok=ok,
        content_type=content_type,
        is_json=is_json,
        raw_len=len(raw_to_save),
        saved_path=str(fpath),
        error=parsed_error,
        from_cache=False,
    )


# --- semantic extraction (long) ---

OBS_FIELDNAMES = [
    "CODICE_SCUOLA",
    "NOME_SCUOLA",
    "endpoint_key",
    "title",
    "series_index",
    "series_name",
    "category_index",
    "category",
    "value",
    "info",
    "note",
    "kind",
]

def load_json(path: Path) -> Optional[Any]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

def iter_observations_from_obj(
    obj: Dict[str, Any],
    codice: str,
    nome: str,
    endpoint_key: str,
    include_kind: bool,
) -> Iterable[Dict[str, Any]]:
    title = obj.get("title")
    info = obj.get("info")
    note = obj.get("note")
    categories = obj.get("categories")
    series = obj.get("series")

    if not isinstance(series, list) or not series:
        return

    def s_data_is_list(s: Any) -> bool:
        return isinstance(s, dict) and isinstance(s.get("data"), list)

    any_list_data = any(s_data_is_list(s) for s in series)
    all_scalar = all(isinstance(s, dict) and not isinstance(s.get("data"), list) for s in series)

    if isinstance(categories, list) and any_list_data:
        kind = "by_categories"
        for si, s in enumerate(series):
            if not isinstance(s, dict):
                continue
            sname = s.get("name")
            data = s.get("data")
            if not isinstance(data, list):
                continue
            for ci, val in enumerate(data):
                cat = categories[ci] if ci < len(categories) else None
                row = {
                    "CODICE_SCUOLA": codice,
                    "NOME_SCUOLA": nome,
                    "endpoint_key": endpoint_key,
                    "title": title,
                    "series_index": si,
                    "series_name": sname,
                    "category_index": ci,
                    "category": cat,
                    "value": val,
                    "info": info,
                    "note": note,
                }
                if include_kind:
                    row["kind"] = kind
                yield row

    elif categories is None and all_scalar:
        kind = "by_name_scalar"
        for si, s in enumerate(series):
            if not isinstance(s, dict):
                continue
            cat = s.get("name")
            val = s.get("data") if "data" in s else s.get("y")
            row = {
                "CODICE_SCUOLA": codice,
                "NOME_SCUOLA": nome,
                "endpoint_key": endpoint_key,
                "title": title,
                "series_index": None,
                "series_name": None,
                "category_index": si,
                "category": cat,
                "value": val,
                "info": info,
                "note": note,
            }
            if include_kind:
                row["kind"] = kind
            yield row

    elif categories is None and any_list_data:
        kind = "index_only"
        for si, s in enumerate(series):
            if not isinstance(s, dict):
                continue
            sname = s.get("name")
            data = s.get("data")
            if not isinstance(data, list):
                continue
            for ci, val in enumerate(data):
                row = {
                    "CODICE_SCUOLA": codice,
                    "NOME_SCUOLA": nome,
                    "endpoint_key": endpoint_key,
                    "title": title,
                    "series_index": si,
                    "series_name": sname,
                    "category_index": ci,
                    "category": None,
                    "value": val,
                    "info": info,
                    "note": note,
                }
                if include_kind:
                    row["kind"] = kind
                yield row

def build_anagrafica_base_wide_csv(out_csv: Path, raw_dir: Path, schools_map: Dict[str, str]) -> int:
    written = 0
    header_written = False
    fieldnames: List[str] = ["CODICE_SCUOLA", "NOME_SCUOLA"]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer: Optional[csv.DictWriter] = None

        for codice, nome in schools_map.items():
            p = raw_dir / safe_filename(codice) / "anagrafica_base.json"
            obj = load_json(p)
            if not isinstance(obj, dict):
                continue
            scuole = obj.get("scuole") or []
            if not scuole or not isinstance(scuole[0], dict):
                continue

            s0 = scuole[0]
            row: Dict[str, Any] = {"CODICE_SCUOLA": codice, "NOME_SCUOLA": nome}
            for k, v in s0.items():
                row[str(k)] = v
            if "esito" in obj:
                row["esito"] = obj.get("esito")
            if "numeroTotaleElementi" in obj:
                row["numeroTotaleElementi"] = obj.get("numeroTotaleElementi")

            if not header_written:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                header_written = True

            assert writer is not None
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            written += 1

    return written

def build_observations_semantic_csv(out_csv: Path, raw_dir: Path, schools_map: Dict[str, str], include_kind: bool) -> int:
    written = 0
    fieldnames = OBS_FIELDNAMES if include_kind else [c for c in OBS_FIELDNAMES if c != "kind"]

    endpoint_set = {k for k, _ in ENDPOINTS if k != "anagrafica_base"}

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for codice, nome in schools_map.items():
            school_dir = raw_dir / safe_filename(codice)
            if not school_dir.exists():
                continue
            for ep in endpoint_set:
                p = school_dir / f"{safe_filename(ep)}.json"
                obj = load_json(p)
                if not isinstance(obj, dict):
                    continue
                for row in iter_observations_from_obj(obj, codice, nome, ep, include_kind=include_kind):
                    if not include_kind:
                        row.pop("kind", None)
                    w.writerow({k: row.get(k, "") for k in fieldnames})
                    written += 1

    return written


# --- async orchestration ---

@dataclass
class Job:
    codice: str
    nome: str
    endpoint_key: str
    template: str

async def progress_reporter(
    *,
    start_ts: float,
    queue: asyncio.Queue,
    totals: Dict[str, int],
    targets: Dict[str, int],
    report_every: int,
    stop_event: asyncio.Event,
    lock: asyncio.Lock,
) -> None:
    prev_done = 0
    prev_ts = start_ts
    while not stop_event.is_set():
        await asyncio.sleep(max(1, report_every))
        async with lock:
            req_done = totals.get("requests_total", 0)
            ok = totals.get("requests_ok", 0)
            http_err = totals.get("requests_http_error", 0)
            exc = totals.get("requests_exc", 0)
            from_cache = totals.get("from_cache", 0)
            n429 = totals.get("requests_429", 0)
            schools_done = totals.get("schools_done", 0)
            schools_total = totals.get("schools_total", 0)
            schools_skipped = totals.get("schools_skipped", 0)
            target_schools = targets.get("target_schools", 0)
            req_total_est = targets.get("requests_total_est", 0)

        now = time.time()
        dt = max(1e-6, now - prev_ts)
        rate = (req_done - prev_done) / dt
        prev_done = req_done
        prev_ts = now
        elapsed = int(now - start_ts)

        qsz = queue.qsize()
        # ETA grezza su requests
        eta_s = None
        if req_total_est > 0 and rate > 0:
            remaining = max(0, req_total_est - req_done)
            eta_s = int(remaining / rate)

        eta_txt = f" eta={eta_s}s" if eta_s is not None else ""
        
        schools_pct = (schools_done / schools_total * 100.0) if schools_total else 0.0
        req_pct = (req_done / req_total_est * 100.0) if req_total_est else 0.0

        print(
            "[PROGRESS]"
            f" t={elapsed}s"
            f"schools_done={schools_done}/{schools_total} ({schools_pct:.2f}%) "
            f"req_done={req_done}/{req_total_est} ({req_pct:.2f}%) "
            f" ok={ok} http_err={http_err} exc={exc} 429={n429} cache={from_cache}"
            f" rate={rate:.2f} req/s"
            f" q={qsz}"
            f"{eta_txt}"
        )

async def worker(
    name: str,
    queue: asyncio.Queue[Optional[Job]],
    session: aiohttp.ClientSession,
    inflight_sem: asyncio.Semaphore,
    raw_dir: Path,
    blocks_dir: Path,
    results_by_school: Dict[str, List[ResultRow]],
    school_done_counts: Dict[str, int],
    expected_per_school: int,
    totals: Dict[str, int],
    lock: asyncio.Lock,
    *,
    skip_existing: bool,
    fresh: bool,
    retries: int,
    base_backoff: float,
    retry_4xx: bool,
    progress: bool,
) -> None:
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            return

        rr = await fetch_one_endpoint(
            session=session,
            inflight_sem=inflight_sem,
            raw_dir=raw_dir,
            codice=job.codice,
            nome=job.nome,
            endpoint_key=job.endpoint_key,
            template=job.template,
            skip_existing=skip_existing,
            fresh=fresh,
            retries=retries,
            base_backoff=base_backoff,
            retry_4xx=retry_4xx,
        )

        async with lock:
            results_by_school.setdefault(job.codice, []).append(rr)

            # contatori
            totals["requests_total"] += 1
            if rr.from_cache:
                totals["from_cache"] += 1
            if rr.status == -1:
                totals["requests_exc"] += 1
            elif rr.ok:
                totals["requests_ok"] += 1
            else:
                totals["requests_http_error"] += 1
                if rr.status == 429:
                    totals["requests_429"] += 1

            # progress per scuola
            school_done_counts[job.codice] = school_done_counts.get(job.codice, 0) + 1
            done_now = (school_done_counts[job.codice] >= expected_per_school)

            # scuola completata -> checkpoint immediato
            if done_now:
                rows = results_by_school.get(job.codice, [])
                write_school_checkpoint(blocks_dir, job.codice, rows)
                results_by_school.pop(job.codice, None)
                totals["schools_done"] += 1

        if progress:
            src = "CACHE" if rr.from_cache else "HTTP"
            print(
                f"[{name}] {src} codice={rr.CODICE_SCUOLA} "
                f"ep={rr.endpoint_key} status={rr.status} "
                f"url={rr.url} final_url={rr.final_url} "
                f"bytes={rr.raw_len} err={rr.error}"
            )

        queue.task_done()


async def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    blocks_dir = outdir / "blocks_partial"
    if args.fresh and blocks_dir.exists():
        shutil.rmtree(blocks_dir)

    df = read_schools_csv(Path(args.input), sep=args.sep)

    # Filtri territoriali
    if args.areageografica:
        df = df[df["AREAGEOGRAFICA"] == args.areageografica]
    if args.regione:
        df = df[df["REGIONE"] == args.regione]
    if args.provincia:
        df = df[df["PROVINCIA"] == args.provincia]

    if args.school:
        df = df[df["CODICE_SCUOLA"] == args.school]
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    schools = df.to_dict(orient="records")
    schools_map = {str(s["CODICE_SCUOLA"]).strip(): str(s["NOME_SCUOLA"]).strip() for s in schools}

    done = get_done_schools(blocks_dir)
    release_id = args.release_id.strip() or datetime.now().strftime("%Y-%m-%d")

    expected_per_school = len(ENDPOINTS)

    # target schools (not already done)
    target_schools: List[Tuple[str, str]] = []
    schools_skipped = 0
    for s in schools:
        codice = str(s["CODICE_SCUOLA"]).strip()
        nome = str(s["NOME_SCUOLA"]).strip()
        if not codice:
            continue
        if safe_filename(codice) in done and not args.fresh:
            schools_skipped += 1
            continue
        target_schools.append((codice, nome))

    # Queue + state shared
    queue: asyncio.Queue[Optional[Job]] = asyncio.Queue(maxsize=max(100, args.queue_max))
    results_by_school: Dict[str, List[ResultRow]] = {}
    school_done_counts: Dict[str, int] = {}
    lock = asyncio.Lock()

    inflight_sem = asyncio.Semaphore(max(1, args.inflight))
    conn_limit = max(1, args.conn_limit)

    connector = aiohttp.TCPConnector(
        limit=conn_limit,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    timeout = _timeout(args)

    totals: Dict[str, int] = {
        "schools_total": len(schools),
        "schools_skipped": schools_skipped,
        "schools_done": 0,
        "requests_total": 0,
        "requests_ok": 0,
        "requests_http_error": 0,
        "requests_exc": 0,
        "requests_429": 0,
        "from_cache": 0,
    }

    targets = {
        "target_schools": len(target_schools),
        "requests_total_est": len(target_schools) * expected_per_school,
    }

    start_ts = time.time()
    stop_event = asyncio.Event()

    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, connector=connector, timeout=timeout) as session:
        # Reporter
        reporter_task = None
        if args.report_every > 0:
            reporter_task = asyncio.create_task(
                progress_reporter(
                    start_ts=start_ts,
                    queue=queue,
                    totals=totals,
                    targets=targets,
                    report_every=args.report_every,
                    stop_event=stop_event,
                    lock=lock,
                )
            )

        workers = [
            asyncio.create_task(
                worker(
                    f"w{i+1}",
                    queue,
                    session,
                    inflight_sem,
                    raw_dir,
                    blocks_dir,
                    results_by_school,
                    school_done_counts,
                    expected_per_school,
                    totals,
                    lock,
                    skip_existing=args.skip_existing,
                    fresh=args.fresh,
                    retries=args.retries,
                    base_backoff=args.backoff,
                    retry_4xx=args.retry_4xx,
                    progress=args.progress,
                )
            )
            for i in range(max(1, args.workers))
        ]

        for codice, nome in target_schools:
            for endpoint_key, template in ENDPOINTS:
                await queue.put(Job(codice=codice, nome=nome, endpoint_key=endpoint_key, template=template))

        # sentinelle di stop
        for _ in workers:
            await queue.put(None)

        await queue.join()
        for t in workers:
            await t

        stop_event.set()
        if reporter_task is not None:
            try:
                await reporter_task
            except asyncio.CancelledError:
                pass

    # flush eventuali risultati rimasti in RAM
    remaining = list(results_by_school.items())
    for codice, rows in remaining:
        write_school_checkpoint(blocks_dir, codice, rows)
        results_by_school.pop(codice, None)

    # endpoints catalog
    catalog_path = outdir / "endpoints_catalog.csv"
    with open(catalog_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "endpoint_key", "title", "url_template"])
        w.writeheader()
        for k, tpl in ENDPOINTS:
            w.writerow({"source": "sic_unica", "endpoint_key": f"sic:{k}", "title": k, "url_template": tpl})

    # postprocess
    anag_written = 0
    obs_written = 0
    if not args.no_postprocess:
        if not args.no_anagrafica:
            anag_written = build_anagrafica_base_wide_csv(outdir / "anagrafica_base_wide.csv", raw_dir, schools_map)
        if not args.no_observations:
            obs_written = build_observations_semantic_csv(
                outdir / "observations_semantic.csv",
                raw_dir,
                schools_map,
                include_kind=(not args.no_kind),
            )

    manifest = {
        "release_id": release_id,
        "source": "sic_unica",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "csv": str(Path(args.input)),
            "n_schools_after_filter": len(schools),
            "target_schools": targets["target_schools"],
            "limit": args.limit,
            "single_school": args.school or None,
            "filters": {
                "areageografica": args.areageografica or None,
                "regione": args.regione or None,
                "provincia": args.provincia or None,
            },
        },
        "run": {
            "workers": args.workers,
            "inflight": args.inflight,
            "conn_limit": args.conn_limit,
            "queue_max": args.queue_max,
            "timeout": {
                "total": args.timeout_total,
                "connect": args.timeout_connect,
                "read": args.timeout_read,
            },
            "retries": args.retries,
            "backoff": args.backoff,
            "retry_4xx": bool(args.retry_4xx),
            "skip_existing": bool(args.skip_existing),
            "fresh": bool(args.fresh),
            "report_every": args.report_every,
        },
        "totals": totals,
        "outputs": {
            "raw_dir": str(raw_dir),
            "blocks_dir": str(blocks_dir),
            "endpoints_catalog": str(catalog_path),
            "anagrafica_base_wide_csv": None if args.no_postprocess or args.no_anagrafica else str(outdir / "anagrafica_base_wide.csv"),
            "observations_semantic_csv": None if args.no_postprocess or args.no_observations else str(outdir / "observations_semantic.csv"),
            "anagrafica_rows": anag_written,
            "observations_rows": obs_written,
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK")
    print("outdir:", outdir)
    print("raw:", raw_dir)
    print("blocks:", blocks_dir)
    print("endpoints_catalog:", catalog_path)
    if not args.no_postprocess and not args.no_anagrafica:
        print("anagrafica_base_wide.csv:", outdir / "anagrafica_base_wide.csv", f"(rows={anag_written})")
    if not args.no_postprocess and not args.no_observations:
        print("observations_semantic.csv:", outdir / "observations_semantic.csv", f"(rows={obs_written})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="input.csv")
    p.add_argument("--sep", default=";", help="separatore CSV input (default ;)")
    p.add_argument("--outdir", default="out_scuolainchiaro")

    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--school", default="", help="se valorizzato, processa solo questo CODICE_SCUOLA")

    # Filtri territoriali
    p.add_argument("--areageografica", default="")
    p.add_argument("--regione", default="")
    p.add_argument("--provincia", default="")

    # Async tuning
    p.add_argument("--workers", type=int, default=8, help="numero worker async")
    p.add_argument("--inflight", type=int, default=16, help="richieste HTTP contemporanee")
    p.add_argument("--conn-limit", type=int, default=32, help="limite connessioni TCP totali")
    p.add_argument("--queue-max", type=int, default=2000, help="max job in coda (backpressure)")

    # Timeout granulari
    p.add_argument("--timeout-total", type=int, default=60)
    p.add_argument("--timeout-connect", type=int, default=15)
    p.add_argument("--timeout-read", type=int, default=45)

    # Retry
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--backoff", type=float, default=0.8)
    p.add_argument("--retry-4xx", action="store_true", help="sconsigliato: retry anche su 4xx")

    p.add_argument("--skip-existing", action="store_true", help="usa cache raw se presente")
    p.add_argument("--fresh", action="store_true", help="cancella checkpoint e forza run")

    p.add_argument("--no-anagrafica", action="store_true")
    p.add_argument("--no-observations", action="store_true")
    p.add_argument("--no-kind", action="store_true")
    p.add_argument("--no-postprocess", action="store_true")

    p.add_argument("--progress", action="store_true", help="log per richiesta (molto verboso)")
    p.add_argument("--report-every", type=int, default=30, help="stampa progress aggregato ogni N secondi (0=disabilita)")
    p.add_argument("--release-id", default="", help="identificatore snapshot (default: YYYY-MM-DD)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(run(args))
