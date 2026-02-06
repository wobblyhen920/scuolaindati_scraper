#!/usr/bin/env python3
"""
Scraper per le pagine "panoramica" di Scuola in Chiaro / UNICA.

Per ogni scuola in input, visita la pagina e legge le cifre dai div
.sic-chisiamo-panoramica-card-cifra e .sic-chisiamo-panoramica-card-cifra-last.
CIFRA_LAST può contenere decimali in formato italiano (es. "23,4").

Uso:
  python sic_scraper_vw.py --input scuole.csv --out kpi.csv --sep ";" --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import re
from datetime import datetime, timezone
from urllib.parse import quote

import aiohttp
import pandas as pd
pd.options.mode.string_storage = "python"
from bs4 import BeautifulSoup
from tqdm import tqdm

# cifra intera
DIGITS_RE = re.compile(r"(\d+)")
# numero formato IT: 1.234,56 o 123,45 o 123
NUM_IT_RE = re.compile(r"(-?\d{1,3}(?:\.\d{3})*(?:,\d+)?|-?\d+(?:,\d+)?)")


def _extract_int(text: str):
    if not text:
        return None
    cleaned = text.replace(".", "").replace(",", "")
    m = DIGITS_RE.search(cleaned)
    return int(m.group(1)) if m else None


def _extract_float_it(text: str):
    """Estrae un numero in formato italiano (punto migliaia, virgola decimali)."""
    if not text:
        return None
    m = NUM_IT_RE.search(text)
    if not m:
        return None
    s = m.group(1).strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def build_url(codice: str, nome_scuola: str) -> str:
    codice = str(codice).strip()
    nome_scuola = str(nome_scuola).strip()
    return (
        "https://unica.istruzione.gov.it/cercalatuascuola/istituti/"
        f"{quote(codice, safe='')}/{quote(nome_scuola, safe='')}"
    )


async def fetch_and_parse(
    session: aiohttp.ClientSession,
    url: str,
    timeout_s: int,
) -> tuple[object, object, object, str | None]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            status = resp.status
            if status != 200:
                return None, None, status, f"http_status_{status}"

            html = await resp.text()

    except asyncio.TimeoutError:
        return None, None, None, "timeout"
    except aiohttp.ClientError as e:
        return None, None, None, f"aiohttp_error:{type(e).__name__}:{e}"
    except Exception as e:
        return None, None, None, f"unexpected_error:{type(e).__name__}:{e}"

    soup = BeautifulSoup(html, "lxml")

    els_cifra = soup.select("div.sic-chisiamo-panoramica-card-cifra")
    els_last = soup.select("div.sic-chisiamo-panoramica-card-cifra-last")

    cifra = _extract_int(els_cifra[0].get_text(" ", strip=True)) if els_cifra else None
    cifra_last = _extract_float_it(els_last[0].get_text(" ", strip=True)) if els_last else None

    if cifra is None and cifra_last is None:
        return None, None, 200, "selectors_not_found_or_no_digits"

    return cifra, cifra_last, 200, None


async def worker(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    idx: int,
    nome: str,
    codice: str,
    timeout_s: int,
) -> tuple[int, dict]:
    fetched_at = datetime.now(timezone.utc).isoformat()

    if not nome or not codice:
        return idx, {
            "URL_UNICA": "",
            "CIFRA": "",
            "CIFRA_LAST": "",
            "HTTP_STATUS": "",
            "ERROR": "missing_nome_or_codice",
            "FETCHED_AT_UTC": fetched_at,
        }

    url = build_url(codice, nome)

    async with sem:
        cifra, cifra_last, status, error = await fetch_and_parse(session, url, timeout_s)

    return idx, {
        "URL_UNICA": url,
        "CIFRA": "" if cifra is None else str(cifra),
        "CIFRA_LAST": "" if cifra_last is None else str(cifra_last),
        "HTTP_STATUS": "" if status is None else str(status),
        "ERROR": error or "",
        "FETCHED_AT_UTC": fetched_at,
    }


async def run_async(
    df: pd.DataFrame,
    timeout_s: int,
    concurrency: int,
    user_agent: str,
) -> dict[int, dict]:
    sem = asyncio.Semaphore(concurrency)

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    }

    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)
    results: dict[int, dict] = {}

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = []
        for i, row in df.iterrows():
            nome = (row.get("NOME_SCUOLA") or "").strip()
            codice = (row.get("CODICE_SCUOLA") or "").strip()
            tasks.append(worker(sem, session, i, nome, codice, timeout_s))

        # progress bar
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="UNICA async scrape"):
            idx, payload = await coro
            results[idx] = payload

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV input con NOME_SCUOLA e CODICE_SCUOLA")
    ap.add_argument("--out", required=True, help="CSV output")
    ap.add_argument("--sep", default=";", help="Separatore CSV input/output (default ';')")
    ap.add_argument("--encoding", default="utf-8", help="Encoding CSV (default utf-8)")
    ap.add_argument("--timeout", type=int, default=25, help="Timeout HTTP totale (secondi)")
    ap.add_argument("--limit", type=int, default=0, help="Limita a N righe (0=tutte)")
    ap.add_argument("--concurrency", type=int, default=20, help="Numero richieste simultanee")
    ap.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
        help="User-Agent",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=args.sep, encoding=args.encoding, dtype=str)
    # forza object per evitare casino con ArrowStringArray
    df = df.astype("object")

    needed = {"NOME_SCUOLA", "CODICE_SCUOLA"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    # Pre-alloc colonne output
    n = len(df)
    for col in ["URL_UNICA", "CIFRA", "CIFRA_LAST", "HTTP_STATUS", "ERROR", "FETCHED_AT_UTC"]:
        df[col] = pd.Series([""] * n, dtype="object")

    results = asyncio.run(
        run_async(
            df=df,
            timeout_s=args.timeout,
            concurrency=max(1, int(args.concurrency)),
            user_agent=args.user_agent,
        )
    )

    for i, payload in results.items():
        for k, v in payload.items():
            df.at[i, k] = v

    df.to_csv(args.out, sep=args.sep, encoding=args.encoding, index=False)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()

