#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_long_snv.py

CSV long -> CSV long pulito (senza wide).

Regole:
- drop DESCRITTORE == "1.1.b.1"
- drop COLONNA == "label"

- DESCRITTORE == "2.2.a.1":
  - keep solo COLONNA in {"punteggio medio (1)", "diff. escs (2)"}
  - split MODALITA:
      SPLIT_A1 = prima di " - "
      SPLIT_A2 = tra " - " e " | "
      SPLIT_A3 = dopo " | "
  - se manca un separatore richiesto -> FATAL

- DESCRITTORE == "2.2.b.2":
  - split basato su COLONNA (MODALITA non usata)
  - rimuove prefisso "variabilit* dei punteggi - " (anche mojibake)
  - split COLONNA ripulita:
      SPLIT_B1 = prima del 1° " - "
      SPLIT_B2 = tra 1° e 2° " - "
      SPLIT_B3 = tra 2° " - " e 1° " | "
      SPLIT_B4 = tra 1° e 2° " | "
      SPLIT_B5 = resto
  - se mancano separatori richiesti -> FATAL

Output:
- rimuove in output: MODALITA, COLONNA, ORDINE_GRADO
- aggiunge: SPLIT_A1..3, SPLIT_B1..5
"""

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Tuple

KEEP_22A1 = {"punteggio medio (1)", "diff. escs (2)"}
MAX_BAD_EXAMPLES = 25
MAX_FIELD_CHARS = 240  # evita log enormi

DROP_OUTPUT_COLS = {"MODALITA", "ORDINE_GRADO", "NOME_DESCRITTORE"}


def norm_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def trunc(s: str, n: int = MAX_FIELD_CHARS) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def detect_delimiter(sample: str) -> str:
    cand = [",", ";", "\t"]
    counts = {c: sample.count(c) for c in cand}
    if max(counts.values()) == 0:
        return ","
    return max(counts, key=counts.get)


def split_22a1(modalita: str) -> Tuple[str, str, str]:
    m = modalita or ""
    if " - " not in m:
        raise ValueError('missing separator " - " in MODALITA')
    left, rest = m.split(" - ", 1)
    if " | " not in rest:
        raise ValueError('missing separator " | " after " - " in MODALITA')
    mid, right = rest.split(" | ", 1)
    return norm_space(left), norm_space(mid), norm_space(right)


def strip_var_prefix(col: str) -> str:
    """
    Rimuove un prefisso del tipo:
      "variabilitÃ  dei punteggi - "
      "variabilità dei punteggi - "
      "variabilita dei punteggi - "
    e varianti con spazi/maiuscole/mojibake.

    Strategia: se all'inizio compare "variabilit" (case-insensitive),
    elimina tutto fino a "dei punteggi - " incluso (non-greedy).
    """
    s = re.sub(r"\s+", " ", (col or "").strip())

    if not re.match(r"^\s*variabilit", s, flags=re.IGNORECASE):
        return s

    s2 = re.sub(
        r"^\s*variabilit.*?dei\s+punteggi\s*-\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s2.strip()


def split_22b2(colonna: str) -> Tuple[str, str, str, str, str]:
    s = strip_var_prefix(colonna)

    parts_dash = s.split(" - ")
    if len(parts_dash) < 3:
        raise ValueError('not enough " - " in COLONNA (need >= 2 after prefix removal)')

    b1 = parts_dash[0]
    b2 = parts_dash[1]
    tail = " - ".join(parts_dash[2:])

    parts_bar = tail.split(" | ")
    if len(parts_bar) < 3:
        raise ValueError('not enough " | " in COLONNA tail (need >= 2)')

    b3 = parts_bar[0]
    b4 = parts_bar[1]
    b5 = " | ".join(parts_bar[2:])

    return norm_space(b1), norm_space(b2), norm_space(b3), norm_space(b4), norm_space(b5)


def fatal_with_examples(msg: str, bad: List[Dict[str, str]]) -> None:
    print("\nFATAL:", msg, file=sys.stderr)
    print(f"Bad rows: {len(bad)} (show up to {MAX_BAD_EXAMPLES})", file=sys.stderr)
    for i, ex in enumerate(bad[:MAX_BAD_EXAMPLES], start=1):
        print(
            f"{i:02d} | line={ex.get('_line','?')} | CODICE_SCUOLA={ex.get('CODICE_SCUOLA','')}"
            f" | DESCRITTORE={ex.get('DESCRITTORE','')} | ERR={ex.get('__error','')}"
            f"\n     COLONNA={trunc(ex.get('COLONNA',''))}"
            f"\n     MODALITA={trunc(ex.get('MODALITA',''))}",
            file=sys.stderr,
        )
    raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV input (long)")
    ap.add_argument("--out", dest="out", required=True, help="CSV output (long pulito)")
    ap.add_argument("--delimiter", default=None, help="Delimiter input (default: autodetect tra , ; \\t)")
    ap.add_argument("--encoding", default="utf-8", help="Encoding input (default: utf-8)")
    args = ap.parse_args()

    bad_22a1: List[Dict[str, str]] = []
    bad_22b2: List[Dict[str, str]] = []

    rows_in = 0
    rows_out = 0
    drop_11b1 = 0
    drop_label = 0
    drop_22a1_other = 0

    tmp_out = args.out + ".tmp"

    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    extras = [
        "SPLIT_A1", "SPLIT_A2", "SPLIT_A3",
        "SPLIT_B1", "SPLIT_B2", "SPLIT_B3", "SPLIT_B4", "SPLIT_B5"
    ]

    try:
        with open(args.inp, "r", newline="", encoding=args.encoding, errors="replace") as f_in:
            sample = f_in.read(4096)
            f_in.seek(0)
            delim_in = args.delimiter or detect_delimiter(sample)

            reader = csv.DictReader(f_in, delimiter=delim_in)
            if not reader.fieldnames:
                raise SystemExit("Input CSV vuoto o senza header.")

            # Output schema = input schema - DROP_OUTPUT_COLS + extras
            out_fields = [c for c in reader.fieldnames if c not in DROP_OUTPUT_COLS]
            for e in extras:
                if e not in out_fields:
                    out_fields.append(e)

            with open(tmp_out, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=out_fields, delimiter=",", quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()

                for line_no, row in enumerate(reader, start=2):
                    rows_in += 1

                    descr = (row.get("DESCRITTORE") or "").strip()
                    col = (row.get("COLONNA") or "").strip()

                    if descr == "1.1.b.1":
                        drop_11b1 += 1
                        continue
                    if col == "label":
                        drop_label += 1
                        continue

                    # reset split cols
                    for e in extras:
                        row[e] = ""

                    if descr == "2.2.a.1":
                        if col not in KEEP_22A1:
                            drop_22a1_other += 1
                            continue
                        try:
                            a1, a2, a3 = split_22a1(row.get("MODALITA") or "")
                            row["SPLIT_A1"], row["SPLIT_A2"], row["SPLIT_A3"] = a1, a2, a3
                        except Exception as e:
                            bad_22a1.append({
                                "_line": str(line_no),
                                "CODICE_SCUOLA": row.get("CODICE_SCUOLA", "") or "",
                                "DESCRITTORE": descr,
                                "COLONNA": row.get("COLONNA", "") or "",
                                "MODALITA": row.get("MODALITA", "") or "",
                                "__error": str(e),
                            })
                            continue

                    elif descr == "2.2.b.2":
                        try:
                            b1, b2, b3, b4, b5 = split_22b2(row.get("COLONNA") or "")
                            row["SPLIT_B1"], row["SPLIT_B2"], row["SPLIT_B3"], row["SPLIT_B4"], row["SPLIT_B5"] = b1, b2, b3, b4, b5
                        except Exception as e:
                            bad_22b2.append({
                                "_line": str(line_no),
                                "CODICE_SCUOLA": row.get("CODICE_SCUOLA", "") or "",
                                "DESCRITTORE": descr,
                                "COLONNA": row.get("COLONNA", "") or "",
                                "MODALITA": row.get("MODALITA", "") or "",
                                "__error": str(e),
                            })
                            continue

                    # costruisce la riga output eliminando i campi vietati
                    out_row = {k: row.get(k, "") for k in out_fields}
                    writer.writerow(out_row)
                    rows_out += 1

        if bad_22a1:
            fatal_with_examples("DESCRITTORE=2.2.a.1: righe non splittabili.", bad_22a1)
        if bad_22b2:
            fatal_with_examples("DESCRITTORE=2.2.b.2: righe non splittabili.", bad_22b2)

        os.replace(tmp_out, args.out)

    finally:
        if os.path.exists(tmp_out) and not os.path.exists(args.out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass

    print("OK")
    print(f"input_rows={rows_in}")
    print(f"output_rows={rows_out}")
    print(f"dropped_descr_1.1.b.1={drop_11b1}")
    print(f"dropped_col_label={drop_label}")
    print(f"dropped_2.2.a.1_other_columns={drop_22a1_other}")
    print(f"delimiter_in={delim_in!r}")
    print(f"encoding_in={args.encoding}")
    print("dropped_output_cols=" + ",".join(sorted(DROP_OUTPUT_COLS)))


if __name__ == "__main__":
    main()
