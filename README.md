# Scuola in Dati

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg)](https://pandas.pydata.org/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

🔗 **Webapp:** [scuolaindati.it](https://scuolaindati.it) | [github.com/wobblyhen920/scuolaindati_webapp](https://github.com/wobblyhen920/scuolaindati_webapp)

---

## Il problema

In Italia i dati scolastici sono formalmente pubblici. In pratica:

- sono **dispersi** tra più piattaforme (UNICA/Scuola in Chiaro, Sistema Nazionale di Valutazione/RAV, MIUR OpenData...);
- sono **consultabili una scuola alla volta**, con interfacce pensate per genitori che devono scegliere dove iscrivere i figli.
- sono **impossibili da scaricare** in modo sistematico.
- sono **inutilizzabili** per confronti, analisi, ricerca.

Il risultato è che i dati ci sono, ma **non funzionano come dati**.

---

## Cosa fa questo progetto

Scuola in Dati **ricompone** quello che già esiste. Non crea nuovi dati, non interpreta: raccoglie, normalizza e rende leggibile come insieme ciò che oggi è frammentato.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FONTI PUBBLICHE                             │
├─────────────────┬─────────────────────┬─────────────────────────────┤
│  Scuola in      │  Scuola in Chiaro   │  RAV / Sistema Nazionale    │
│  Chiaro (API)   │  (HTML)             │  di Valutazione             │
└────────┬────────┴──────────┬──────────┴──────────────┬──────────────┘
         │                   │                         │
         ▼                   ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SCRAPING + NORMALIZZAZIONE                      │
│  • Conservazione dei dati grezzi                                    │
│  • Parsing robusto tramite BeautifulSoup                            │
│  • Canonicalizzazione codici meccanografici                         │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RELEASE VERSIONATE                             │
│  • Dataset nazionale (ALL) + dataset per macro-area geoografica     │
│  • Formati CSV e Parquet                                            │
│  • Manifest con metadati                                            │
│  • Merge opzionali con Istat IDISE e classificazione SNAI           │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                      │
│  ~40.000 scuole  ·  120+ variabili  ·  CSV/Parquet                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## I dati

Il dataset copre al momento **~40.000 scuole italiane** (tutti gli ordini e gradi) con **oltre 120 variabili**:

| Categoria | Cosa contiene |
|-----------|---------------|
| **Risultati INVALSI** | Punteggi medi, variabilità tra classi, confronto con benchmark territoriali |
| **Esami di stato** | Distribuzione voti, percentuali di eccellenza, tassi di ammissione |
| **Studenti** | Iscritti, diplomati, tassi di abbandono e trasferimento |
| **Docenti** | Composizione, stabilità, tipologie contrattuali |
| **Territorio** | Comune, provincia, regione, macro-area, classificazione SNAI (aree interne) |
| **Contesto socio-economico** | Indice ESCS (background familiare degli studenti) |

### Fonti

| Fonte | Canale | Descrizione |
|-------|--------|-------------|
| **Scuola in Chiaro** | **API** | Endpoint JSON che popolano i grafici delle schede scuola |
| **Scuola in Chiaro** | **HTML** | Informazioni visibili solo nelle pagine web (via Selenium) |
| **RAV/SNV** | **HTML** | Rapporti di Autovalutazione - tabelle con indicatori INVALSI e di processo |

---

## Struttura del repository

```
scuolaindati/
│
├── apps/
│   ├── _sic_scraper_vapi/     # Scraper API Scuola in Chiaro (async)
│   ├── _sic_scraper_vw/       # Scraper HTML Scuola in Chiaro (Selenium)
│   ├── _rav_scraper/          # Scraper RAV/SNV (async)
│   └── _rav_cleaner/          # Pulizia e normalizzazione dati RAV
│
├── utils/
│   ├── SNAI.xlsx              # Classificazione aree interne
│   └── Elenco-comuni-italiani.xlsx  # Anagrafica ISTAT comuni
│
├── releases/                  # Output versionati
│   ├── latest -> 20250215_...  # Symlink all'ultima release
│   └── 20250215_.../
│       ├── ALL/               # Dataset nazionale
│       │   ├── wide.csv
│       │   ├── wide.parquet
│       │   └── manifest.json
│       └── areas/             # Dataset per macro-area
│           ├── NORD/
│           ├── CENTRO/
│           └── SUD/
│
└── release_roller.py          # Builder delle release
```

---

## Quick start

### Requisiti

```bash
Python >= 3.10
pandas >= 2.0
pyarrow
openpyxl
aiohttp
selenium  # solo per scraper HTML
```

### Installazione

```bash
git clone https://github.com/wobblyhen920/scuolaindati.git
cd scuolaindati
pip install pandas pyarrow openpyxl aiohttp
```

### Costruire una release

```bash
python release_roller.py \
  --inputs ./apps/_sic_scraper_vapi/out_sic_* \
  --release \
  --by-area \
  --merge-snai \
  --merge-istat-comuni \
  --merge-sic-studenti
```

Questo comando:
1. Trova tutti i file `observations_semantic.csv` negli input
2. Costruisce una nuova release con timestamp
3. Crea il pacchetto nazionale (`ALL/`) e quelli per macro-area
4. Arricchisce con dati territoriali (SNAI, ISTAT)
5. Scrive tutto in `releases/<timestamp>/`

---

## Componenti

### 1. Scuola in Chiaro API scraper

**File:** `apps/_sic_scraper_vapi/sic_scraper_vapi.async.v3.py`

Scraper asincrono che interroga gli endpoint JSON di UNICA/Scuola in Chiaro.

```bash
python sic_scraper_vapi.async.v3.py \
  --input scuole.csv \
  --outdir out_sic \
  --workers 10 \
  --retries 3
```

**Output:**
- `raw/<CODICE_SCUOLA>/<endpoint>.json` — risposte grezze
- `observations_semantic.csv` — dati in formato long
- `anagrafica_base_wide.csv` — anagrafica scuole
- `manifest.json` — metadati del run

<details>
<summary><strong>Tutti i parametri CLI</strong></summary>

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--input` | `input.csv` | CSV con elenco scuole |
| `--outdir` | `out_scuolainchiaro` | Directory output |
| `--workers` | auto | Worker asincroni |
| `--inflight` | auto | Richieste in parallelo |
| `--timeout-total` | 30 | Timeout totale (sec) |
| `--retries` | 2 | Tentativi per richiesta |
| `--skip-existing` | false | Salta scuole già scaricate |
| `--areageografica` | — | Filtra per area |
| `--regione` | — | Filtra per regione |
| `--provincia` | — | Filtra per provincia |

</details>

---

### 2. Scuola in Chiaro HTML scraper

**File:** `apps/_sic_scraper_vw/sic_scraper_vw.py`

Estrae informazioni visibili solo nelle pagine HTML (usa Selenium).

```bash
python sic_scraper_vw.py \
  --input scuole.csv \
  --out sic_html_out.csv \
  --concurrency 20
```

---

### 3. RAV scraper

**File:** `apps/_rav_scraper/rav_scraper.py`

Scarica i Rapporti di Autovalutazione dal Sistema Nazionale di Valutazione.

```bash
python rav_scraper.py \
  --concurrency 3 \
  --timeout 60
```

I RAV sono la fonte più complessa: tabelle irregolari, strutture che variano tra scuole, indicatori non uniformi. Lo scraper conserva sempre l'HTML grezzo per debug.

---

### 4. RAV cleaner

**File:** `apps/_rav_cleaner/clean_long_snv.py`

Pulisce e normalizza i dati RAV estratti.

```bash
python clean_long_snv.py \
  --in observations_raw.csv \
  --out observations_clean.csv
```

---

### 5. Release roller

**File:** `release_roller.py`

Assembla tutto in release versionate.

```bash
python release_roller.py \
  --inputs ./out_sic_* ./out_rav_* \
  --release \
  --by-area \
  --merge-snai \
  --merge-istat-comuni
```

<details>
<summary><strong>Tutti i parametri CLI</strong></summary>

| Parametro | Descrizione |
|-----------|-------------|
| `--inputs` | Directory/file con `observations_semantic.csv` |
| `--release` | Crea release versionata (altrimenti richiede `--out-*`) |
| `--release-root` | Directory base per le release (default: `releases`) |
| `--by-area` | Crea anche pacchetti per macro-area geografica |
| `--merge-snai` | Aggiunge classificazione SNAI (aree interne) |
| `--merge-istat-comuni` | Aggiunge dati ISTAT sui comuni |
| `--merge-sic-studenti` | Aggiunge dati studenti da HTML scraper |
| `--merge-snv` | Aggiunge dati SNV/RAV puliti |
| `--chunksize` | Righe per chunk (default: 100.000) |

</details>

---

## Formato dei dati

### Release

Ogni release contiene:

```
releases/20250215_143022/
├── manifest.json          # Metadati release
├── ALL/
│   ├── wide.csv           # Una riga per scuola, tutte le variabili come colonne
│   ├── wide.parquet       # Stesso contenuto, formato Parquet (compressione zstd)
│   ├── long/              # Formato long (più flessibile)
│   │   └── part-*.parquet
│   └── manifest.json      # Metadati pacchetto
└── areas/
    ├── NORDOVEST/
    ├── NORDEST/
    ├── CENTRO/
    ├── SUD/
    └── ISOLE/
```

### Convenzioni nomi colonne

- **Prefisso numerico** (`10_`, `20_`, ...): variabili da fonti specifiche, ordinate per tema
- **Prefisso `*`**: variabili calcolate/derivate (non presenti nelle fonti originali)
- **Prefisso `80_rav_`**: variabili dai RAV/SNV

Esempi:
```
CODICE_SCUOLA                    # Codice meccanografico
denominazione                    # Nome scuola
comune                           # Comune
tipoDiIstruzione                 # Tipo (PC=classico, PS=scientifico, ...)
10_studenti_iscritti             # Numero iscritti
20_docenti_tempo_indeterminato   # Docenti a tempo indeterminato
*40_esame_voto_share_top_pct     # % voti alti (calcolata)
80_rav_22a1_punteggio_italiano   # Punteggio INVALSI italiano
```

---

## Webapp

La webapp ([scuolaindati.it](https://scuolaindati.it)) permette di:

- **scaricare** release complete (CSV/Parquet);
- **esplorare** con filtri (tipo scuola, territorio, ricerca libera);
- **visualizzare** grafici automatici per ogni variabile;
- **esportare** sottoinsiemi personalizzati;
- **usare le API** per integrazioni programmatiche

**Qui repository della webapp: [https://github.com/wobblyhen920/scuolaindati_webapp](https://github.com/wobblyhen920/scuolaindati_webapp)**

---

## Licenza

Questo software è rilasciato sotto **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.it)**.

Puoi copiare, modificare e riutilizzare, a condizione di:
- attribuire la fonte
- rilasciare opere derivate sotto la stessa licenza

La licenza si applica al **software e alla struttura del dataset**. I dati originari restano di titolarità delle rispettive amministrazioni pubbliche.

---

## Contatti per segnalazioni, idee, collaborazioni

📧 **scuolaindati@proton.me**

