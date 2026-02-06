🔗 **Scuola in Dati (webapp):** https://scuolaindati.it/



# scuolaindati_scraper

**scuolaindati_scraper** è l’infrastruttura tecnica che alimenta *Scuola in Dati*.  
Prende dati scolastici pubblici italiani oggi dispersi in pagine web ed endpoint pensati per la consultazione manuale, e li trasforma in **dataset ordinati, versionati e riutilizzabili**. Non crea nuovi dati, non “interpreta” al posto di nessuno: **ricompone**, **normalizza** e **rende leggibile come insieme** ciò che già esiste.



## Perché questo progetto

In Italia i dati scolastici sono formalmente pubblici, ma di fatto sono distribuiti su più piattaforme, consultabili quasi sempre una scuola alla volta, sono difficili da scaricare in modo sistematico, poco adatti a confronti territoriali o temporali. Il risultato è che, pur esistendo, **non funzionano come dati**.
Il progetto di Scuola in Dati nasce per risolvere esattamente questo scarto - per rendere possibile un uso analitico, comparativo e pubblico delle informazioni scolastiche.



## Cosa fa

Questo software:

- scarica dati da **UNICA / Scuola in Chiaro** e dal **Sistema Nazionale di Valutazione (RAV/SNV)**;
- conserva sempre una copia dei dati grezzi, per controllo e riproducibilità;
- li porta in formati strutturati (long e wide);
- costruisce **release versionate**, nazionali e per macro-area geografica;
- tiene separati scraping, pulizia e pubblicazione.

Il risultato sono cartelle di dati pronte per essere analizzate, esplorate in dashboard, riutilizzate in ricerca, giornalismo o progettazione civica.



## Come è organizzato il codice

La struttura del repository riflette questa logica modulare:

`
apps/
├── _sic_scraper_vapi → dati Scuola in Chiaro (API)
├── _sic_scraper_vw → dati Scuola in Chiaro (HTML)
├── _rav_scraper → dati RAV / SNV (HTML)
└── _rav_cleaner → pulizia e normalizzazione RAV
release_roller.py → costruzione dei dataset finali
releases/ → output versionati della release
utils/ → file di supporto (ISTAT comuni, SNAI)`

Ogni componente fa una cosa sola, mentre il punto di ricomposizione è unico - `release_roller.py`.



## Le fonti utilizzate

Scuola in Dati utilizza tre canali: gli endpoint delle API che popolano i grafici di Scuola in Chiaro; le pagine web di Scuola in Chiaro; le pagine web dei RAV delle scuole. 

### Scuola in Chiaro

Il progetto usa due canali distinti:

- **endpoint Scuola in Chiaro (API)**: restituiscono dati strutturati (serie, categorie, valori numerici) per singola scuola;

- **pagine Scuola in Chiaro (HTML)**: alcune informazioni sono esposte solo nelle pagine web e vengono lette direttamente dall’HTML attraverso Selenium.



### RAV / Sistema Nazionale di Valutazione

I RAV sono una delle fonti più complesse a causa di tabelle irregolari, strutture che variano tra scuole, indicatori espressi in modi non sempre uniformi.
Il codice scarica le pagine, le trasforma in dati “long”, applica regole di pulizia esplicite prima di usarle nei dataset finali.



## Le release

Tutto confluisce in **release versionate**, nella cartella `releases/`.

Ogni release:
- ha un identificativo temporale;
- contiene un pacchetto nazionale (`ALL/`);
- può includere pacchetti per macro-area (`areas/`);
- ha un `manifest.json` che documenta cosa contiene e da dove viene.

Un symlink `releases/latest` punta sempre all’ultima release disponibile. 



## Costruire una release 

Questo è un comando tipico:

`python3 release_roller.py 
  --inputs ./apps/_sic_scraper_vapi/out_sic_* 
  --release 
  --by-area 
  --merge-snai 
  --merge-sic-studenti 
  --merge-istat-comuni`
  
In pratica prende gli output degli scraper UNICA; costruisce una nuova release; suddivide i dati anche per macro-area geografica; arricchisce con informazioni territoriali (SNAI, ISTAT comuni) e studenti da Scuola in Chiaro. Il tutto finisce in una nuova cartella sotto releases/, senza sovrascrivere nulla.



## Componenti
### 1) Scuola in Chiaro API scraper (async)

**File:**  
`apps/_sic_scraper_vapi/sic_scraper_vapi.async.v3.py`

- mantiene una lista  
  `ENDPOINTS: List[Tuple[str, str]]`  
  composta da `endpoint_key` e `url_template`  
  (molti endpoint puntano a `https://unica.istruzione.gov.it/...`);

- legge una lista di scuole da CSV:
  - flag `--input` (default: `input.csv`);
  - separatore `--sep` (default: `;`);

- richiede in input almeno queste colonne (check esplicito):
  - `CODICESCUOLA`
  - `DENOMINAZIONESCUOLA`
  - `AREAGEOGRAFICA`
  - `REGIONE`
  - `PROVINCIA`

  che vengono rinominate internamente in:
  - `CODICE_SCUOLA`
  - `NOME_SCUOLA`;

- salva sempre raw e checkpoint:
  - raw:  
    `outdir/raw/<CODICE_SCUOLA>/<endpoint>.json | .txt`
  - checkpoint per scuola:  
    `outdir/blocks_partial/<CODICE_SCUOLA>.csv`;

- espone parametri di tuning async e retry via CLI  
  (workers, inflight, conn-limit, timeout, retries, ecc.);

- produce (se non viene disabilitato il postprocess) nella `--outdir`:
  - `anagrafica_base_wide.csv`
  - `observations_semantic.csv`
  - `endpoints_catalog.csv`
  - `manifest.json`

**CLI:**

- input/output:
  - `--input` (default: `input.csv`)
  - `--sep` (default: `;`)
  - `--outdir` (default: `out_scuolainchiaro`)

- filtri:
  - `--areageografica`
  - `--regione`
  - `--provincia`
  - `--school`

- performance:
  - `--workers`
  - `--inflight`
  - `--conn-limit`
  - `--queue-max`

- timeout:
  - `--timeout-total`
  - `--timeout-connect`
  - `--timeout-read`

- retry:
  - `--retries`
  - `--backoff`
  - `--retry-4xx`

- cache / checkpoint:
  - `--skip-existing`
  - `--fresh`

- output:
  - `--no-anagrafica`
  - `--no-observations`
  - `--no-kind`
  - `--no-postprocess`



### 2) Scuola in Chiaro HTML scraper

**File:**  
`apps/_sic_scraper_vw/sic_scraper_vw.py`

- richiede un CSV di input con colonne:
  - `NOME_SCUOLA`
  - `CODICE_SCUOLA`
  (check esplicito);

- scrive un CSV di output con colonne aggiunte (pre-allocate):
  - `URL_UNICA`
  - `CIFRA`
  - `CIFRA_LAST`
  - `HTTP_STATUS`
  - `ERROR`
  - `FETCHED_AT_UTC`

**CLI:**

- `--input` (required)
- `--out` (required)
- `--sep` (default: `;`)
- `--encoding` (default: `utf-8`)
- `--timeout` (default: `25`)
- `--limit` (default: `0`)
- `--concurrency` (default: `20`)
- `--user-agent` (default: stringa Chrome)



### 3) RAV scraper (async)

**File:**  
`apps/_rav_scraper/rav_scraper.py`

- legge file con nomi hardcoded:
  - `INPUT_CSV = "input.csv"`
  - `ENDPOINTS_CSV = "endpoints.csv"`;

- in `input.csv` richiede colonne:
  - `CODICE_SCUOLA`
  - `CODICE_ISTITUTO`
  (check esplicito);

- in `endpoints.csv` richiede colonne:
  - `CODICE_EP`
  - `URL`
  (check esplicito);

- scrive output con timestamp (`RUN_TS`):
  - `output_<RUN_TS>.csv`
  - `wide_<RUN_TS>.csv`
  - `failed_schools_<RUN_TS>.csv`

- cache HTML:
  - `html/html_raw_<RUN_TS>`

- blocchi atomici:
  - `blocks_partial/`
  - `blocks_partial/meta.json`

- logging su file fisso:
  - `error_log.txt`

- modalità release:
  - se non viene passato `--no-release`, costruisce uno snapshot in  
    `--release-root / --release-id`
  - se `--release-id` non è specificato, viene usata la data ISO corrente.

**CLI:**

- `--release-root` (default: `releases`)
- `--release-id` (default vuoto → data ISO)
- `--no-release`
- `--concurrency` (default: `3`)
- `--timeout` (default: `60`)
- `--limit`
- `--school`
- `--verbose`
- `--fresh`
- `--legacy-normalization`
- `--textnorm`

**Snapshot release**`)

Dentro `releases/<release_id>/` scrive:
- `schools.csv`
- `endpoints.csv`
  - colonne rinominate in lower-case;
  - `CODICE_EP` → `endpoint_key` quando presente;
- copia `observations_semantic.csv`
- `manifest.json`

Aggiorna il symlink releases/latest → releases/<release_id>

### 4) SNV long cleaner

**File:**  
`apps/_rav_cleaner/clean_long_snv.py`

- tool che legge un CSV long e scrive un CSV long “pulito”.

**CLI:**

- `--in` (required)
- `--out` (required)
- `--delimiter` (default: autodetect tra `, ; t`)
- `--encoding` (default: `utf-8`)



## Release roller (builder)

**File:**  
`release_roller.py`

- scopre gli input cercando `observations_semantic.csv` nei path passati a `--inputs`
  (ricerca ricorsiva: `**/observations_semantic*.csv`);

- carica opzionalmente l’anagrafica cercando `anagrafica*.csv`
  negli stessi input (disattivabile con `--no-anagrafica`);

- normalizza i dati long a chunk, scrive:
  - `long/part-*.parquet`;

- legge il dataset long e costruisce un dataset wide;

- in modalità `--release`, scrive **sempre** per ogni pacchetto:
  - `long/part-*.parquet`
  - `wide.csv`
  - `wide.parquet` (compressione `zstd`, richiede `pyarrow`)
  - `manifest.json`
    (conteggi, file input, parti long, flag di merge effettivi)

### CLI 

- `--inputs` (required, `nargs +`)
- `--chunksize` (default: `100_000`)
- `--progress-every` (default: `20`)
- `--no-anagrafica`
- `--anagrafica-glob` (default: `anagrafica*.csv`)

### Modalità release

- `--release`
  - scrive in `releases/<timestamp>/`
  - non usa `--out-*`
- `--release-root` (default: `releases`)
- `--by-area`
  - crea anche `areas/<AREA>/...` oltre a `ALL`

### Merge opzionali (tutti espliciti)

- `--merge-snai`
  - `--snai-xlsx` (default: `./utils/SNAI.xlsx`)
- `--merge-istat-comuni`
  - `--istat-comuni-xlsx` (default: `./utils/Elenco-comuni-italiani.xlsx`)
- `--merge-sic-studenti`
  - `--sic-studenti-csv` (default: `./apps/_sic_scraper_vw/sic_scraper_vw_out.csv`)
- `--merge-snv`
  - `--snv-clean-csv` (default: `./apps/_rav_cleaner/observations_clean.csv`)

### Modalità non-release

Se **non** usi `--release`, il programma richiede **tutti e tre**:

- `--out-long-dir`
- `--out-wide-csv`
- `--out-wide-parquet`


## Licenza
Questo software e la documentazione sono rilasciati sotto Creative Commons Attribution–ShareAlike 4.0 (CC BY-SA 4.0): puoi copiare, modificare e riutilizzare il materiale, a condizione di attribuire l’autore e di rilasciare ogni opera derivata sotto la stessa licenza. Il riuso per ricerca, analisi critica e finalità civiche è incoraggiato. L’uso come prodotto o servizio commerciale è possibile, ma comporta responsabilità esplicite.
La licenza non si applica ai dati originari (che restano di titolarità delle rispettive amministrazioni pubbliche) ma al software utilizzato.



## Contatti
Segnalazioni, idee, collaborazioni:
scuolaindati@proton.me
