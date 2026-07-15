# Tutorial — Oscar RAG App

A complete, step-by-step guide to build and run this project from scratch on
a fresh laptop. **Time**: ~30 minutes plus one-time model downloads
(Ollama model ~5 GB, MiniLM embedding model ~90 MB).

Every step lists:

- **Do this** — the exact command / action.
- **You should see** — the expected result.
- **If it fails** — what to check.

> **Note on the dataset**: this tutorial uses our **98th Academy Awards
> (2026)** files as the running example. The pipeline itself is generic —
> to plug in a different dataset (any RDF triples + any text corpus),
> swap the files in `ontology/` and update `.env`. See
> [§14.3 Switching to a different dataset](#143-switching-to-a-different-dataset-entirely).

---

## Table of contents

1. [What you're building](#1-what-youre-building)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Python environment](#4-python-environment)
5. [Start Weaviate (Docker)](#5-start-weaviate-docker)
6. [Set up GraphDB](#6-set-up-graphdb)
7. [Install the local LLM (Ollama)](#7-install-the-local-llm-ollama)
8. [Configure `.env`](#8-configure-env)
9. [Ingest the text corpus into Weaviate](#9-ingest-the-text-corpus-into-weaviate)
10. [Command-line smoke tests](#10-command-line-smoke-tests)
11. [Run the web app](#11-run-the-web-app)
12. [Use it in the browser](#12-use-it-in-the-browser)
13. [Test via `curl`](#13-test-via-curl)
14. [Adding more data later](#14-adding-more-data-later)
15. [File layout](#15-file-layout)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. What you're building

A web app that answers questions about the **98th Academy Awards (2026)**
using two different Retrieval-Augmented Generation pipelines the user can
switch between with a radio button:

| Pipeline      | Data source                              | How retrieval works                                                                 |
| ------------- | ---------------------------------------- | ----------------------------------------------------------------------------------- |
| **Graph RAG** | `ontology/oscars.trig` (schema + data in one file) | LLM writes SPARQL from the ontology → GraphDB runs it → LLM formats the rows        |
| **Text RAG**  | `ontology/oscar.txt` (film descriptions) | Question is embedded → nearest chunks fetched from Weaviate → LLM answers from them |

The two data sources are **complementary on purpose**:

- The text file describes plots, cast, production — **no award outcomes**.
- The graph contains every winner, nomination, milestone — **no plot text**.

Each pipeline shines on different questions, which makes the comparison
demo compelling.

---

## 2. Architecture

```
                        Browser (http://localhost:5000)
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │            Flask app  (wsgi.py)           │
              │                                           │
              │   GET  /            ─▶  chat UI (Jinja)   │
              │   POST /ask/graph   ─▶  Graph RAG         │
              │   POST /ask/text    ─▶  Text RAG          │
              └────────────┬─────────────────┬────────────┘
                           │                 │
                           ▼                 ▼
                ┌──────────────────┐   ┌──────────────────┐
                │    Graph RAG     │   │     Text RAG     │
                │  (src/graph_rag) │   │  (src/text_rag)  │
                │                  │   │                  │
                │  Qwen writes     │   │  MiniLM embeds Q │
                │  SPARQL, GraphDB │   │  Weaviate returns│
                │  runs it, Qwen   │   │  top-K chunks    │
                │  formats answer  │   │  Qwen answers    │
                └────────┬─────────┘   └────────┬─────────┘
                         │                      │
                         ▼                      ▼
                 ┌──────────────┐        ┌──────────────┐
                 │   GraphDB    │        │   Weaviate   │
                 │   (:7200)    │        │    (:8080)   │
                 └──────────────┘        └──────────────┘
```

Everything runs locally. No cloud calls unless you swap the LLM yourself.

---

## 3. Prerequisites

Install these before doing anything else. Everything is free.

### 3.1 Python 3.10 or newer

**Do this:**

```bash
python3 --version
```

**You should see:** `Python 3.10.x` or higher.

**If it fails:** install from https://www.python.org/downloads/ (Windows / Linux)
or `brew install python@3.12` on macOS.

### 3.2 Docker Desktop

We use it to run Weaviate (the vector DB for Text RAG).

- **macOS / Windows**: download from https://www.docker.com/products/docker-desktop and install.
- **Linux**: install `docker-ce` and `docker-compose-plugin` from your package manager.

Start Docker Desktop, wait for it to finish booting.

**Verify:**

```bash
docker --version
docker compose version
```

**You should see:** `Docker version 24.x` (or newer) and `Docker Compose version v2.x`.

### 3.3 Ontotext GraphDB Free

Stores the RDF triples that Graph RAG queries.

- Get it here: https://graphdb.ontotext.com/ → "Download Free"
- **macOS**: download the `.dmg`, drag GraphDB to Applications, launch it.
- **Windows / Linux**: download the installer / zip, follow the on-screen wizard.

Once launched, it exposes a **Workbench UI at http://localhost:7200**.

**Verify:**

```bash
open http://localhost:7200      # macOS
# On Windows/Linux: open the URL in a browser.
```

**You should see:** the GraphDB welcome page.

### 3.4 Ollama

Runs Qwen 2.5 Coder 7B (our LLM) locally.

- Download from https://ollama.com/download
- Install and launch. Ollama starts a background service on `http://localhost:11434`.

**Verify:**

```bash
ollama --version
```

**You should see:** something like `ollama version 0.4.x`.

---

## 4. Python environment

Create an isolated Python environment so nothing conflicts with your system
packages.

**Do this:**

```bash
python3 -m venv .venv
source .venv/bin/activate           # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**You should see:** each package printing `Successfully installed …`. First
install takes a few minutes because of PyTorch (needed by SentenceTransformer).

**If it fails:** on Apple Silicon Macs you may need `pip install --upgrade
pip setuptools wheel` first. On Windows, install "Microsoft C++ Build Tools"
if you get a compile error.

---

## 5. Start Weaviate (Docker)

Weaviate is our vector DB for Text RAG. The container is defined in
`docker-compose.yml` (already in the repo).

**Do this:**

```bash
docker compose up -d
```

**You should see:**

```
[+] Running 3/3
 ✔ Network llm-applications-unica_default  Created
 ✔ Volume  weaviate-data                   Created
 ✔ Container weaviate                      Started
```

**Verify:**

```bash
curl -s http://localhost:8080/v1/meta | jq .version
```

**You should see:** `"1.28.0"`.

**If it fails:** Docker Desktop isn't running (open it and wait), or port
8080 is taken by another process. Check with `lsof -i :8080`.

**Useful later:**

```bash
docker compose stop        # pause (data preserved in volume)
docker compose start       # resume
docker compose logs -f     # tail Weaviate logs
docker compose down        # remove container (data still preserved)
docker compose down -v     # wipe everything, including the vector index
```

---

## 6. Set up GraphDB

Open the GraphDB Workbench in your browser: **http://localhost:7200**.

### 6.1 Create the repository

1. Left sidebar → **Setup** → **Repositories**.
2. Click **Create new repository** → **GraphDB Repository**.
3. **Repository ID**: `BIP-DB`
4. Leave every other field at its default.
5. Click **Create**.

**You should see:** the new repository appears in the list.

### 6.2 Make `BIP-DB` the active repository

- Top-right dropdown (next to the account icon) → click it → select **BIP-DB**.
- The repo name appears in the top bar to confirm it's active.

### 6.3 Import the Oscar triples

1. Left sidebar → **Import** → **User data** tab.
2. Click **Upload RDF files**.
3. Select `ontology/oscars.trig` from the repo folder.
4. Click **Import** on the file's row. Wait for it to say `Imported successfully`.

**You should see:** one entry in the "Recent imports" table, green. The file contains both the schema (classes and properties) and the individuals (films, people, awards) — no separate labels file is needed.

### 6.4 Verify with a SPARQL query

1. Left sidebar → **SPARQL**.
2. Paste this query:

```sparql
PREFIX : <http://oscars2026.org#>
SELECT ?film WHERE {
  ?film :wonAward :BestPicture .
}
```

3. Click **Run** (or press Ctrl/Cmd + Enter).

**You should see:** one row: `film = http://oscars2026.org#OneBattleAfterAnother`.

**If the row is missing:** the import didn't take. Go back to step 6.3 and re-import.

---

## 7. Install the local LLM (Ollama)

Qwen 2.5 Coder 7B is the model that writes SPARQL (Graph RAG) and formats
answers (Text RAG).

**Do this:**

```bash
ollama pull qwen2.5-coder:7b
```

**You should see:** a download progress bar. About 5 GB, one-time.

**Verify:**

```bash
ollama list
```

**You should see:** a row containing `qwen2.5-coder:7b`.

**Want a bigger model?** `ollama pull qwen2.5-coder:14b` and update
`OLLAMA_MODEL` in `.env` (next step). Nothing else changes.

---

## 8. Configure `.env`

**Do this:**

```bash
cp .env.example .env
```

Open `.env` in a text editor. Confirm each value matches your setup.

| Variable                | What it does                                        | Default                                  |
| ----------------------- | --------------------------------------------------- | ---------------------------------------- |
| `GRAPHDB_URL`           | Where GraphDB listens                               | `http://localhost:7200`                  |
| `GRAPHDB_REPOSITORY`    | Repository ID you created in §6.1                   | `BIP-DB`                                 |
| `GRAPHDB_ONTOLOGY_FILE` | Ontology file the LLM reads to know the graph's shape | `ontology/oscars.trig`                 |
| `GRAPHDB_NAMESPACE`     | Namespace pinned into every generated SPARQL PREFIX | `http://oscars2026.org#`                 |
| `OLLAMA_MODEL`          | Local model tag — must match `ollama list`          | `qwen2.5-coder:7b`                       |
| `WEAVIATE_COLLECTION`   | Collection name in Weaviate                         | `OscarFilms`                             |
| `OSCAR_TEXT_FILE`       | Text corpus to embed                                | `ontology/oscar.txt`                     |
| `EMBEDDING_MODEL`       | SentenceTransformer model                           | `sentence-transformers/all-MiniLM-L6-v2` |
| `TEXT_RAG_TOP_K`        | How many chunks to retrieve per question            | `3`                                      |
| `PORT`                  | Port the Flask app listens on                       | `5000`                                   |

Save and close.

---

## 9. Ingest the text corpus into Weaviate

Now we push the film descriptions into Weaviate as embedded vectors.

**Do this:**

```bash
python -m src.ingest_text
```

**What this script does, step by step:**

1. **Reads** `ontology/oscar.txt` — one non-empty line becomes one chunk.
2. **Embeds** each chunk with SentenceTransformer (MiniLM, 384-dim vectors, normalised for cosine similarity).
3. **Drops** any existing `OscarFilms` collection in Weaviate.
4. **Recreates** the collection with no built-in vectorizer (we bring our own vectors).
5. **Batch-inserts** every chunk + its vector.

**You should see:**

```
Ingested 10 chunks into OscarFilms.
```

The MiniLM model auto-downloads (~90 MB) on the first run and is cached
under `~/.cache/huggingface/`.

**If it fails:** Weaviate is not running (`docker compose ps`), or your
`OSCAR_TEXT_FILE` path is wrong (`.env`).

---

## 10. Command-line smoke tests

Before starting the web app, confirm each pipeline works on its own.

### 10.1 Graph RAG (structured / factual questions)

```bash
python -m src.graph_rag "Which film won Best Picture?"
```

**You should see (roughly):**

```
The film that won Best Picture was One Battle After Another.
```

Try a few more:

```bash
python -m src.graph_rag "Which director won Best Director?"
python -m src.graph_rag "How many nominations did Sinners get?"
```

### 10.2 Text RAG (plot / narrative questions)

```bash
python -m src.text_rag "What is the movie Sinners about?"
```

**You should see (roughly):** a JSON blob whose `.answer` field describes
the film's plot (Mississippi Delta, vampires, twin brothers…), with a
`[doc_N]` citation.

**If either fails:** GraphDB or Weaviate is down. See §16 Troubleshooting.

---

## 11. Run the web app

**Do this:**

```bash
./run.sh
```

(Or manually: `python wsgi.py`. On Windows without bash, just run
`python wsgi.py`.)

**You should see:**

```
 * Serving Flask app 'wsgi'
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://<your-lan-ip>:5000
```

**What starts up under the hood:**

1. Flask boots on port 5000.
2. `wsgi.py` calls `build_chain()` from `src/graph_rag.py`, which:
    - Loads the ontology TTL.
    - Wires the LangChain SPARQL chain.
    - Caches it (LRU) so subsequent requests skip this cost.
3. Three routes registered:
    - `GET  /` — chat UI
    - `POST /ask/graph` — Graph RAG endpoint
    - `POST /ask/text` — Text RAG endpoint
    - `GET  /health` — `{status: ok}` for smoke tests

---

## 12. Use it in the browser

Open **http://localhost:5000**.

You'll see:

- A large empty chat area at the top.
- A grey bar at the bottom with:
    - Two radio buttons: **Graph RAG** (selected) / **Text RAG**.
    - A large text input.
    - A **Prompt!** button.

**Try:**

1. Type: _"Which film won Best Picture?"_
2. Select **Graph RAG**.
3. Click **Prompt!** (or Ctrl+Enter / Cmd+Enter).
4. Wait 5–30 seconds for the model to answer (first call is slower).

Then switch to **Text RAG** and ask: _"What is Sinners about?"_.

### 12.1 The killer demo — same question, different pipeline

Ask both pipelines _"Tell me about Sinners"_.

- **Graph RAG** answers with structured facts (director, 16 nominations, 4 wins) — because that's what's in the trig.
- **Text RAG** answers with the plot and setting — because that's what's in the corpus.

That contrast is your report's headline result.

---

## 13. Test via `curl`

**Health check:**

```bash
curl -s http://localhost:5000/health
```

→ `{"status":"ok"}`

**Graph RAG:**

```bash
curl -s -X POST http://localhost:5000/ask/graph \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which film won Best Picture?"}' | jq
```

**Text RAG:**

```bash
curl -s -X POST http://localhost:5000/ask/text \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is Sinners about?"}' | jq
```

**Response envelope (same for both endpoints):**

```json
{
  "mode": "graph" | "text",
  "question": "...",
  "answer": "...",
  "sources": {...}     // only present on /ask/text (list of retrieved chunks)
}
```

**Error responses:**

- Missing body / no question → `400 {"error": "missing 'question'"}`
- Backend blew up → `502 {"error": "<ExceptionClass>: <message>"}`

---

## 14. Adding more data later

The code is data-agnostic — nothing is hard-coded. Adding data doesn't
require any code change.

### 14.1 Add more triples (grow the knowledge graph)

1. Extend `ontology/oscars.trig` (add new individuals inside the `:oscars2026 { … }` block) or drop a new `.trig` file into `ontology/`.
2. In the GraphDB Workbench: **Import → User data → Upload RDF files** — same as §6.3.
3. If you introduced **new predicates or classes**, add them to the T-Box (schema) section at the top of `ontology/oscars.trig` so the LLM knows about them. Otherwise it will keep writing SPARQL using only the old vocabulary.
4. No restart needed. GraphDB serves the new data immediately.

### 14.2 Add more text (grow the vector corpus)

1. Append paragraphs to `ontology/oscar.txt` — one paragraph per line, blank lines ignored.
2. Re-run:
    ```bash
    python -m src.ingest_text
    ```
    This **wipes** the `OscarFilms` collection and rebuilds it with the new content.
3. No restart of the Flask app needed. The next Text RAG query hits the refreshed index.

### 14.3 Switching to a different dataset entirely

The Oscar files are a working example. To plug in any other domain (movies, drugs, football matches, papers…), the pipeline stays the same — you just swap inputs. Note the **asymmetry** between the two pipelines:

- **Graph RAG** — the triples live **inside GraphDB**, not in our repo. Our code doesn't read them from disk; it queries the SPARQL endpoint at request time. So switching graph data = importing new triples into GraphDB. The `.trig` files under `ontology/` are only there for convenience (re-importing, version control) — the pipeline doesn't need them at runtime.
- **Text RAG** — the corpus **is** a file our code reads at ingest time (`ontology/oscar.txt` by default). Switching text data = replacing that file (or pointing `OSCAR_TEXT_FILE` at another path) and re-running `python -m src.ingest_text`.

Steps:

1. **Replace the text corpus.** Put your text file anywhere; point `OSCAR_TEXT_FILE` in `.env` at it (default is `ontology/oscar.txt`). One paragraph per non-empty line becomes one chunk.
2. **Load new triples into GraphDB.** Either create a fresh repository (§6.1) or import into the existing one (§6.3). The source file can live anywhere on your machine — GraphDB reads it during import and copies it into its own storage. You don't need to keep the file in this repo afterward.
3. **Write a matching ontology file** for the LLM. Copy `ontology/oscars.trig` as a template; declare the classes and properties of your new domain in the T-Box section (ideally with `rdfs:label` / `rdfs:comment` / `rdfs:domain` / `rdfs:range` — the LLM leans on these to write good SPARQL). This one **does** need to live in the repo (or wherever `GRAPHDB_ONTOLOGY_FILE` points).
4. **Update `.env`:**
    - `GRAPHDB_REPOSITORY` → the GraphDB repo holding the new triples
    - `GRAPHDB_ONTOLOGY_FILE` → path to your new schema TTL
    - `GRAPHDB_NAMESPACE` → the `@prefix :` value from your TTL (this gets pinned into every generated SPARQL query)
    - `WEAVIATE_COLLECTION` → any name (e.g. `Movies`, `Drugs`, `MyCorpus`)
    - `OSCAR_TEXT_FILE` → path to your new text file (the env var name is legacy — feel free to rename it in `.env.example` and in the two Python files if you want)
5. **Reingest** the text: `python -m src.ingest_text`.
6. **Restart** the Flask app (`./run.sh`) so `build_chain()` picks up the new schema.

No source code change is required — everything the pipeline needs comes from `.env`, the schema TTL, and the text file.

### 14.4 Preparing your data to help the LLM

The pipeline's answer quality follows directly from how well you shape the data. A short checklist by pipeline:

**For Graph RAG (`.trig` files) — help the LLM write good SPARQL:**

- **Type every individual with `rdf:type`.** `?f a :Film` is the most natural pattern the LLM knows.
- **Give every class and property an `rdfs:label`** — the LLM matches user words to labels.
- **Add an `rdfs:comment`** on anything non-obvious (edge cases, conventions, sub-properties).
- **Use `rdfs:domain` / `rdfs:range` on properties** — these carry the strongest hints about what connects to what.
- **Keep one consistent namespace** matching `GRAPHDB_NAMESPACE` in `.env`. Mixed namespaces confuse both the LLM and reasoners.
- **Give categorical IRIs an `rdfs:label`** (e.g. `:BestPicture rdfs:label "Best Picture"`) so filter-style questions like *"who won Best Picture?"* work naturally.
- **Prefer human-readable local names** (`:MichaelBJordan`, not `:person_47`). The LLM often uses them verbatim in queries.
- **Group individuals inside one named graph** so you can query, drop, or version them together.

**For Text RAG (`oscar.txt`) — help retrieval find the right chunk:**

- **One topic per non-empty line.** Each line becomes one embedded chunk. Blank lines are ignored.
- **Make each chunk self-contained** — repeat the film / entity name inside the chunk instead of using pronouns. Retrieval sees each chunk in isolation; without the name it can't be matched to a question about that entity.
- **Include key entity names verbatim** (film titles, character names, actor names). They're the anchors semantic search leans on.
- **Keep chunks roughly 50–200 words.** Long enough to carry meaning; short enough for the LLM to reason over the top-K set without losing context.
- **Don't paste tabular / structured data as prose** (award tables, cast lists) — that's what the graph is for. Text should describe things narratively.
- **Refresh the index after every edit** — `python -m src.ingest_text` wipes and rebuilds the Weaviate collection.

---

## 15. File layout

```
LLM-Applications-UniCA/
│
├── wsgi.py                    # Flask app — routes + boot-time chain init
├── run.sh                     # Convenience: activate venv + start wsgi.py
├── docker-compose.yml         # Weaviate container definition
├── requirements.txt           # Python dependencies
├── .env / .env.example        # Configuration (secrets in .env — gitignored)
├── .gitignore                 # Excludes .env, __pycache__, .venv/
├── tutorial.md                # This file
├── README.md                  # Short project overview
│
├── src/
│   ├── graph_rag.py           # Graph RAG chain + Ollama SPARQL wrapper
│   ├── text_rag.py            # Text RAG — embed → Weaviate → LLM
│   └── ingest_text.py         # One-shot: load oscar.txt into Weaviate
│
├── ontology/
│   ├── oscars.trig            # Full ontology + dataset (schema T-Box + individuals A-Box)
│   └── oscar.txt              # Text corpus (film plots, cast, production)
│
├── templates/                 # Jinja templates (Flask + Bootstrap)
│   ├── base.html
│   └── home.html
│
└── static/
    └── js/
        └── home.js            # Chat UI: submit → POST → render answer
```

Every function in `src/*.py` and `wsgi.py` has a one-line docstring — read
them top-to-bottom to understand the flow.

---

## 16. Troubleshooting

| Symptom                                             | Cause                                                | Fix                                                                    |
| --------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------- |
| `Cannot connect to the Docker daemon`               | Docker Desktop isn't running                         | Open Docker Desktop, wait ~10s for boot                                |
| Weaviate 404 on `/v1/meta`                          | Container not started or crashed                     | `docker compose logs weaviate`                                         |
| `model 'qwen2.5-coder:7b' not found (404)`          | Ollama model not pulled                              | `ollama pull qwen2.5-coder:7b`                                         |
| `/ask/graph` returns 502 "SPARQL query is invalid"  | Small LLM produced malformed SPARQL                  | Rephrase the question, or switch to a larger model                     |
| `/ask/graph` answer says "I don't have information" | SPARQL ran but returned zero rows                    | Check that predicates in `ontology/oscars.trig` schema match individuals |
| `/ask/text` returns 502 "collection not found"      | Weaviate collection empty                            | Run `python -m src.ingest_text`                                        |
| Port `:7200` already in use                         | Native GraphDB + something else on 7200              | Stop the other; keep only GraphDB                                      |
| Port `:5000` already in use                         | Old `wsgi.py` still running / macOS AirPlay Receiver | Kill it (`lsof -i :5000` then `kill <pid>`) or change `PORT` in `.env` |
| Web page loads but button does nothing              | Static file 404                                      | Make sure `wsgi.py` was started from the repo root                     |
| `FileNotFoundError: File ontology/… does not exist` | Wrong CWD or bad `.env` path                         | Always run commands from the repo root; check `GRAPHDB_ONTOLOGY_FILE`  |

---

## What next

- Every source file has small, self-contained functions with docstrings. Read them.
- To hand this over to a teammate: they follow this file end-to-end.
