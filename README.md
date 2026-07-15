# Oscar RAG — Graph RAG vs Text RAG

## The idea

A single web app that answers the **same natural-language questions** against
the **same domain** (the 98th Academy Awards, 2026) using **two different
Retrieval-Augmented Generation pipelines** side by side. The user picks
which pipeline to use with a radio button, so the comparison is direct.

We built the two datasets to be **complementary on purpose**: the RDF
knowledge graph holds every award outcome (winners, nominations, milestones)
but no film plots; the text corpus describes every film's plot and cast but
mentions no outcomes. Each pipeline shines on different questions, which
makes the demo tell a clear story about what each retrieval style is good
at.

## The architecture

```
Browser ─▶ Flask (wsgi.py) ─┬─▶ /ask/graph ─▶ Qwen writes SPARQL ─▶ GraphDB ─▶ Qwen formats answer
                            └─▶ /ask/text  ─▶ MiniLM embeds Q  ─▶ Weaviate top-K ─▶ Qwen answers
```

Two independent pipelines behind the same Flask app. Both use the **same
local LLM** (Qwen 2.5 Coder 7B via Ollama), but Graph RAG wraps it to force
clean SPARQL output while Text RAG uses it plainly for prose generation.
Everything runs locally — no cloud API calls.

## How Graph RAG works (`src/graph_rag.py`)

Graph RAG is a **three-stage chain**, orchestrated by LangChain's
`OntotextGraphDBQAChain`:

1. **SPARQL generation.** The LLM receives the ontology
   (`ontology/oscars.trig` — classes, properties, `rdfs:label`,
   `rdfs:domain`, `rdfs:range`) as its "map of the graph", plus the
   user's question. It writes a SPARQL query grounded in that map.
2. **Query execution.** The generated query is sent to GraphDB's
   SPARQL endpoint (`http://localhost:7200/repositories/<repo>`).
   GraphDB executes it against the loaded triples and returns
   structured rows (bindings).
3. **Answer formatting.** The LLM sees the rows and the original
   question, and writes a natural-language reply grounded in those
   exact rows — no fabrication.

Example — the question *"Which film won Best Picture?"* becomes:

```sparql
PREFIX : <http://oscars2026.org#>
SELECT ?f WHERE { ?f :wonAward :BestPicture . }
```

GraphDB returns one row (`:OneBattleAfterAnother`), and the LLM turns it
into *"The film that won Best Picture was One Battle After Another."*

**Two engineering details that matter:**

- `OllamaForSparql` — a small `ChatOllama` subclass that only kicks in on
  the SPARQL-generation stage (detected by a marker in LangChain's prompt).
  It prepends a system message pinning `PREFIX : <http://oscars2026.org#>`
  and strips markdown fences that small models like Qwen often add.
  The answer-formatting stage uses the plain LLM.
- The chain is built once at Flask boot (`@lru_cache`), so the ontology
  is loaded and the LangChain graph client is warm before the first
  request lands.

## How Text RAG works (`src/text_rag.py` + `src/ingest_text.py`)

Text RAG is classic **dense retrieval**, split into an offline ingest step
and an online query step.

**Ingest (one-shot, `python -m src.ingest_text`):**

1. Read `ontology/oscar.txt` — one paragraph per non-empty line becomes
   one chunk.
2. Encode each chunk with SentenceTransformer
   (`all-MiniLM-L6-v2` → 384-dim vectors, normalised for cosine similarity).
3. Drop the previous `OscarFilms` collection in Weaviate and recreate it
   (with `Configure.Vectorizer.none()` — we supply our own vectors).
4. Batch-insert every `{text, chunk_id, source} + vector` pair.

**Query (every `/ask/text` request):**

1. Embed the user's question with the same SentenceTransformer.
2. `collection.query.near_vector(top_k=3)` — Weaviate returns the three
   nearest chunks by cosine distance.
3. Format those chunks as numbered passages (`[doc_0]`, `[doc_1]`, `[doc_2]`)
   and drop them into a prompt template. The prompt instructs the LLM to
   answer **only** from the passages, cite the source ID, and say
   *"The passages do not contain this information"* if they don't. This
   prevents the model from silently falling back on its training memory.
4. The LLM's reply is returned to the browser along with the retrieved
   chunks as `sources` — so the frontend can show what evidence the answer
   was grounded on.

The SentenceTransformer model and Weaviate client are both `@lru_cache`d
so the model loads once and connections stay open for the lifetime of the
Flask process.

## How the web app wires it up (`wsgi.py`)

A single Flask app exposes three routes:

- `GET  /`             — renders the chat UI (`templates/home.html`).
- `POST /ask/graph`    — passes the question through the Graph RAG chain.
- `POST /ask/text`     — passes the question through the Text RAG pipeline.

Both `/ask/*` endpoints return the same JSON envelope
(`{mode, question, answer, sources?}`) so the frontend can render either
response identically. The chat UI (`static/js/home.js`, ~40 lines) reads
the input + selected mode, POSTs to `/ask/${mode}`, and renders the reply
as a chat bubble.

**Infrastructure:** Weaviate runs in a Docker container defined in
`docker-compose.yml` (`localhost:8080`), GraphDB runs natively
(`localhost:7200`), and Ollama serves Qwen locally (`localhost:11434`).

## Setup

Full walkthrough (fresh laptop → working demo in ~30 min) is in
[`tutorial.md`](tutorial.md). Once set up, one command starts everything:

```bash
./run.sh          # http://localhost:5000
```
