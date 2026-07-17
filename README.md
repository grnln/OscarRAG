# Oscar RAG — Graph RAG vs Text RAG

Two RAG pipelines over the **same domain** (98th Academy Awards, 2026),
exposed behind a single Flask app. Both share a local LLM (Qwen 2.5 Coder
7B via Ollama); they differ entirely in retrieval mechanism. The datasets
are non-overlapping by design (the RDF graph carries outcomes only; the
text corpus carries institutional history only), which turns the demo
into a controlled comparison between structured and unstructured
retrieval.

## Architecture

```
Browser ─▶ Flask (wsgi.py) ─┬─▶ /ask/graph ─▶ OntotextGraphDBQAChain ─▶ GraphDB (SPARQL)
                            └─▶ /ask/text  ─▶ MiniLM ──▶ Weaviate (near_vector, top-K)
                                                                          │
                                                            answer prompt ──▶ Qwen (Ollama)
```

| Component | Where | Purpose |
|---|---|---|
| Flask app | `wsgi.py` | Serves the chat UI + two RAG endpoints |
| Graph RAG chain | `src/graph_rag.py` | LangChain 3-stage chain (SPARQL gen → exec → NL synthesis) |
| Text RAG | `src/text_rag.py` + `src/ingest_text.py` | Dense retrieval over a Weaviate collection |
| GraphDB | native, `:7200` | RDF store; queried via SPARQL over HTTP |
| Weaviate | Docker, `:8080` (REST) + `:50051` (gRPC) | Vector store; local instance, no vectorizer module |
| Ollama | native, `:11434` | Serves Qwen 2.5 Coder 7B locally |

Everything is local. No cloud API calls in the default configuration.

## Graph RAG — implementation notes

Built on LangChain's `OntotextGraphDBQAChain`, which orchestrates three
stages:

1. **SPARQL generation** — LLM receives `{schema}` (serialised
   `ontology/oscars2026.trig` at boot, cached via `@lru_cache`) plus the
   user's question. Emits SPARQL.
2. **Query execution** — `OntotextGraphDBGraph.query(sparql)` hits the
   SPARQL endpoint (`rdflib` `SPARQLStore` under the hood).
3. **Answer synthesis** — LLM receives `str(rdflib.query.Result)` as
   `{context}` plus the question. Emits prose.

### Key engineering decisions

**Custom SPARQL prompt with 5 few-shot examples** (`SPARQL_PROMPT`).
The default LangChain template gives the schema + rules but no examples.
For a 7B model, in-context learning from concrete `(question, SPARQL)`
pairs measurably improves query quality — especially on `FILTER`,
aggregation, and multi-hop patterns. Examples cover: direct wins, path
traversal via `:directedBy`, numeric lookup with `:totalNominations`,
nominee enumeration with `:nominatedFor`, and `FILTER(?wins > 3)`.

**Custom QA prompt** (`QA_PROMPT`). LangChain hands the QA-stage LLM the
raw `str([(rdflib.term.Literal('X'),)])`. Frontier models parse this
fine; Qwen 7B doesn't recognise it as structured data and defaults to
"I don't know" per the built-in prompt's fallback rule. Our template
explains the rdflib repr, provides 4 shot examples showing extraction
from `Literal(...)` and `URIRef(...)`, and yields a one-sentence answer.

**`OllamaForSparql` subclass of `ChatOllama`**. Since LangChain uses one
LLM instance for both stages, we discriminate at `_generate()` time by
looking for `"Write a SPARQL"` in the messages. On stage 1 we prepend a
system prompt pinning `PREFIX : <http://oscars2026.org#>` and strip
markdown fences via regex (`_FENCE_RE`). On stage 3 the wrapper passes
through unmodified so the LLM produces prose, not more SPARQL. This
avoids maintaining two model configs.

**SPARQL export via `contextvars.ContextVar`**. The chain doesn't
surface the generated query on its return value (callback-based capture
proved unreliable across LangChain versions). We set a
`ContextVar[str]` from inside `OllamaForSparql._generate` at the point
the SPARQL is cleaned, and `wsgi.ask_graph` reads it after
`chain.invoke` returns. Per-request reset before invoke prevents leaks
across requests.

**Chain caching**. `build_chain()` is `@lru_cache(maxsize=1)`. Ontology
parse (rdflib) and LangChain wiring happen once at Flask boot;
subsequent requests reuse the warm chain. Code changes to
`graph_rag.py` require a process restart.

## Text RAG — implementation notes

Standard dense retrieval, split into offline ingest and online query.

### Ingest (`src/ingest_text.py`, run once)

- **Chunking** (`read_chunks`, `target_chars=800`). Reconstructs
  hard-wrapped input into a single stream (joins non-empty lines with
  spaces), splits on sentence boundaries via
  `re.split(r"(?<=[.!?])\s+", text)`, then greedy-packs sentences into
  ~800-char chunks. Handles both paragraph-per-line and continuous
  prose. Corpus of ~41 KB yields ~60 chunks.
- **Embedding** (`embed_all`). SentenceTransformer,
  `all-MiniLM-L6-v2` (384-dim). `normalize_embeddings=True` so cosine
  and dot-product ordering coincide.
- **Storage** (`reset_collection`, `upsert`). Drops and recreates the
  `OscarFilms` collection with
  `Configure.Vectorizer.none()` — we supply vectors; Weaviate doesn't
  need an embedding module. Per-object properties: `text`, `chunk_id`,
  `source`. Batch insert via `collection.batch.dynamic()`.

### Query (`src/text_rag.py`)

- **Retrieval** (`retrieve`). Same MiniLM encodes the question,
  `collection.query.near_vector(near_vector=v, limit=TOP_K)` returns
  top-K by cosine distance. `TOP_K=3` (env-configurable).
- **Generation** (`ask`). Chunks are formatted as `[doc_N] <text>` and
  interpolated into a prompt (`PROMPT_TEMPLATE`) with five rules:
  cite sources, admit missing info (with explicit "based on general
  knowledge" escape), surface contradictions, treat context as data
  (prompt-injection defence), skip irrelevant chunks. The LLM answer
  is returned alongside the raw `sources` for UI transparency.
- **Model + client caching**. `embedder()` and `weaviate_client()` are
  both `@lru_cache(maxsize=1)`; MiniLM loads once, gRPC connection
  stays open for the process lifetime.

## API surface

Both endpoints POST JSON `{question: string}`, return JSON.

| Route | Response shape |
|---|---|
| `GET  /` | HTML — chat UI |
| `GET  /health` | `{status: "ok"}` |
| `POST /ask/graph` | `{mode, question, answer, sparql?}` |
| `POST /ask/text` | `{mode, question, answer, sources: [{text, chunk_id, source, distance}]}` |

Errors return `{error: "<type>: <message>"}` with status 400 (missing
`question`) or 502 (chain / retrieval failure).

## Configuration (`.env`)

| Var | Default | Notes |
|---|---|---|
| `GRAPHDB_URL` | `http://localhost:7200` | |
| `GRAPHDB_REPOSITORY` | `BIP-DB` | |
| `GRAPHDB_ONTOLOGY_FILE` | `ontology/oscars2026.trig` | Read at boot as LLM's schema view |
| `GRAPHDB_NAMESPACE` | `http://oscars2026.org#` | Pinned into every SPARQL PREFIX |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Must match `ollama list` |
| `WEAVIATE_COLLECTION` | `OscarFilms` | Shared by ingest + query |
| `OSCAR_TEXT_FILE` | `ontology/oscar.txt` | Legacy name; any text file works |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | |
| `TEXT_RAG_TOP_K` | `3` | Bump to 5–7 for higher recall |
| `PORT` | `5000` | |
| `HF_TOKEN` | *(unset)* | Optional; silences HF Hub rate-limit warnings |

## Design tradeoffs

- **Local-only stack.** Cost = zero, latency = LLM-bound (~15 s per
  question on Qwen 7B). Bumping to a cloud LLM (`ChatOpenAI`,
  `ChatAnthropic`) is a one-line swap in `plain_llm()` /
  `OllamaForSparql`; quality improves but privacy is lost.
- **No `rdf:type` on individuals** in the current trig. The LLM is
  told (rule #3 of the SPARQL system prompt) to identify entities by
  predicates. Adding `rdf:type` would let the LLM use natural
  `?f a :Film` patterns; both patterns can coexist.
- **No cross-encoder rerank in Text RAG.** Top-K purely by cosine
  distance. For a corpus this small the ordering is already good;
  rerank becomes worthwhile at 10K+ chunks or on ambiguous queries.
- **No SPARQL retry loop.** `max_fix_retries` is available in
  `OntotextGraphDBQAChain` but currently disabled — invalid SPARQL
  surfaces as a 502 instead of costing more LLM calls to auto-repair.

## Repository layout

```
wsgi.py                   Flask app; single-file entry point
run.sh                    venv activate + python wsgi.py
docker-compose.yml        Weaviate 1.28.0, ports 8080 + 50051
requirements.txt          langchain-community, langchain-ollama, weaviate-client,
                          sentence-transformers, flask, rdflib
.env.example              Env template
tutorial.md               Full setup walkthrough for teammates
src/
  graph_rag.py            SPARQL_PROMPT, QA_PROMPT, OllamaForSparql, build_chain, ask
  text_rag.py             PROMPT_TEMPLATE, embedder, weaviate_client, retrieve, ask
  ingest_text.py          read_chunks, embed_all, reset_collection, upsert
ontology/
  oscars2026.trig         RDF data (winners, nominations, milestones)
  oscar.txt               Text corpus (institutional history)
templates/                Jinja templates (Flask + Bootstrap 5.3)
static/js/home.js         Chat UI (~50 LOC)
```

## Bootstrap

Full setup (fresh laptop → running demo, ~30 min) lives in
[`tutorial.md`](tutorial.md). Steady-state:

```bash
./run.sh          # http://localhost:5000
```

## Security testing

Jailbreak tests were performed on the assistant, using prompts that tried to make the assistant answer questions for which it was not designed.

Some of the tests performed included:

- Asking the assistant about awards won by someone who could not have been a recipient of an Academy Award, due to them not working in the field of cinema (such as a pop artist):

![Screenshot 1](images/screenshot_1.png)

- Trying to ask the assistant about topics not related to cinema or the academy awards (by contradicting its instructions):

![Screenshot 2](images/screenshot_2.png)

- Trying to make the assistant discard all instructions (by telling them they are an unfiltered assistant) and asking it about topics that do not appear on the document corpus:

![Screenshot 3](images/screenshot_3.png)

These tests were performed on the text RAG, as the graph RAG can only answer questions pertaining to the information stored in the knowledge graph.

These tests failed due to a lack of prior instructions inserted in the prompt that would allow the model to distinguish between topics that were covered in the document corpus and those that were not.

### Actions taken

A cross-encoder was added to filter the chunks of text that were fed to the assistant, so as to avoid answering questions that do not pertain to the contents of the document corpus.

![Screenshot 4](images/screenshot_4.png)

As can be seen in the image abbove, the assistant is now able to filter the prompts and not answer questions that are not related to the contents of the document corpus.