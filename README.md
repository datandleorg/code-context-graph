# Code Context Graph (CCG)

Production-grade code indexing and search with two-stage retrieval (OpenAI embeddings + cross-encoder re-ranking) and graph-augmented RAG.

**Contents:** [Install](#install) · [Environment](#environment) · [Architecture](#architecture-overview) · [Vector storage](#vector-storage-default-vs-qdrant) · [CLI reference](#cli-reference) · [API](#api-server-fastapi) · [Search pipeline](#search-api--pipeline-how-it-works) · [Codebase](#codebase-overview)

---

## Install

From this directory:

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
# or: uv sync
```

**Note:** `tree-sitter` is pinned to `<0.22` so that `tree-sitter-languages` (used for Python, JavaScript, TypeScript, Java scope trees and graph edges) works. Newer tree-sitter versions use a different API and would result in 0 edges.

## Environment

Embeddings use **OpenAI** only. Set your API key via a `.env` file (loaded automatically by the CLI and API server) or export it:

```bash
# Copy the example and add your key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

`**.env.example`:**

```env
# Required for embeddings (ingest, search, agent, serve)
OPENAI_API_KEY=sk-your-openai-api-key

# Optional: where named indexes are stored (default: ./ccg-indexes)
# CCG_INDEX_ROOT=./ccg-indexes
```

- **CLI** (`main.py ingest|search|agent|watch|serve`) and **FastAPI server** both load `.env` from the project root via `python-dotenv`.
- You can still pass `--openai-api-key` (CLI) or `openai_api_key` in request bodies to override.
- **Embedding model:** default is `text-embedding-3-small`; override with `--embedding-model` (CLI) or `embedding_model` in the ingest/search config.

---

## Architecture overview

CCG has three layers:

1. **Ingest** — Parse code (Tree-sitter AST) → build scope nodes and a **code graph** (CALLS, USES_TYPE, SIBLING) → embed node text with **OpenAI** → store nodes in **SQLite** (shadow index), graph in **JSON**, and vectors in either **on-disk files** (default) or **Qdrant** (optional).
2. **Search** — Embed query → **vector search** (top `initial_k`) → **rerank** (cross-encoder, top `top_k`) → **graph expansion** (one-hop or multi-hop BFS) → return context string and/or **references** (full node object per hit: id, type, path, name, class_name, signature, line_start, line_end; no content).
3. **API / Agent** — FastAPI server exposes ingest/search/summarize; CLI `agent` runs an interactive ReAct agent that uses search as a tool.

Data flow:

- **Parser** (`ccg/parser`) → **ScopeNode** list  
- **CodeGraph** (`ccg/graph`) ← nodes + edges from parser  
- **ShadowIndex** (`ccg/shadow_index`) ← nodes (SQLite), edges (SQLite)  
- **VectorStore** (`ccg/vector_store`) ← embeddings: either **in-memory + save/load to index_dir/vectors/** or **Qdrant** when `qdrant_url` is set  
- **Collector** (`ccg/collector`) runs vector search → rerank → graph expansion and returns (context, references)

### Index directory layout

Each index lives in a single directory (either `<repo>/.ccg` or `CCG_INDEX_ROOT/<id>`). Contents:


| Path                       | Description                                                                                                                                                            |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ccg.db`                   | SQLite DB: `nodes` table (id, path, type, name, class_name, content, ghost_text, signature, line_start, line_end, …), `edges` table (source_id, target_id, edge_type). |
| `graph.json`               | Code graph: nodes and edges (CALLS, USES_TYPE, SIBLING) for load/save.                                                                                                 |
| `vectors/`                 | Used only when **not** using Qdrant. See [Vector storage](#vector-storage-default-vs-qdrant).                                                                          |
| `vectors/vectors.npy`      | NumPy array of embedding vectors (float32).                                                                                                                            |
| `vectors/vector_ids.json`  | JSON array of node IDs in same order as rows in `vectors.npy`.                                                                                                         |
| `vectors/vector_meta.json` | `model`, `dimensions`, `embedding_provider` (for matching at query time).                                                                                              |
| `manifest.json`            | Content hashes per file + root hash for incremental ingest.                                                                                                            |
| `ast/`                     | Optional AST JSON files per parsed file (path with `/` → `_`_).                                                                                                        |


---

## Vector storage: default vs Qdrant

**Default (no Qdrant):** Vectors are stored **in process memory** and **persisted under the index** in `index_dir/vectors/` as `vectors.npy`, `vector_ids.json`, and `vector_meta.json`. At ingest, `VectorStore.upsert()` adds to `_in_memory_vectors` and then `save_to_dir(vectors_dir)` writes to disk. At search, `load_from_dir(vectors_dir)` loads them back into memory and search is cosine similarity over that list. No external service is required.

**Why Qdrant is optional:** Qdrant is a separate vector DB. You use it when you want:

- **Scale** — Millions of vectors; approximate nearest-neighbor (ANN) instead of linear scan.
- **Persistence** — Vectors live in Qdrant; no need to load full index into memory at search time.
- **Multiple processes** — Several workers or services can share the same Qdrant collection.

If you don’t set `qdrant_url`, CCG works out of the box with on-disk + in-memory vectors. You do **not** need Qdrant for normal use.

**How to use Qdrant:** Pass `qdrant_url` in the ingest/search **config** (e.g. `http://localhost:6333`). The CLI and the HTTP API request bodies do not currently expose `qdrant_url`; use programmatic calls such as `ingest_codebase(..., config={"qdrant_url": "http://localhost:6333"})` and `search_codebase(..., config={"qdrant_url": "...", "index_id": "..."})`. When `qdrant_url` is set:

- **Ingest:** Vectors are sent to Qdrant (collection name `ccg` by default); `save_to_dir` is a no-op.
- **Search:** VectorStore connects to Qdrant and runs `query_points`; `load_from_dir` is a no-op.
- Node IDs are stored in the point payload as `node_id`; dimensions and distance (cosine) match the embedding model.

**Using another vector DB:** The vector layer is abstracted in `ccg/vector_store.py` as the `VectorStore` class. It currently has two backends: in-memory (+ save/load to dir) and Qdrant. To plug in another store (e.g. Pinecone, Weaviate, Chroma), you would:

1. Implement the same interface: `embed(texts)`, `upsert(node_ids, texts)`, `search(query_vector, k)`, `delete_ids(node_ids)`, and optionally `load_from_dir`/`save_to_dir` (or no-op if the DB is remote).
2. Extend `VectorStore` (or add a new class) that uses your client and pass it into the ingest/search path. The runner and collector only need a store that supports `embed_single`, `search`, `upsert`, and `delete_ids`; they do not depend on Qdrant-specific types.

---

## CLI reference

All commands are run as `python main.py <command> [options]`. `.env` is loaded automatically (see [Environment](#environment)).

### Command: `ingest`

Index a codebase. First run is full; subsequent runs are incremental (only changed files) unless `--full` is used.


| Option              | Description                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| `root_path`         | (positional) Root directory to index.                                                                   |
| `--id`              | Named index ID. Index is stored under `CCG_INDEX_ROOT/<id>`. Required for `agent` and for search by ID. |
| `--index-dir`       | Override index directory (default: `<root>/.ccg` or `CCG_INDEX_ROOT/<id>` if `--id` is set).            |
| `--openai-api-key`  | OpenAI API key (overrides `OPENAI_API_KEY` env).                                                        |
| `--embedding-model` | OpenAI embedding model (default: `text-embedding-3-small`).                                             |
| `--full`            | Force full re-ingest; ignore manifest and incremental logic.                                            |


**Examples:**

```bash
python main.py ingest /path/to/repo
python main.py ingest /path/to/repo --id frontend
python main.py ingest /path/to/repo --id my-backend --full
```

### Command: `watch`

Watch a directory and run incremental ingest after file changes (debounced).


| Option              | Description                                                           |
| ------------------- | --------------------------------------------------------------------- |
| `root_path`         | (positional) Root to watch.                                           |
| `--id`              | Named index ID (same as ingest).                                      |
| `--index-dir`       | Override index directory.                                             |
| `--debounce`        | Seconds to wait after last change before running ingest (default: 2). |
| `--openai-api-key`  | OpenAI API key.                                                       |
| `--embedding-model` | Must match the index.                                                 |


**Example:** `python main.py watch /path/to/repo --id frontend --debounce 3`

**Where the watcher runs:** The watcher runs **only when you start it from the CLI** (`python main.py watch ...`). It runs in that same process and blocks until you press Ctrl+C. It does **not** run inside the FastAPI server or as a background daemon. So:

- To keep an index up to date as files change, run `watch` in a separate terminal (or under a process manager like systemd/supervisor) on the machine that has access to the repo directory.
- The watcher observes the **root_path** you pass (e.g. `/path/to/repo`) recursively; it writes updates to the index (by default `CCG_INDEX_ROOT/<id>` or `--index-dir`). Search and the API server can use that index from another process.

**No logs when you edit a file?** (1) **root_path must be the repo that contains the file** — e.g. to watch `galent-poc-agent-builder-frontend/src/app/agents/page.tsx`, run `watch` with `root_path` = the frontend repo root: `python main.py watch /path/to/galent-poc-agent-builder-frontend --id your-id`. (2) Wait at least **debounce** seconds (default 2) after saving; you should then see `File change detected: <path> (ingest in 2.0s)` and then `Running incremental ingest ...`. (3) **Only certain extensions are watched** (same as ingest): `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.hpp`. Changes to `.css`, `.json`, `.md`, `.html`, etc. do **not** trigger the watcher. (4) Paths under `.cursor`, `node_modules`, etc. are ignored. Run with `LOG_LEVEL=DEBUG` (or set logging to DEBUG) to see `Ignored (extension not watched): ...` for other file types.

### Command: `search`

Run semantic + graph search and print context (or references only with `--references-only`).


| Option              | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `query`             | (positional) Natural-language search query.                      |
| `--id`              | Named index ID (from ingest `--id`).                             |
| `--index-dir`       | Index directory if not using `--id` (default: `.ccg`).           |
| `--openai-api-key`  | For query embedding.                                             |
| `--embedding-model` | Must match index (default from `vector_meta.json`).              |
| `--top-k`           | Number of results after rerank (default: 5).                     |
| `--initial-k`       | Vector search candidates before rerank (default: 50).            |
| `--max-hops`        | 1 = one-hop graph expansion; 2+ = BFS graph search (default: 1). |
| `--max-graph-nodes` | Max nodes from graph when `max-hops` ≥ 2 (default: 50).          |
| `--references-only` | Print only file, function, and line refs (no code content).      |


**Examples:**

```bash
python main.py search "auth login" --id frontend
python main.py search "auth login" --index-dir /path/to/repo/.ccg
python main.py search "auth login" --id frontend --references-only --max-hops 2
```

### Command: `agent`

Interactive ReAct agent that can search the codebase via a tool. Requires an index created with `ingest --id`.


| Option             | Description                                 |
| ------------------ | ------------------------------------------- |
| `--id`             | (required) Named index ID.                  |
| `--model`          | OpenAI chat model (default: `gpt-4o-mini`). |
| `--openai-api-key` | OpenAI API key.                             |


**Example:** `python main.py agent --id frontend --model gpt-4o`

### Command: `serve`

Run the FastAPI server (ingest, search, search/summarize, clear, health).


| Option   | Description                     |
| -------- | ------------------------------- |
| `--host` | Bind host (default: `0.0.0.0`). |
| `--port` | Port (default: 8010).           |


**Example:** `python main.py serve --port 8010`

---

## Usage

### 1. Ingest (with a named ID for the agent)

Save an index under a **named ID** so you can use it with the agent and search by ID. Indexes are stored under `CCG_INDEX_ROOT` (default: `./ccg-indexes`). Ensure `OPENAI_API_KEY` is set (e.g. in `.env`).

```bash
# Ingest with ID (recommended for agent)
python main.py ingest /path/to/repo --id my-backend

# Optional: set where indexes live
export CCG_INDEX_ROOT=~/.ccg/indexes
python main.py ingest /path/to/repo --id my-backend
```

Without `--id`, the index is created at `<root>/.ccg` (project-local).

- **Incremental (default):** After the first run, re-running ingest only re-parses and updates **changed** files (content hashing + Merkle-style manifest in `index_dir/manifest.json`). Use `--full` to force a full re-index.

### 2. Watch (incremental on file change)

Run a watcher that triggers **incremental ingest** when files change (no full re-parse):

```bash
python main.py watch /path/to/repo --id my-backend
```

- Uses the same content hashing and manifest as incremental ingest; only changed files are re-parsed.
- `--debounce 2` (default 2 seconds) waits after the last change before running ingest.
- Stop with Ctrl+C.

### 3. Search

By **index ID** (after ingest with `--id`):

```bash
python main.py search "auth login logic" --id my-backend
```

By **index directory** (e.g. repo-local `.ccg`):

```bash
python main.py search "auth logic" --index-dir /path/to/repo/.ccg
```

**Graph search (multi-hop):** Use `--max-hops 2` (or higher) to traverse the graph from the top-k seeds instead of only one-hop expansion. `--max-graph-nodes` caps how many nodes are added (default 50).

```bash
python main.py search "auth login logic" --id my-backend --max-hops 2 --max-graph-nodes 80
```

**References only (no code content):** Use `--references-only` to get only file path, function/symbol name, and line range for each hit (no snippet text). Useful for navigation or tooling.

```bash
python main.py search "auth login logic" --id my-backend --references-only
# Output: path:line_start-line_end  name (or class_name.name)
```

The API and CLI always compute **references** (full node per hit: id, type, path, name, class_name, signature, line_start, line_end; same shape as graph nodes but without content). With `--references-only` the response returns only that list and no `context` string.

### 4. ReAct agent (interactive, with code RAG tool)

Run an agent that can **search the codebase** via a tool. You must use an **index ID** created with `ingest --id`. Uses `OPENAI_API_KEY` from `.env` or environment.

```bash
python main.py agent --id my-backend
```

Then type questions; the agent can call the search tool to pull in relevant code and answer.

- **Exit:** type `exit`, `quit`, or `q`.
- **Model:** `python main.py agent --id my-backend --model gpt-4o`

### 5. API server (FastAPI)

Run HTTP endpoints for ingest, search, and LLM-summarised search:

```bash
python main.py serve --port 8010
# Or: uvicorn ccg.server:app --host 0.0.0.0 --port 8010
```


| Endpoint            | Method | Description                                                                                                                                                                                                   |
| ------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/ingest`           | POST   | Index a codebase (body: `root_path`, optional `index_id`, `openai_api_key`, `embedding_model`). Uses `OPENAI_API_KEY` from env if not in body.                                                                |
| `/search`           | POST   | Semantic + graph search. Returns `context` (code snippets) and `references` (full node per hit: id, type, path, name, class_name, signature, line_start, line_end; no content). Set `references_only: true` in body to return only `references`. |
| `/search/summarize` | POST   | Same as search, then an LLM summarizes the context into a short answer (body: `query`, `index_id` or `index_dir`, optional `model`, `max_context_chars`)                                                      |
| `/index/clear`      | POST   | Clear a single index (body: `index_id` or `index_dir`) — deletes the index directory and all contents                                                                                                         |
| `/indexes`          | DELETE | Clear all named indexes under CCG_INDEX_ROOT                                                                                                                                                                  |
| `/indexes/clear`    | POST   | Same as DELETE /indexes (clear all)                                                                                                                                                                           |
| `/health`           | GET    | Health check                                                                                                                                                                                                  |


Example:

```bash
# Ingest (with index ID)
curl -X POST http://localhost:8010/ingest -H "Content-Type: application/json" \
  -d '{"root_path": "/path/to/repo", "index_id": "my-backend"}'

# Search (raw context + references)
curl -X POST http://localhost:8010/search -H "Content-Type: application/json" \
  -d '{"query": "auth login", "index_id": "my-backend"}'

# Search (references only: file, function, line range)
curl -X POST http://localhost:8010/search -H "Content-Type: application/json" \
  -d '{"query": "auth login", "index_id": "my-backend", "references_only": true}'

# LLM-summarised search
curl -X POST http://localhost:8010/search/summarize -H "Content-Type: application/json" \
  -d '{"query": "Where is user authentication handled?", "index_id": "my-backend", "model": "gpt-4o-mini"}'

# Clear a single index (by ID)
curl -X POST http://localhost:8010/index/clear -H "Content-Type: application/json" \
  -d '{"index_id": "my-backend"}'

# Clear a single index (by path)
curl -X POST http://localhost:8010/index/clear -H "Content-Type: application/json" \
  -d '{"index_dir": "/path/to/repo/.ccg"}'

# Clear all named indexes under CCG_INDEX_ROOT
curl -X DELETE http://localhost:8010/indexes
# Or use POST (if DELETE returns 404, e.g. server not restarted or proxy):
curl -X POST http://localhost:8010/indexes/clear
```

**Postman:** Import the collection from `postman/CCG-API.postman_collection.json`.

- **Variables:** `base_url` (default `http://localhost:8010`), `index_id` (e.g. `my-backend`). Set these in the collection or environment.
- The collection includes: Health, Ingest, Search, Search (by index_dir), **Search (references only)**, Search Summarize, Clear index, Clear all indexes.

**Search response shape:** Every `/search` response includes:

- `query` (string): The search query.
- `context` (string): Concatenated code snippets. Empty when `references_only: true`.
- `references` (array): List of full node objects for each hit (same shape as graph nodes, without content): `id`, `type`, `path`, `name`, `class_name`, `signature`, `line_start`, `line_end`. Use for navigation, tooling, or opening at line.

Example reference object:

```json
{
  "id": "abc123",
  "type": "function",
  "path": "src/utils/helper.ts",
  "name": "formatDate",
  "class_name": null,
  "signature": "formatDate(d: Date): string",
  "line_start": 10,
  "line_end": 15
}
```

---

## Search API — pipeline (how it works)

The search pipeline is **three steps**: vector search → rerank → graph expansion/search.

1. **Vector search (semantic)**
  The query is embedded with OpenAI and matched against all indexed nodes (by cosine similarity). The top `**initial_k`** candidates (default 50) are kept. This is fast approximate retrieval.
2. **Rerank (cross-encoder)**
  A cross-encoder (e.g. BGE-Reranker) scores each (query, snippet) pair for the `initial_k` candidates. The top `**top_k`** (default 5) are kept. This improves precision.
3. **Graph expansion or graph search (after reranking)**
  - `**max_hops=1` (default):** For each of the top-k nodes, add **one-hop** neighbors in the code graph (CALLS, USES_TYPE, SIBLING). These neighbors are the direct callers/callees, types, and siblings. No BFS.  
  - `**max_hops>=2`:** Run **graph search**: BFS from the top-k nodes as seeds, following CALLS/USES_TYPE/SIBLING for up to `max_hops` hops, capped by `max_graph_nodes`. This pulls in more related code (e.g. call chains).

So: **after reranking we do graph expansion (one-hop) or graph search (multi-hop)**. The final result is the union of the top-k reranked nodes plus the nodes added by the graph step. That set is returned as `context` (concatenated snippets) and `references` (full node object per node: id, type, path, name, class_name, signature, line_start, line_end; no content).

---

## How search works

CCG combines **AST-based graph indexing** and **semantic (vector) search**, then optionally expands results using the graph (G-RAG).

### Indexing (ingest)

1. **AST-based scope tree (Tree-sitter)**
  Code is parsed per file into a **scope tree** (file → class → function). Each node has:
  - **Content** (e.g. function body or class signature + docstring)
  - **Ghost text**: `path: <file> | class: <name> | <code>` so the model gets file/class context
  - **Content hash** for staleness
2. **Graph (CALLS, USES_TYPE, SIBLING)**
  A **code graph** is built from the AST:
  - **CALLS**: function A calls function B
  - **USES_TYPE**: node uses/returns a type T (link to T’s definition)
  - **SIBLING**: other functions in the same file/class  
   Stored in memory (NetworkX) and persisted (e.g. JSON/SQLite) for fast traversal.
3. **Semantic embeddings**
  Each node’s **ghost text** (or content) is embedded with **OpenAI** (default: `text-embedding-3-small`). Vectors are stored in a vector store (in-memory + on-disk, or Qdrant).
4. **Shadow index**
  Raw code, hashes, and metadata live in **SQLite**; vectors and graph are separate for fast reads and traversal.
5. **AST JSON (ingest)**
  For each file parsed with a scope tree (Python, JavaScript, TypeScript, Java), the Tree-sitter AST is written to `index_dir/ast/<path>.json` (path with `/` → `__`). Other languages (Go, Rust, C/C++) get one scope node per file and no AST file.

**Scope trees and edges by language**  

- **Python**: file, class, function; CALLS and USES_TYPE from AST.  
- **JavaScript / JSX**: file, class, function_declaration, method_definition; CALLS from call_expression.  
- **TypeScript / TSX**: same as JavaScript (class, function, method); CALLS from call_expression.  
- **Java**: file, class, interface, method_declaration, constructor_declaration; CALLS from method_invocation.  
- **Go, Rust, C/C++**: one file-level node per file (no edges).

If you see `edges_created: 0`, ensure tree-sitter parsers are available (`tree-sitter-languages` includes JavaScript, TypeScript, TSX, Java) and that the codebase has classes or functions.

### Search (query time)

1. **Semantic search (OpenAI embeddings)**
  The query is embedded with the same OpenAI model and matched against stored vectors. Top **initial_k** (e.g. 50) candidates are retrieved by cosine similarity.
2. **Optional re-ranking (cross-encoder)**
  A **cross-encoder** (e.g. BGE-Reranker) scores (query, snippet) pairs. Top **top_k** (e.g. 5) are kept to improve precision.
3. **Graph expansion or graph search (G-RAG)**
  - **One-hop (default, `max_hops=1`)**: For each top-k node, add neighbors by CALLS, USES_TYPE, and SIBLING (up to `neighbor_limit` per type).  
  - **Multi-hop graph search (`max_hops>=2`)**: BFS from the top-k seeds along CALLS/USES_TYPE/SIBLING for up to `max_hops` hops, capped by `max_graph_nodes`. Each added node is labeled with hop depth and edge type (Dependency / Type / Sibling).  
   This produces a single **context string** (code snippets + labels) for the LLM or API response.

So: **AST + graph** define *what* is indexed and *how* context is expanded; **semantic search** (and optional rerank) decide *which* nodes are relevant. The pipeline is: **semantic retrieval → optional rerank → graph expansion or graph search → context string**.

---

## Components

- **Scope-tree parser**: AST-based (Tree-sitter) chunking at file/class/function level + ghost text + content hash.
- **Graph**: CALLS, USES_TYPE, SIBLING edges for G-RAG (dependency/type/sibling hops).
- **Shadow index**: SQLite (nodes + edges), optional Qdrant or on-disk vectors.
- **Vector store**: OpenAI embeddings for semantic search (default `text-embedding-3-small`).
- **Reranker**: Optional cross-encoder (BGE-Reranker) for two-stage retrieval.
- **Collector**: Vector search → rerank → graph expansion (one-hop) or graph search (multi-hop BFS) → single LLM context string.

---

## Codebase overview


| File                      | Purpose                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**main.py`**             | CLI entry: subparsers for `ingest`, `watch`, `search`, `agent`, `serve`. Loads `.env`, maps args to config, calls runner or server.                                                                                                                                                                                                                                         |
| `**ccg/runner.py**`       | Ingest and search orchestration. `ingest_codebase()`: manifest diff, full vs incremental, parser → graph → shadow index → vector store; `search_codebase()`: loads index, builds VectorStore/Reranker/Graph/ShadowIndex, calls `get_llm_context`, returns context + references. Defines index paths (`_index_root()`, `_default_config()`), clear_index, clear_all_indexes. |
| `**ccg/vector_store.py**` | **VectorStore**: OpenAI embeddings (`embed`, `embed_single`), and either in-memory + `save_to_dir`/`load_from_dir` (vectors.npy, vector_ids.json, vector_meta.json) or Qdrant (`qdrant_url`). Methods: `upsert`, `search`, `delete_ids`, `clear`.                                                                                                                           |
| `**ccg/collector.py`**    | **get_llm_context()**: vector search → rerank → graph expansion (one-hop or multi-hop BFS) → builds context string and reference list. **References**: full node per hit (id, type, path, name, class_name, signature, line_start, line_end; no content). Uses ShadowIndex and CodeGraph. |
| `**ccg/graph.py`**        | **CodeGraph** (NetworkX): nodes (id, path, name, class_name, content, signature), edges (CALLS, USES_TYPE, SIBLING). `get_neighbors()`, `get_nodes_within_hops()` (BFS), load/save JSON.                                                                                                                                                                                    |
| `**ccg/shadow_index.py`** | **ShadowIndex** (SQLite): `nodes` table (full node fields + line_start/line_end), `edges` table. `get_node()`, `upsert_nodes()`, `delete_nodes_by_paths()`, `save_edges`/`load_edges`. Used at search time for snippet content and reference metadata.                                                                                                                      |
| `**ccg/parser.py`**       | **ScopeNode**, file discovery, Tree-sitter parsing (Python, JS/TS, Java, Go, Rust, C/C++). Builds scope tree (file/class/function), ghost text, content hash, CALLS/USES_TYPE from AST.                                                                                                                                                                                     |
| `**ccg/manifest.py`**     | Content hashing and manifest for incremental ingest: `compute_file_hashes()`, `diff_manifest()`, `load_manifest()`, `save_manifest()`, Merkle-style root hash.                                                                                                                                                                                                              |
| `**ccg/reranker.py**`     | **Reranker**: cross-encoder (FlagEmbedding BGE-Reranker) for (query, doc) scoring. Falls back to vector-search order if model fails to load.                                                                                                                                                                                                                                |
| `**ccg/watcher.py`**      | **run_watcher()**: watchdog on repo directory, debounce, then call `ingest_codebase` incrementally. Ignores VCS/build/node_modules etc. Invoked only from CLI `watch`; runs in that process (blocking). Not part of the API server.                                                                                                                                         |
| `**ccg/agent.py`**        | **run_agent()**: ReAct agent (LangGraph) with a `search_code` tool that calls `search_codebase` with fixed `index_id`. Interactive loop.                                                                                                                                                                                                                                    |
| `**ccg/server.py`**       | FastAPI app: `/ingest`, `/search`, `/search/summarize`, `/index/clear`, `/indexes` (DELETE), `/indexes/clear` (POST), `/health`. Request/response models; loads `.env`.                                                                                                                                                                                                     |


**Data flow summary:** Parser → ScopeNodes → CodeGraph + ShadowIndex + VectorStore (ingest). Search: VectorStore.search → Reranker.rerank → Collector (graph expansion) → context + references from ShadowIndex/Graph.

---

**Detailed process:** [docs/INGESTION_SEARCH_AND_WATCHER.md](docs/INGESTION_SEARCH_AND_WATCHER.md) describes ingestion (full vs incremental, Merkle-style manifest and content hashing), search (vector → rerank → graph), and the watcher (debounce → incremental ingest).

**Flowcharts:** [docs/FLOWCHARTS.md](docs/FLOWCHARTS.md) has Mermaid flowcharts for [ingestion](docs/FLOWCHARTS.md#1-ingestion-process) and [search](docs/FLOWCHARTS.md#2-search-process) separately.