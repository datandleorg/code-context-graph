# Code Context Graph (CCG)

Production-grade code indexing and search with two-stage retrieval (OpenAI embeddings + cross-encoder re-ranking) and graph-augmented RAG.

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

**`.env.example`:**

```env
# Required for embeddings (ingest, search, agent, serve)
OPENAI_API_KEY=sk-your-openai-api-key

# Optional: where named indexes are stored (default: ./ccg-indexes)
# CCG_INDEX_ROOT=./ccg-indexes
```

- **CLI** (`main.py ingest|search|agent|watch|serve`) and **FastAPI server** both load `.env` from the project root via `python-dotenv`.
- You can still pass `--openai-api-key` (CLI) or `openai_api_key` in request bodies to override.
- **Embedding model:** default is `text-embedding-3-small`; override with `--embedding-model` (CLI) or `embedding_model` in the ingest/search config.

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

The API and CLI always compute **references** (path, name, class_name, line_start, line_end); with `--references-only` the response returns only that list and no `context` string.

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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Index a codebase (body: `root_path`, optional `index_id`, `openai_api_key`, `embedding_model`). Uses `OPENAI_API_KEY` from env if not in body. |
| `/search` | POST | Semantic + graph search. Returns `context` (code snippets) and `references` (path, name, class_name, line_start, line_end). Set `references_only: true` in body to skip content and return only `references`. |
| `/search/summarize` | POST | Same as search, then an LLM summarizes the context into a short answer (body: `query`, `index_id` or `index_dir`, optional `model`, `max_context_chars`) |
| `/index/clear` | POST | Clear a single index (body: `index_id` or `index_dir`) — deletes the index directory and all contents |
| `/indexes` | DELETE | Clear all named indexes under CCG_INDEX_ROOT |
| `/indexes/clear` | POST | Same as DELETE /indexes (clear all) |
| `/health` | GET | Health check |

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
- `references` (array): List of `{ path, name, class_name?, line_start, line_end }` for each hit (file, function/symbol, line range). Use for navigation or tooling without parsing the full context.

---

## Search API — pipeline (how it works)

The search pipeline is **three steps**: vector search → rerank → graph expansion/search.

1. **Vector search (semantic)**  
   The query is embedded with OpenAI and matched against all indexed nodes (by cosine similarity). The top **`initial_k`** candidates (default 50) are kept. This is fast approximate retrieval.

2. **Rerank (cross-encoder)**  
   A cross-encoder (e.g. BGE-Reranker) scores each (query, snippet) pair for the `initial_k` candidates. The top **`top_k`** (default 5) are kept. This improves precision.

3. **Graph expansion or graph search (after reranking)**  
   - **`max_hops=1` (default):** For each of the top-k nodes, add **one-hop** neighbors in the code graph (CALLS, USES_TYPE, SIBLING). These neighbors are the direct callers/callees, types, and siblings. No BFS.  
   - **`max_hops>=2`:** Run **graph search**: BFS from the top-k nodes as seeds, following CALLS/USES_TYPE/SIBLING for up to `max_hops` hops, capped by `max_graph_nodes`. This pulls in more related code (e.g. call chains).

So: **after reranking we do graph expansion (one-hop) or graph search (multi-hop)**. The final result is the union of the top-k reranked nodes plus the nodes added by the graph step. That set is returned as `context` (concatenated snippets) and `references` (path, name, line range for each node).

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

**Detailed process:** [docs/INGESTION_SEARCH_AND_WATCHER.md](docs/INGESTION_SEARCH_AND_WATCHER.md) describes ingestion (full vs incremental, Merkle-style manifest and content hashing), search (vector → rerank → graph), and the watcher (debounce → incremental ingest).

**Flowcharts:** [docs/FLOWCHARTS.md](docs/FLOWCHARTS.md) has Mermaid flowcharts for [ingestion](docs/FLOWCHARTS.md#1-ingestion-process) and [search](docs/FLOWCHARTS.md#2-search-process) separately.
