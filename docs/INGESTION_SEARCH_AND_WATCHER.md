# CCG: Ingestion, Search, and Watcher — Detailed Process

This document describes how the Code Context Graph (CCG) **ingests** a codebase, how **search** works (semantic + graph), and how the **watcher** keeps the index up to date using content hashing and a Merkle-style manifest.

---

## 1. Ingestion

Ingestion builds the index: scope nodes (file/class/function), the code graph (CALLS, USES_TYPE, SIBLING), embeddings, and a file manifest for incremental updates.

### 1.1 Full vs incremental

- **Full ingest**  
  - Runs when: first run (no manifest), or you pass `--full` (CLI) / `incremental=False` (API).  
  - Steps: discover all files → parse all → build graph → write shadow index + graph + vectors → write manifest.  
  - Manifest: for each discovered file we store `path → content_hash` and a **root hash** (Merkle-style: hash of sorted `path:hash` pairs).

- **Incremental ingest**  
  - Runs when: `incremental=True` (default) and a manifest + graph already exist.  
  - Steps: discover files → compute current content hashes → **diff** against manifest (new / changed / deleted) → only re-parse and update those files; then merge into existing graph, shadow index, and vectors.

So: we **do not** re-parse the whole repo on every ingest when using the default incremental path; we use **content hashing and a Merkle-style manifest** to update only what changed.

### 1.2 Content hashing and manifest (Merkle-style)

- **Per-file hash**  
  - For each file we compute `SHA-256(normalized_content)` (normalized = strip, `\r\n` → `\n`).  
  - Stored in **manifest**: `index_dir/manifest.json` with shape `{ "root_hash": "...", "files": { "path": "hash", ... } }`.

- **Root hash**  
  - `root_hash = SHA-256(concat(sort(path + "\0" + hash for path, hash in files)))`.  
  - Any file add/change/delete changes the root hash. Used to know “something changed”; the actual diff uses the `files` map.

- **Diff**  
  - **New**: paths in current discovery not in previous manifest.  
  - **Changed**: path in both but `current_hash != previous_hash`.  
  - **Deleted**: paths in previous manifest not in current discovery.  
  - Only **new** and **changed** files are re-parsed; **deleted** (and **changed**) paths have their nodes removed from the index before re-adding nodes for changed files.

### 1.3 Ingestion pipeline (per run)

1. **Discovery**  
   - Recursive walk from repo root; include only configured extensions (e.g. `.py`, `.js`, `.ts`, …); skip ignored dirs (`node_modules`, `.git`, `__pycache__`, etc.) and ignored file patterns (e.g. `.min.js`).

2. **Hashing (incremental only)**  
   - Compute current `path → content_hash` for all discovered files.  
   - Load previous manifest; diff → `new_paths`, `changed_paths`, `deleted_paths`.  
   - If nothing changed, exit (index unchanged).

3. **Removal (incremental only)**  
   - Paths to clear: `deleted_paths ∪ changed_paths`.  
   - **Shadow index**: delete all nodes with `path` in that set (and edges touching them).  
   - **Graph**: load from disk, remove those node IDs, save.  
   - **Vectors**: load from dir (or Qdrant), `delete_ids(removed_ids)`, then persist if in-memory.

4. **Parse**  
   - **Full**: parse all discovered files.  
   - **Incremental**: parse only files under `new_paths` and `changed_paths` (via `parse_files(root, paths_to_parse, extensions)`).

5. **Scope tree and graph**  
   - Parser produces **scope nodes** (file / class / function) with: `id`, `path`, `name`, `content`, `ghost_text`, `signature`, `content_hash`, `calls`, `uses_types`.  
   - **Graph**:  
     - **Full**: build new graph from all nodes; add SIBLING (same file/class); add CALLS and USES_TYPE from parser.  
     - **Incremental**: add only new nodes to existing graph; add SIBLING among new nodes; add CALLS/USES_TYPE for new nodes using `existing_by_name_path` so edges can point to existing nodes.

6. **Shadow index**  
   - **Full**: upsert all nodes; replace all edges with the new edge list.  
   - **Incremental**: upsert only new nodes; replace edges with the **full** edge list (graph already merged).

7. **Vectors**  
   - **Full**: embed all nodes’ `ghost_text`; upsert all.  
   - **Incremental**: embed only new nodes; upsert (merge: in-memory replaces by id, Qdrant overwrites by point id).

8. **Manifest**  
   - Write updated `manifest.json`: all current `path → content_hash` and new `root_hash`.

9. **AST JSON (optional)**  
   - For each file that produced an AST (Python, JavaScript, TypeScript, Java), write `index_dir/ast/<safe_path>.json` with a serialized Tree-sitter tree. Other languages (Go, Rust, C/C++) get one scope node per file and no AST file.

Result: index (graph, shadow index, vectors, manifest, ast/) reflects the current codebase; later ingests only touch changed files when incremental is used.

**Scope trees and edges by language**  
- **Python**: file, class, function; CALLS, USES_TYPE, SIBLING.  
- **JavaScript / JSX / TypeScript / TSX**: file, class, function_declaration, method_definition; CALLS (from call_expression), SIBLING.  
- **Java**: file, class, interface, method_declaration, constructor_declaration; CALLS (from method_invocation), SIBLING.  
- **Go, Rust, C/C++**: one file-level node per file (no edges).  

If `edges_created: 0`, ensure `tree-sitter-languages` is installed and the repo contains classes or functions in a supported language.

---

## 2. Search

Search returns a single **context string** (code snippets + labels) for the query, by combining **semantic retrieval**, optional **reranking**, and **graph expansion or graph search**.

### 2.1 High-level flow

1. **Embed query** with the same bi-encoder used at ingest.  
2. **Vector search**: top `initial_k` (e.g. 50) nodes by similarity.  
3. **Rerank** (optional): cross-encoder scores (query, snippet); keep top `top_k` (e.g. 5).  
4. **Graph expansion or graph search**: from those top-k nodes, add related code using the graph; output one concatenated context string.

### 2.2 One-hop expansion (default, `max_hops=1`)

- For each of the top-k nodes:  
  - Add **CALLS** neighbors (signatures) — “Dependency”.  
  - Add **USES_TYPE** neighbors (type definitions) — “Type”.  
  - Add **SIBLING** neighbors (same file/class) — “Sibling”.  
- Each neighbor is added once (dedup by node id), with a small per-type limit (`neighbor_limit`).

### 2.3 Graph search (multi-hop, `max_hops ≥ 2`)

- **Seeds**: the top-k nodes from vector + rerank.  
- **BFS** from seeds along CALLS, USES_TYPE, SIBLING for up to `max_hops` hops.  
- Results are capped by `max_graph_nodes`; each added node is labeled with hop depth and edge type (e.g. `[Hop 2] Dependency`).  
- No re-parsing; this is pure **traversal** over the existing graph.

### 2.4 Output

- **Context**: A single string — top-k node contents first, then expanded/graph-search nodes with comments like `// Dependency: ...`, `// [Hop 2] Type: ...`. This is what the API returns as `context` (or what the agent/LLM receives).
- **References**: A list of full node objects for each hit (same shape as graph nodes, without content): `id`, `type`, `path`, `name`, `class_name`, `signature`, `line_start`, `line_end`. The API always returns this array alongside `context`; use `references_only: true` to skip the context string and get only references.

---

## 3. Watcher

The watcher keeps the index up to date when files change, by running **incremental ingest** after a short quiet period (debounce).

### 3.1 Role

- **No re-parsing of the whole repo**: only files that actually changed (by content hash) are re-parsed.  
- **Uses the same manifest and content hashing** as incremental ingest: Merkle-style root and per-file hashes determine what is new/changed/deleted.

### 3.2 Process

1. **Watch** the repo root recursively (e.g. via `watchdog`).  
2. **Filter events**: ignore non-source paths (same idea as discovery: ignore `node_modules`, `.git`, wrong extensions, etc.).  
3. **Debounce**: on any relevant create/modify/delete, set a “pending ingest” flag and record the time.  
4. **Timer loop**: when “pending” and at least `debounce_seconds` (default 2) have passed since the last event, run **one** incremental ingest:  
   - `ingest_codebase(root_path, config=config, incremental=True)`.  
5. **Incremental ingest** (see §1): load manifest, hash all discovered files, diff (new/changed/deleted), remove nodes for deleted/changed paths, parse only new+changed, merge into graph/shadow/vectors, save manifest.  
6. Repeat until the process is stopped (e.g. Ctrl+C).

### 3.3 Usage

- **CLI**: `python main.py watch <root_path> [--id ID] [--debounce 2]`.  
- **Config**: same as ingest (`--id`, `--index-dir`, `--openai-api-key`, `--embedding-model`).  
- **First run**: if there is no index yet, the first trigger runs ingest; with no manifest it behaves as a **full** ingest, then subsequent runs are incremental.

### 3.4 Summary

- **Watcher** = file system events + debounce → trigger **incremental ingest**.  
- **Incremental ingest** = content hashing + Merkle-style manifest → only **update changed files** in the graph and index.  
- So: we **do** use a Merkle tree–style approach (root hash + per-file hashes) and **do not** re-parse all files on every change when the watcher runs.

---

## 4. Summary table

| Component      | What it does |
|----------------|--------------|
| **Manifest**   | Stores `path → content_hash` and a Merkle-style `root_hash`; used to diff and decide what to re-parse. |
| **Full ingest**| Discover all → parse all → build graph + shadow + vectors → write manifest. |
| **Incremental ingest** | Discover all → hash all → diff vs manifest → remove nodes for deleted/changed paths → parse only new+changed → merge into graph/shadow/vectors → save manifest. |
| **Search**     | Vector search → optional rerank → one-hop expansion or multi-hop graph search → single context string. |
| **Watcher**    | Watch repo → debounce → run incremental ingest so the graph updates when files change. |

This is the detailed process of ingestion (with Merkle-style content hashing and incremental updates), search (semantic + graph), and the watcher.
