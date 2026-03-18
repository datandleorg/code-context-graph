# CCG: Ingestion and Search Flowcharts

## 1. Ingestion process

```mermaid
flowchart TB
    subgraph Start[" "]
        A[Ingest request: root_path, index_id / index_dir]
    end

    A --> B{Manifest exists &<br/>incremental=True?}
    B -->|No| FULL[Full ingest]
    B -->|Yes| INC[Incremental path]

    subgraph FullIngest["Full ingest"]
        FULL --> D1[Discover files<br/>extensions, ignore dirs]
        D1 --> D2[Parse all files<br/>scope tree: file/class/function]
        D2 --> D3[Write AST JSON<br/>index_dir/ast/]
        D3 --> D4[Build graph: nodes +<br/>SIBLING, CALLS, USES_TYPE]
        D4 --> D5[Shadow index: upsert nodes,<br/>save edges]
        D5 --> D6[Save graph.json]
        D6 --> D7[Embed ghost_text<br/>OpenAI or sentence-transformers]
        D7 --> D8[Vector store: upsert<br/>save to dir if in-memory]
        D8 --> D9[Compute file hashes,<br/>save manifest.json]
        D9 --> OUT1[Return: files_found,<br/>nodes_created, edges_created]
    end

    subgraph Incremental["Incremental ingest"]
        INC --> I1[Discover files]
        I1 --> I2[Load manifest,<br/>compute current hashes]
        I2 --> I3[diff_manifest: new,<br/>changed, deleted paths]
        I3 --> I4{Any changes?}
        I4 -->|No| OUT2[Return: files_unchanged]
        I4 -->|Yes| I5[Shadow: delete nodes<br/>by path for deleted+changed]
        I5 --> I6[Load graph, remove<br/>those node IDs]
        I6 --> I7[Vector store: load from dir,<br/>delete_ids]
        I7 --> I8[Parse only new+changed files]
        I8 --> I9[Write AST for parsed files]
        I9 --> I10[Add new nodes to graph,<br/>SIBLING + add_edges_from_parser]
        I10 --> I11[Shadow: upsert new nodes,<br/>save full edge list]
        I11 --> I12[Save graph, upsert vectors,<br/>save manifest]
        I12 --> OUT3[Return: incremental,<br/>nodes_created, edges_created]
    end
```

**Artifacts written**

| Step        | Output |
|------------|--------|
| Parse      | Scope nodes (id, path, type, name, content, ghost_text, calls, uses_types) |
| AST        | `index_dir/ast/<path>.json` (Python, JS, TS, Java) |
| Graph      | `index_dir/graph.json` (nodes + edges) |
| Shadow     | `index_dir/ccg.db` (nodes + edges tables) |
| Vectors    | `index_dir/vectors/` (vectors.npy, vector_ids.json, vector_meta.json) or Qdrant |
| Manifest   | `index_dir/manifest.json` (path → content_hash, root_hash) |

---

## 2. Search process

```mermaid
flowchart TB
    subgraph Start[" "]
        Q[Search request: query, index_id or index_dir, top_k, initial_k, max_hops]
    end

    Q --> L[Load index: shadow_index, graph,<br/>vector_store, reranker]

    L --> E[Embed query<br/>same bi-encoder as index]
    E --> V[Vector search: top initial_k<br/>by cosine similarity]
    V --> R[Cross-encoder rerank<br/>query × snippet → top_k]
    R --> G{max_hops >= 2?}

    G -->|Yes| GS[Graph search: BFS from top_k seeds<br/>along CALLS, USES_TYPE, SIBLING<br/>up to max_hops, cap max_graph_nodes]
    G -->|No| GH[One-hop: for each top_k node,<br/>get neighbors by CALLS, USES_TYPE, SIBLING<br/>neighbor_limit per type]

    GS --> C[Build context string:<br/>top_k contents + hopped/neighbor snippets<br/>with labels Dependency / Type / Sibling]
    GH --> C

    C --> OUT[Return context string<br/>for LLM or API response]
```

**Search data flow**

| Stage           | Input                    | Output |
|-----------------|--------------------------|--------|
| Embed           | Query text               | Query vector |
| Vector search   | Query vector, k=initial_k | List of (node_id, score) |
| Rerank          | Query, (node_id, content) pairs | Top top_k (node_id, score) |
| Graph expansion | top_k node IDs, graph    | Neighbor nodes (1-hop or BFS) |
| Context build   | top_k contents + neighbors | Single string (code + // labels) |

---

## 3. Quick reference

- **Ingestion:** Discover → Parse (scope tree) → Graph (nodes + edges) → Shadow index + graph.json → Embed → Vectors + manifest. Incremental uses manifest diff and only re-parses changed files.
- **Search:** Embed query → Vector search (initial_k) → Rerank (top_k) → Graph expansion (1-hop or multi-hop BFS) → Concatenate context string.
