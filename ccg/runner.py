"""
CCG ingest and search entry points (used by main CLI and by the agent).
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

INDEX_DIR_NAME = ".ccg"
AST_DIR_NAME = "ast"


def _ast_safe_path(rel_path: str) -> str:
    """Turn path like 'src/foo.py' into a safe filename for AST JSON (e.g. src__foo.py.json)."""
    safe = rel_path.replace("/", "__").replace("\\", "__")
    for c in ('..', ':', '*', '?'):
        safe = safe.replace(c, "_")
    return f"{safe}.json"


def _write_ast_files(index_dir: Path, ast_by_path: Dict[str, Any]) -> None:
    """Write AST JSON files to index_dir/ast/ (one file per path)."""
    if not ast_by_path:
        return
    ast_dir = index_dir / AST_DIR_NAME
    ast_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, ast_dict in ast_by_path.items():
        try:
            safe = _ast_safe_path(rel_path)
            out_path = ast_dir / safe
            out_path.write_text(json.dumps(ast_dict, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to write AST for %s: %s", rel_path, e)
    logger.info("Wrote %d AST file(s) to %s", len(ast_by_path), ast_dir)


def _index_root() -> Path:
    return Path(os.environ.get("CCG_INDEX_ROOT", Path.cwd() / "ccg-indexes")).resolve()


def clear_index(
    index_id: Optional[str] = None,
    index_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Remove an index (delete its directory and all contents: ccg.db, graph.json, vectors/, manifest.json, ast/).
    Provide index_id (under CCG_INDEX_ROOT) or index_dir (absolute path). Returns {"cleared": True, "index_dir": str} or error.
    """
    if index_id and index_dir:
        return {"error": "Provide only one of index_id or index_dir"}
    if not index_id and not index_dir:
        return {"error": "Provide index_id or index_dir"}
    if index_id:
        path = _index_root() / index_id
    else:
        path = Path(index_dir).resolve()
    if not path.is_dir():
        return {"error": f"Index directory not found: {path}"}
    try:
        shutil.rmtree(path)
        logger.info("Cleared index at %s", path)
        return {"cleared": True, "index_dir": str(path), "index_id": index_id}
    except Exception as e:
        logger.exception("Failed to clear index %s: %s", path, e)
        return {"error": str(e)}


def clear_all_indexes() -> Dict[str, Any]:
    """
    Remove all named indexes under CCG_INDEX_ROOT (each subdirectory is treated as an index).
    Returns {"cleared": True, "cleared_ids": [str]} or error.
    """
    root = _index_root()
    if not root.is_dir():
        return {"cleared": True, "cleared_ids": []}
    cleared: List[str] = []
    try:
        for child in list(root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                shutil.rmtree(child)
                cleared.append(child.name)
                logger.info("Cleared index: %s", child.name)
        return {"cleared": True, "cleared_ids": cleared}
    except Exception as e:
        logger.exception("Failed to clear all indexes: %s", e)
        return {"error": str(e), "cleared_ids": cleared}


def _default_config(root_path: str | Path, index_id: Optional[str] = None) -> Dict[str, Any]:
    root = Path(root_path).resolve()
    if index_id:
        index_dir = _index_root() / index_id
    else:
        index_dir = root / INDEX_DIR_NAME
    return {
        "index_dir": str(index_dir),
        "index_id": index_id,
        "sqlite_path": str(index_dir / "ccg.db"),
        "graph_path": str(index_dir / "graph.json"),
        "vectors_dir": str(index_dir / "vectors"),
        "qdrant_url": None,
        "extensions": None,
        "embedding_model": None,
        "reranker_model": None,
    }


def _run_full_ingest(
    root: Path,
    index_dir: Path,
    sqlite_path: Path,
    graph_path: Path,
    vectors_dir: Path,
    extensions: Optional[Set[str]],
    embedding_model: str,
    qdrant_url: Optional[str],
    openai_api_key: Optional[str],
    embedding_provider: Optional[str],
    index_id: Optional[str],
) -> Dict[str, Any]:
    from ccg.parser import parse_codebase as parse_cb, discover_files
    from ccg.graph import CodeGraph
    from ccg.manifest import save_manifest, compute_file_hashes
    from ccg.shadow_index import ShadowIndex
    from ccg.vector_store import VectorStore

    logger.info("Discovering files (extensions=%s) ...", extensions or "default")
    files = discover_files(root, extensions)
    if not files:
        return {"files_found": 0, "nodes_created": 0, "edges_created": 0}
    logger.info("Parsing %d files ...", len(files))
    ast_by_path: Dict[str, Any] = {}
    nodes = parse_cb(root, extensions, collect_ast=ast_by_path)
    if not nodes:
        return {"files_found": len(files), "nodes_created": 0, "edges_created": 0}
    _write_ast_files(index_dir, ast_by_path)
    logger.info("Building graph (%d nodes) ...", len(nodes))
    graph = CodeGraph()
    graph.build_from_nodes(nodes)
    graph.add_edges_from_parser(nodes)
    if graph.edge_count() == 0:
        logger.warning(
            "No graph edges created. Edges (CALLS, USES_TYPE, SIBLING) are built when scope-tree parsing succeeds "
            "(Python, JavaScript, TypeScript, Java). If tree-sitter is missing or files have no classes/functions, edges stay empty. See README."
        )

    with ShadowIndex(sqlite_path) as idx:
        idx.upsert_nodes(nodes)
        edges = [{"source": e["source"], "target": e["target"], "edge_type": e["edge_type"]} for e in graph.to_dict()["edges"]]
        idx.save_edges(edges)
    graph.save(graph_path)

    logger.info("Embedding %d nodes ...", len(nodes))
    vector_store = VectorStore(
        embedding_model=embedding_model,
        qdrant_url=qdrant_url,
        openai_api_key=openai_api_key,
        embedding_provider=embedding_provider or ("openai" if openai_api_key else None),
    )
    vector_store.upsert([n.id for n in nodes], [n.ghost_text for n in nodes])
    if not qdrant_url:
        vector_store.save_to_dir(vectors_dir)

    current_hashes = compute_file_hashes(root, files)
    save_manifest(index_dir, current_hashes)
    logger.info("Full ingest complete: files=%d, nodes=%d, edges=%d", len(files), len(nodes), graph.edge_count())
    out = {"files_found": len(files), "nodes_created": len(nodes), "edges_created": graph.edge_count()}
    if index_id:
        out["index_id"] = index_id
        out["index_dir"] = str(index_dir)
    return out


def ingest_codebase(
    root_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    incremental: bool = True,
) -> Dict[str, Any]:
    """
    Ingest (index) a codebase. When incremental=True and a manifest exists, only re-parse
    and update changed/new/deleted files (content hashing + Merkle-style root). Otherwise full ingest.
    """
    from ccg.parser import discover_files, parse_files
    from ccg.graph import CodeGraph
    from ccg.manifest import load_manifest, save_manifest, compute_file_hashes, diff_manifest
    from ccg.shadow_index import ShadowIndex
    from ccg.vector_store import VectorStore

    logger.info("Ingest started for root_path=%s (incremental=%s)", root_path, incremental)
    root = Path(root_path).resolve()
    if not root.is_dir():
        logger.error("Not a directory: %s", root)
        return {"error": f"Not a directory: {root}"}

    index_id = (config or {}).get("index_id")
    cfg = _default_config(root, index_id=index_id)
    if config:
        cfg.update(config)
    index_dir = Path(cfg["index_dir"])
    sqlite_path = Path(cfg["sqlite_path"])
    graph_path = Path(cfg["graph_path"])
    vectors_dir = Path(cfg["vectors_dir"])
    extensions: Optional[Set[str]] = cfg.get("extensions")
    qdrant_url = cfg.get("qdrant_url")
    openai_api_key = cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    embedding_provider = cfg.get("embedding_provider")
    if openai_api_key and embedding_provider != "sentence-transformers":
        embedding_model = cfg.get("embedding_model") or "text-embedding-3-small"
    else:
        embedding_model = cfg.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"

    index_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(root, extensions)
    if not files:
        logger.warning("No files found; nothing to ingest")
        return {"files_found": 0, "nodes_created": 0, "edges_created": 0}

    # Incremental: diff against manifest and only update changed/new/deleted
    previous_hashes, _ = load_manifest(index_dir)
    if incremental and previous_hashes is not None and Path(graph_path).exists():
        current_hashes = compute_file_hashes(root, files)
        new_paths, changed_paths, deleted_paths = diff_manifest(previous_hashes, current_hashes)
        if not new_paths and not changed_paths and not deleted_paths:
            logger.info("No file changes; index unchanged")
            return {
                "incremental": True,
                "files_unchanged": True,
                "files_found": len(files),
                "nodes_created": 0,
                "edges_created": 0,
            }

        paths_to_remove = list(deleted_paths) + list(changed_paths)
        paths_to_parse = [root / p for p in new_paths + changed_paths]
        logger.info("Incremental: %d new, %d changed, %d deleted", len(new_paths), len(changed_paths), len(deleted_paths))

        shadow_index = ShadowIndex(sqlite_path)
        shadow_index.connect()
        removed_ids = shadow_index.delete_nodes_by_paths(paths_to_remove)
        shadow_index.close()

        graph = CodeGraph.load(graph_path)
        graph.remove_nodes(removed_ids)

        vector_store = VectorStore(
            embedding_model=embedding_model,
            qdrant_url=qdrant_url,
            openai_api_key=openai_api_key,
            embedding_provider=embedding_provider or ("openai" if openai_api_key else None),
        )
        if not qdrant_url and (Path(vectors_dir) / "vectors.npy").exists():
            vector_store.load_from_dir(vectors_dir)
        if removed_ids:
            vector_store.delete_ids(removed_ids)

        ast_by_path: Dict[str, Any] = {}
        new_nodes = parse_files(root, paths_to_parse, extensions, collect_ast=ast_by_path)
        _write_ast_files(index_dir, ast_by_path)
        if not new_nodes:
            graph.save(graph_path)
            with ShadowIndex(sqlite_path) as idx:
                all_edges = [{"source": e["source"], "target": e["target"], "edge_type": e["edge_type"]} for e in graph.to_dict()["edges"]]
                idx.save_edges(all_edges)
            save_manifest(index_dir, current_hashes)
            return {"incremental": True, "files_updated": len(paths_to_parse), "nodes_created": 0, "edges_created": graph.edge_count()}

        for node in new_nodes:
            graph.add_node(node)
        by_path_class: Dict[tuple, List[str]] = {}
        for node in new_nodes:
            key = (node.path, node.class_name or "")
            by_path_class.setdefault(key, []).append(node.id)
        for (path, class_name), nids in by_path_class.items():
            for i, nid in enumerate(nids):
                for j, other in enumerate(nids):
                    if i != j:
                        graph.add_edge(nid, other, "SIBLING")
        existing_map = graph.get_by_name_path()
        graph.add_edges_from_parser(new_nodes, existing_by_name_path=existing_map)

        with ShadowIndex(sqlite_path) as idx:
            idx.upsert_nodes(new_nodes)
            all_edges = [{"source": e["source"], "target": e["target"], "edge_type": e["edge_type"]} for e in graph.to_dict()["edges"]]
            idx.save_edges(all_edges)

        graph.save(graph_path)

        vector_store.upsert([n.id for n in new_nodes], [n.ghost_text for n in new_nodes])
        if not qdrant_url:
            vector_store.save_to_dir(vectors_dir)

        save_manifest(index_dir, current_hashes)

        logger.info("Incremental ingest complete: +%d nodes, %d edges", len(new_nodes), graph.edge_count())
        out = {
            "incremental": True,
            "files_updated": len(paths_to_parse),
            "files_deleted": len(deleted_paths),
            "nodes_created": len(new_nodes),
            "edges_created": graph.edge_count(),
        }
        if index_id:
            out["index_id"] = index_id
            out["index_dir"] = str(index_dir)
        return out

    # Full ingest
    logger.info("Full ingest (no manifest or incremental=False)")
    return _run_full_ingest(
        root=root,
        index_dir=index_dir,
        sqlite_path=sqlite_path,
        graph_path=graph_path,
        vectors_dir=vectors_dir,
        extensions=extensions,
        embedding_model=embedding_model,
        qdrant_url=qdrant_url,
        openai_api_key=openai_api_key,
        embedding_provider=embedding_provider or ("openai" if openai_api_key else None),
        index_id=index_id,
    )


def search_codebase(
    query: str,
    top_k: int = 5,
    initial_k: int = 50,
    max_hops: int = 1,
    max_graph_nodes: int = 50,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from ccg.collector import get_llm_context
    from ccg.graph import CodeGraph
    from ccg.reranker import Reranker
    from ccg.shadow_index import ShadowIndex
    from ccg.vector_store import VectorStore

    cfg = config or {}
    index_id = cfg.get("index_id")
    index_dir = cfg.get("index_dir")
    if not index_dir and not index_id:
        return {"error": "config must include index_dir or index_id"}
    if index_id:
        index_dir = _index_root() / index_id
    else:
        index_dir = Path(index_dir)
    sqlite_path = cfg.get("sqlite_path") or index_dir / "ccg.db"
    graph_path = cfg.get("graph_path") or index_dir / "graph.json"
    vectors_dir = cfg.get("vectors_dir") or index_dir / "vectors"
    qdrant_url = cfg.get("qdrant_url")
    openai_api_key = cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    embedding_provider = cfg.get("embedding_provider")
    vector_meta_path = Path(vectors_dir) / "vector_meta.json"
    if vector_meta_path.exists():
        meta = json.loads(vector_meta_path.read_text(encoding="utf-8"))
        embedding_model = cfg.get("embedding_model") or meta.get("model") or "text-embedding-3-small"
    else:
        embedding_model = cfg.get("embedding_model") or ("text-embedding-3-small" if openai_api_key else "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model = cfg.get("reranker_model") or "BAAI/bge-reranker-base"

    if not Path(sqlite_path).exists():
        return {"error": f"Index not found at {sqlite_path}. Run ingest first."}

    shadow_index = ShadowIndex(sqlite_path)
    shadow_index.connect()
    graph = CodeGraph.load(graph_path)
    vector_store = VectorStore(
        embedding_model=embedding_model,
        qdrant_url=qdrant_url,
        openai_api_key=openai_api_key,
        embedding_provider=embedding_provider or ("openai" if openai_api_key else None),
    )
    if not qdrant_url and (Path(vectors_dir) / "vectors.npy").exists():
        vector_store.load_from_dir(vectors_dir)
    reranker = Reranker(model_name=reranker_model)

    # Graph search: max_hops from config overrides positional default
    hops = cfg.get("max_hops", max_hops)
    graph_nodes = cfg.get("max_graph_nodes", max_graph_nodes)

    context = get_llm_context(
        query,
        vector_store=vector_store,
        reranker=reranker,
        graph=graph,
        shadow_index=shadow_index,
        top_k=top_k,
        initial_k=initial_k,
        max_hops=hops,
        max_graph_nodes=graph_nodes,
    )
    shadow_index.close()

    return {"context": context, "query": query}
