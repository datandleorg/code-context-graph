"""
Smart collector: vector search → re-rank → graph expansion → single LLM context string and/or references (file, function, lines).
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .graph import CodeGraph, EDGE_CALLS, EDGE_SIBLING, EDGE_USES_TYPE
from .reranker import Reranker
from .shadow_index import ShadowIndex
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Reference dict keys for search results (file, function/symbol, line range)
REFERENCE_KEYS = ("path", "name", "class_name", "line_start", "line_end")


def _node_to_reference(node: Dict[str, Any]) -> Dict[str, Any]:
    """Build a reference dict (path, name, class_name, line_start, line_end) from a node."""
    return {
        "path": node.get("path", ""),
        "name": node.get("name", ""),
        "class_name": node.get("class_name"),
        "line_start": node.get("start_line") or node.get("line_start"),
        "line_end": node.get("end_line") or node.get("line_end"),
    }


def _edge_label(edge_type: str) -> str:
    if edge_type == EDGE_CALLS:
        return "Dependency"
    if edge_type == EDGE_USES_TYPE:
        return "Type"
    if edge_type == EDGE_SIBLING:
        return "Sibling"
    return edge_type


def get_llm_context(
    query: str,
    vector_store: VectorStore,
    reranker: Reranker,
    graph: CodeGraph,
    shadow_index: ShadowIndex,
    top_k: int = 5,
    initial_k: int = 50,
    include_calls: bool = True,
    include_types: bool = True,
    include_siblings: bool = True,
    neighbor_limit: int = 5,
    max_hops: int = 1,
    max_graph_nodes: int = 50,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Two-stage retrieval + graph expansion (or graph search when max_hops > 1).
    Returns (context_string, references).
    - context_string: concatenated code snippets for LLM context.
    - references: list of {"path", "name", "class_name", "line_start", "line_end"} for each relevant node (order preserved).
    - max_hops=1: one-hop expansion from each top-k node (current behavior).
    - max_hops>=2: graph search (BFS) from top-k seeds up to max_hops, capped by max_graph_nodes.
    """
    query_vector = vector_store.embed_single(query)
    initial = vector_store.search(query_vector, k=initial_k)
    if not initial:
        return "", []

    # Build (node_id, content) for reranker; content from shadow index
    id_to_content: Dict[str, str] = {}
    for nid, _score in initial:
        node = shadow_index.get_node(nid)
        if node:
            id_to_content[nid] = node.get("content") or node.get("ghost_text", "")
    candidates = [(nid, id_to_content.get(nid, "")) for nid, _ in initial]
    ranked = reranker.rerank(query, candidates, top_n=top_k)

    final_parts: List[str] = []
    seen_ids: Set[str] = set()
    ref_order: List[str] = []  # node ids in order of first appearance for references

    def _add_ref(nid: str) -> None:
        if nid not in seen_ids:
            ref_order.append(nid)

    edge_types: List[str] = []
    if include_calls:
        edge_types.append(EDGE_CALLS)
    if include_types:
        edge_types.append(EDGE_USES_TYPE)
    if include_siblings:
        edge_types.append(EDGE_SIBLING)

    if max_hops >= 2 and edge_types:
        # Graph search: multi-hop BFS from top-k seeds
        seed_ids = [nid for nid, _ in ranked]
        hopped = graph.get_nodes_within_hops(
            seed_ids,
            edge_types=edge_types,
            max_hops=max_hops,
            max_nodes=max_graph_nodes,
        )
        # First add top-k node contents in order
        for nid, _score in ranked:
            node = shadow_index.get_node(nid)
            _add_ref(nid)
            seen_ids.add(nid)
            if node:
                content = node.get("content") or node.get("ghost_text", "")
                final_parts.append(content)
        # Then add graph-search results with depth/edge labels
        for node, edge_type, depth in hopped:
            nid = node.get("id", "")
            if nid in seen_ids:
                continue
            _add_ref(nid)
            seen_ids.add(nid)
            label = _edge_label(edge_type)
            sig = node.get("signature", "")
            content_slice = (node.get("content") or "")[:500]
            if edge_type == EDGE_USES_TYPE:
                text = content_slice or sig
            else:
                text = sig or content_slice
            final_parts.append(f"// [Hop {depth}] {label}: {node.get('name', '')}\n{text}")
        context_str = "\n\n".join(final_parts)
        references = _build_references(ref_order, shadow_index, graph)
        return context_str, references

    # One-hop expansion (original behavior)
    for nid, score in ranked:
        node = shadow_index.get_node(nid)
        _add_ref(nid)
        seen_ids.add(nid)
        if node:
            content = node.get("content") or node.get("ghost_text", "")
            final_parts.append(content)

        if not graph.has_node(nid):
            continue

        if include_calls:
            for neighbor in graph.get_neighbors(nid, EDGE_CALLS, limit=neighbor_limit):
                if neighbor["id"] not in seen_ids:
                    final_parts.append(f"// Dependency: {neighbor.get('name', '')}\n{neighbor.get('signature', '')}")
                    _add_ref(neighbor["id"])
                    seen_ids.add(neighbor["id"])

        if include_types:
            for neighbor in graph.get_neighbors(nid, EDGE_USES_TYPE, limit=neighbor_limit):
                if neighbor["id"] not in seen_ids:
                    final_parts.append(f"// Type: {neighbor.get('name', '')}\n{neighbor.get('content', '')[:500]}")
                    _add_ref(neighbor["id"])
                    seen_ids.add(neighbor["id"])

        if include_siblings:
            for neighbor in graph.get_neighbors(nid, EDGE_SIBLING, limit=neighbor_limit):
                if neighbor["id"] not in seen_ids:
                    final_parts.append(f"// Sibling: {neighbor.get('name', '')}\n{neighbor.get('signature', '')}")
                    _add_ref(neighbor["id"])
                    seen_ids.add(neighbor["id"])

    context_str = "\n\n".join(final_parts)
    references = _build_references(ref_order, shadow_index, graph)
    return context_str, references


def _build_references(
    ref_order: List[str],
    shadow_index: ShadowIndex,
    graph: CodeGraph,
) -> List[Dict[str, Any]]:
    """Build list of reference dicts (path, name, class_name, line_start, line_end) from node ids. Prefer shadow_index for line numbers."""
    out: List[Dict[str, Any]] = []
    for nid in ref_order:
        node = shadow_index.get_node(nid)
        if node:
            out.append(_node_to_reference(node))
        else:
            # Graph-only node (e.g. from load); graph may not store line numbers
            gnode = graph.get_node(nid) if graph else None
            if gnode:
                out.append(_node_to_reference(gnode))
    return out
