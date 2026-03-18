"""
Smart collector: vector search → re-rank → graph expansion → single LLM context string.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from .graph import CodeGraph, EDGE_CALLS, EDGE_SIBLING, EDGE_USES_TYPE
from .reranker import Reranker
from .shadow_index import ShadowIndex
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


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
) -> str:
    """
    Two-stage retrieval + graph expansion (or graph search when max_hops > 1).
    Returns a single string for LLM context.
    - max_hops=1: one-hop expansion from each top-k node (current behavior).
    - max_hops>=2: graph search (BFS) from top-k seeds up to max_hops, capped by max_graph_nodes.
    """
    query_vector = vector_store.embed_single(query)
    initial = vector_store.search(query_vector, k=initial_k)
    if not initial:
        return ""

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
            if not node:
                continue
            content = node.get("content") or node.get("ghost_text", "")
            final_parts.append(content)
            seen_ids.add(nid)
        # Then add graph-search results with depth/edge labels
        for node, edge_type, depth in hopped:
            nid = node.get("id", "")
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            label = _edge_label(edge_type)
            sig = node.get("signature", "")
            content_slice = (node.get("content") or "")[:500]
            if edge_type == EDGE_USES_TYPE:
                text = content_slice or sig
            else:
                text = sig or content_slice
            final_parts.append(f"// [Hop {depth}] {label}: {node.get('name', '')}\n{text}")
        return "\n\n".join(final_parts)

    # One-hop expansion (original behavior)
    for nid, score in ranked:
        node = shadow_index.get_node(nid)
        if not node:
            continue
        content = node.get("content") or node.get("ghost_text", "")
        final_parts.append(content)
        seen_ids.add(nid)

        if not graph.has_node(nid):
            continue

        if include_calls:
            for neighbor in graph.get_neighbors(nid, EDGE_CALLS, limit=neighbor_limit):
                if neighbor["id"] not in seen_ids:
                    final_parts.append(f"// Dependency: {neighbor.get('name', '')}\n{neighbor.get('signature', '')}")
                    seen_ids.add(neighbor["id"])

        if include_types:
            for neighbor in graph.get_neighbors(nid, EDGE_USES_TYPE, limit=neighbor_limit):
                if neighbor["id"] not in seen_ids:
                    final_parts.append(f"// Type: {neighbor.get('name', '')}\n{neighbor.get('content', '')[:500]}")
                    seen_ids.add(neighbor["id"])

        if include_siblings:
            for neighbor in graph.get_neighbors(nid, EDGE_SIBLING, limit=neighbor_limit):
                if neighbor["id"] not in seen_ids:
                    final_parts.append(f"// Sibling: {neighbor.get('name', '')}\n{neighbor.get('signature', '')}")
                    seen_ids.add(neighbor["id"])

    return "\n\n".join(final_parts)
