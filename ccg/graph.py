"""
Code graph: nodes and edges (CALLS, USES_TYPE, SIBLING) for G-RAG hopping.
Build from parser output; persist/load via edge list; get_neighbors by edge type.
"""

import json
import logging
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .parser import ScopeNode

logger = logging.getLogger(__name__)

EDGE_CALLS = "CALLS"
EDGE_USES_TYPE = "USES_TYPE"
EDGE_SIBLING = "SIBLING"


class CodeGraph:
    """
    In-memory graph of scope nodes with directed edges by type.
    Node attributes: id, type, path, name, class_name, content, signature.
    """

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()
        self._nodes_by_id: Dict[str, Dict[str, Any]] = {}

    def add_node(self, node: ScopeNode) -> None:
        self._g.add_node(
            node.id,
            type=node.type,
            path=node.path,
            name=node.name,
            class_name=node.class_name,
            content=node.content,
            signature=node.signature,
        )
        self._nodes_by_id[node.id] = {
            "id": node.id,
            "type": node.type,
            "path": node.path,
            "name": node.name,
            "class_name": node.class_name,
            "content": node.content,
            "signature": node.signature,
        }

    def add_edge(self, source_id: str, target_id: str, edge_type: str) -> None:
        self._g.add_edge(source_id, target_id, key=edge_type)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return list of neighbor node dicts (id, name, signature, content) for the given edge type.
        """
        if node_id not in self._g:
            return []
        out: List[Dict[str, Any]] = []
        for _u, succ, keys in self._g.out_edges(node_id, keys=True):
            if keys != edge_type:
                continue
            if succ in self._nodes_by_id:
                out.append(self._nodes_by_id[succ].copy())
            if limit is not None and len(out) >= limit:
                break
        return out

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._nodes_by_id.get(node_id)

    def get_nodes_within_hops(
        self,
        seed_ids: List[str],
        edge_types: Optional[List[str]] = None,
        max_hops: int = 2,
        max_nodes: int = 100,
    ) -> List[Tuple[Dict[str, Any], str, int]]:
        """
        Graph search: BFS from seed nodes following given edge types.
        Returns list of (node_dict, edge_type_used_to_reach, depth).
        depth 1 = one hop from a seed, depth 2 = two hops, etc.
        Seeds are not included. Results are deduplicated by node id (first reach wins).
        """
        if not seed_ids or max_hops < 1:
            return []
        types = edge_types or [EDGE_CALLS, EDGE_USES_TYPE, EDGE_SIBLING]
        seen: set = set(seed_ids)
        out: List[Tuple[Dict[str, Any], str, int]] = []
        # (node_id, depth)
        q: deque = deque((s, 1) for s in seed_ids if self.has_node(s))
        while q and len(out) < max_nodes:
            nid, depth = q.popleft()
            if depth > max_hops:
                continue
            for _u, succ, key in self._g.out_edges(nid, keys=True):
                if key not in types or succ in seen:
                    continue
                seen.add(succ)
                node = self._nodes_by_id.get(succ)
                if node:
                    out.append((node.copy(), key, depth))
                    if len(out) >= max_nodes:
                        break
                    q.append((succ, depth + 1))
        return out

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes_by_id

    def remove_node(self, node_id: str) -> None:
        """Remove node and all edges touching it."""
        if node_id not in self._nodes_by_id:
            return
        self._g.remove_node(node_id)
        del self._nodes_by_id[node_id]

    def remove_nodes(self, node_ids: List[str]) -> None:
        """Remove multiple nodes and their edges."""
        for nid in node_ids:
            self.remove_node(nid)

    def get_by_name_path(self) -> Dict[tuple, str]:
        """Return (path, name) -> node_id for all nodes (for resolving CALLS/USES_TYPE across graph)."""
        out: Dict[tuple, str] = {}
        for nid, attrs in self._nodes_by_id.items():
            path = attrs.get("path", "")
            name = attrs.get("name", "")
            out[(path, name)] = nid
        return out

    def node_count(self) -> int:
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def build_from_nodes(self, nodes: List[ScopeNode]) -> None:
        """
        Add all nodes and derive SIBLING edges (same path; same class_name for functions).
        CALLS and USES_TYPE are added separately via add_edges_from_parser or by resolver.
        """
        by_path_class: Dict[tuple, List[str]] = {}
        by_path: Dict[str, List[str]] = {}
        for node in nodes:
            self.add_node(node)
            key_path_class = (node.path, node.class_name or "")
            by_path_class.setdefault(key_path_class, []).append(node.id)
            by_path.setdefault(node.path, []).append(node.id)
        for (path, class_name), nids in by_path_class.items():
            for i, nid in enumerate(nids):
                for j, other in enumerate(nids):
                    if i != j:
                        self.add_edge(nid, other, EDGE_SIBLING)
        logger.info("Graph nodes and SIBLING edges: %d nodes, %d edges", self.node_count(), self.edge_count())

    def add_edges_from_parser(
        self,
        nodes: List[ScopeNode],
        existing_by_name_path: Optional[Dict[tuple, str]] = None,
    ) -> None:
        """
        Resolve CALLS and USES_TYPE from parser output.
        If existing_by_name_path is provided (e.g. from get_by_name_path()), edges can target existing nodes.
        """
        by_name_path: Dict[tuple, str] = dict(existing_by_name_path or {})
        for node in nodes:
            key = (node.path, node.name)
            by_name_path[key] = node.id
        for node in nodes:
            for call_name, _ in node.calls:
                # Prefer same file
                target_id = by_name_path.get((node.path, call_name))
                if not target_id:
                    for (p, n), nid in by_name_path.items():
                        if n == call_name:
                            target_id = nid
                            break
                if target_id and target_id != node.id:
                    self.add_edge(node.id, target_id, EDGE_CALLS)
            for type_name in node.uses_types:
                tid = by_name_path.get((node.path, type_name)) or next(
                    (nid for (p, n), nid in by_name_path.items() if n == type_name),
                    None,
                )
                if tid and tid != node.id:
                    self.add_edge(node.id, tid, EDGE_USES_TYPE)
        logger.info("Graph CALLS/USES_TYPE edges added; total edges now: %d", self.edge_count())

    def to_dict(self) -> Dict[str, Any]:
        nodes = list(self._nodes_by_id.values())
        edges: List[Dict[str, str]] = []
        for u, v, key in self._g.edges(keys=True):
            edges.append({"source": u, "target": v, "edge_type": key})
        return {"nodes": nodes, "edges": edges}

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=0)

    @classmethod
    def load(cls, path: str | Path) -> "CodeGraph":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        g = cls()
        for n in data["nodes"]:
            g._nodes_by_id[n["id"]] = n
            g._g.add_node(
                n["id"],
                type=n.get("type"),
                path=n.get("path"),
                name=n.get("name"),
                class_name=n.get("class_name"),
                content=n.get("content"),
                signature=n.get("signature", ""),
            )
        for e in data["edges"]:
            g._g.add_edge(e["source"], e["target"], key=e["edge_type"])
        return g
