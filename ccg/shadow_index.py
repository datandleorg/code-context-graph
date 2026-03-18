"""
Shadow index: SQLite for nodes (code + hashes), vector store reference, and graph persistence.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parser import ScopeNode

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    class_name TEXT,
    content TEXT NOT NULL,
    ghost_text TEXT NOT NULL,
    signature TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, edge_type),
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_path ON nodes(path);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id, edge_type);
"""


class ShadowIndex:
    """
    SQLite-backed store for nodes and edges; coordinates with vector store and graph.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ShadowIndex":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def upsert_node(self, node: ScopeNode) -> None:
        self.connect()
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO nodes (id, path, type, name, class_name, content, ghost_text, signature, content_hash, line_start, line_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.path,
                node.type,
                node.name,
                node.class_name,
                node.content,
                node.ghost_text,
                node.signature,
                node.content_hash,
                node.start_line,
                node.end_line,
            ),
        )
        self._conn.commit()

    def upsert_nodes(self, nodes: List[ScopeNode]) -> None:
        self.connect()
        assert self._conn is not None
        logger.info("ShadowIndex: upserting %d nodes to %s", len(nodes), self._db_path)
        for node in nodes:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO nodes (id, path, type, name, class_name, content, ghost_text, signature, content_hash, line_start, line_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.path,
                    node.type,
                    node.name,
                    node.class_name,
                    node.content,
                    node.ghost_text,
                    node.signature,
                    node.content_hash,
                    node.start_line,
                    node.end_line,
                ),
            )
        self._conn.commit()
        logger.info("ShadowIndex: upserted %d nodes", len(nodes))

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        self.connect()
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT id, path, type, name, class_name, content, ghost_text, signature, content_hash, line_start, line_end FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "path": row[1],
            "type": row[2],
            "name": row[3],
            "class_name": row[4],
            "content": row[5],
            "ghost_text": row[6],
            "signature": row[7],
            "content_hash": row[8],
            "start_line": row[9],
            "end_line": row[10],
        }

    def get_nodes_by_ids(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for nid in node_ids:
            n = self.get_node(nid)
            if n:
                out.append(n)
        return out

    def get_node_ids_by_path(self, path: str) -> List[str]:
        """Return all node IDs for the given path (relative posix)."""
        self.connect()
        assert self._conn is not None
        rows = self._conn.execute("SELECT id FROM nodes WHERE path = ?", (path,)).fetchall()
        return [r[0] for r in rows]

    def delete_nodes_by_paths(self, paths: List[str]) -> List[str]:
        """
        Delete all nodes whose path is in the list. Remove any edges touching them.
        Returns list of deleted node IDs.
        """
        if not paths:
            return []
        self.connect()
        assert self._conn is not None
        placeholders = ",".join("?" * len(paths))
        cursor = self._conn.execute(
            f"SELECT id FROM nodes WHERE path IN ({placeholders})",
            paths,
        )
        node_ids = [r[0] for r in cursor.fetchall()]
        if not node_ids:
            return []
        id_placeholders = ",".join("?" * len(node_ids))
        self._conn.execute(
            f"DELETE FROM edges WHERE source_id IN ({id_placeholders}) OR target_id IN ({id_placeholders})",
            node_ids + node_ids,
        )
        self._conn.execute(
            f"DELETE FROM nodes WHERE id IN ({id_placeholders})",
            node_ids,
        )
        self._conn.commit()
        logger.info("ShadowIndex: deleted %d nodes for %d paths", len(node_ids), len(paths))
        return node_ids

    def save_edges(self, edges: List[Dict[str, str]]) -> None:
        """edges: [{"source": id, "target": id, "edge_type": "CALLS"|...}]"""
        self.connect()
        assert self._conn is not None
        logger.info("ShadowIndex: saving %d edges (replacing existing)", len(edges))
        self._conn.execute("DELETE FROM edges")
        for e in edges:
            self._conn.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type) VALUES (?, ?, ?)",
                (e["source"], e["target"], e["edge_type"]),
            )
        self._conn.commit()
        logger.info("ShadowIndex: saved %d edges", len(edges))

    def load_edges(self) -> List[Dict[str, str]]:
        self.connect()
        assert self._conn is not None
        rows = self._conn.execute("SELECT source_id, target_id, edge_type FROM edges").fetchall()
        return [{"source": r[0], "target": r[1], "edge_type": r[2]} for r in rows]

    def load_all_nodes(self) -> List[Dict[str, Any]]:
        self.connect()
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT id, path, type, name, class_name, content, ghost_text, signature, content_hash, line_start, line_end FROM nodes"
        ).fetchall()
        return [
            {
                "id": r[0],
                "path": r[1],
                "type": r[2],
                "name": r[3],
                "class_name": r[4],
                "content": r[5],
                "ghost_text": r[6],
                "signature": r[7],
                "content_hash": r[8],
                "start_line": r[9],
                "end_line": r[10],
            }
            for r in rows
        ]

    def node_count(self) -> int:
        self.connect()
        assert self._conn is not None
        return self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
