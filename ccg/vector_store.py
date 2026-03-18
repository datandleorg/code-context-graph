"""
Vector store: bi-encoder embeddings (OpenAI or sentence-transformers) and Qdrant or in-memory ANN search.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default model when using sentence-transformers
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI embedding model default and dimensions (for when using OpenAI)
OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
OPENAI_MODEL_DIMENSIONS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Batch size for OpenAI API (stay under token limits; 100 texts is safe)
OPENAI_EMBED_BATCH_SIZE = 100

# Max chars per input for OpenAI embeddings (8192 token limit ≈ 32k chars; use 24k to stay safe)
OPENAI_EMBED_MAX_CHARS = 24_000


class VectorStore:
    """
    Embed text with a bi-encoder (OpenAI or sentence-transformers) and run vector search (Qdrant or in-memory).
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        qdrant_url: Optional[str] = None,
        collection_name: str = "ccg",
        openai_api_key: Optional[str] = None,
        embedding_provider: Optional[str] = None,
    ) -> None:
        self._model_name = embedding_model
        self._qdrant_url = qdrant_url
        self._collection_name = collection_name
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._embedding_provider = embedding_provider
        self._model: Any = None
        self._dimensions: Optional[int] = None
        self._qdrant_client: Any = None
        self._in_memory_vectors: List[Tuple[str, List[float]]] = []
        self._use_qdrant = bool(qdrant_url)
        self._use_openai = self._resolve_use_openai()

    def _resolve_use_openai(self) -> bool:
        if self._embedding_provider == "openai":
            return True
        if self._embedding_provider == "sentence-transformers":
            return False
        if self._openai_api_key and self._model_name in OPENAI_MODEL_DIMENSIONS:
            return True
        if self._openai_api_key and "text-embedding" in self._model_name:
            return True
        return False

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._use_openai:
            if not self._openai_api_key:
                raise ValueError("OpenAI embeddings require openai_api_key (or OPENAI_API_KEY env)")
            self._dimensions = OPENAI_MODEL_DIMENSIONS.get(
                self._model_name,
                OPENAI_MODEL_DIMENSIONS.get(OPENAI_DEFAULT_MODEL, 1536),
            )
            self._model = "openai"
            logger.info("Using OpenAI embeddings: model=%s, dim=%s", self._model_name, self._dimensions)
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded sentence-transformers model %s (dim=%s)", self._model_name, self._dimensions)
        except Exception as e:
            logger.error("Failed to load sentence-transformers: %s", e)
            raise
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._get_model()
        assert self._dimensions is not None
        return self._dimensions

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=self._openai_api_key)
        # Truncate each text to stay under OpenAI's 8192-token limit per input
        truncated = [
            t[:OPENAI_EMBED_MAX_CHARS] + ("..." if len(t) > OPENAI_EMBED_MAX_CHARS else "")
            for t in texts
        ]
        out: List[List[float]] = []
        for i in range(0, len(truncated), OPENAI_EMBED_BATCH_SIZE):
            batch = truncated[i : i + OPENAI_EMBED_BATCH_SIZE]
            resp = client.embeddings.create(model=self._model_name, input=batch)
            out.extend([d.embedding for d in resp.data])
        return np.array(out, dtype=np.float32)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        if self._use_openai:
            self._get_model()
            return self._embed_openai(texts)
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True)

    def embed_single(self, text: str) -> List[float]:
        arr = self.embed([text])
        return arr[0].tolist()

    def _get_qdrant(self):
        if self._qdrant_client is None and self._use_qdrant:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._qdrant_client = QdrantClient(url=self._qdrant_url)
            collections = [c.name for c in self._qdrant_client.get_collections().collections]
            if self._collection_name not in collections:
                self._qdrant_client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
                )
            logger.info("Qdrant client connected to %s", self._qdrant_url)
        return self._qdrant_client

    def delete_ids(self, node_ids: List[str]) -> None:
        """Remove vectors for the given node IDs (for incremental ingest)."""
        if not node_ids:
            return
        ids_set = set(node_ids)
        if self._use_qdrant:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            client = self._get_qdrant()
            client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="node_id", match=MatchAny(any=node_ids))]
                ),
            )
            logger.info("Deleted %d vectors from Qdrant", len(node_ids))
        else:
            self._in_memory_vectors = [(nid, v) for nid, v in self._in_memory_vectors if nid not in ids_set]
            logger.info("Deleted %d vectors from in-memory store", len(node_ids))

    def upsert(self, node_ids: List[str], texts: List[str]) -> None:
        if len(node_ids) != len(texts):
            raise ValueError("node_ids and texts length must match")
        vectors = self.embed(texts)
        if self._use_qdrant:
            from qdrant_client.models import PointStruct
            client = self._get_qdrant()
            points = [
                PointStruct(
                    id=abs(hash(nid)) % (2**63),
                    vector=vec.tolist(),
                    payload={"node_id": nid},
                )
                for nid, vec in zip(node_ids, vectors)
            ]
            client.upsert(collection_name=self._collection_name, points=points)
            logger.info("Upserted %d vectors to Qdrant", len(points))
        else:
            # Merge: remove existing entries for these ids, then append new
            ids_set = set(node_ids)
            self._in_memory_vectors = [(nid, v) for nid, v in self._in_memory_vectors if nid not in ids_set]
            self._in_memory_vectors.extend(zip(node_ids, vectors.tolist()))
            logger.info("Upserted %d vectors in-memory (total %d)", len(node_ids), len(self._in_memory_vectors))

    def search(self, query_vector: List[float], k: int = 50) -> List[Tuple[str, float]]:
        """
        Return list of (node_id, score) sorted by similarity (higher better).
        """
        if self._use_qdrant:
            from qdrant_client.models import PointStruct
            client = self._get_qdrant()
            results = client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=k,
                with_payload=True,
            )
            out: List[Tuple[str, float]] = []
            for p in results.points:
                nid = p.payload.get("node_id") if p.payload else None
                score = float(p.score) if p.score is not None else 0.0
                if nid:
                    out.append((nid, score))
            return out
        # In-memory: cosine similarity
        if not self._in_memory_vectors:
            return []
        q = np.array(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores: List[Tuple[str, float]] = []
        for nid, vec in self._in_memory_vectors:
            v = np.array(vec, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            sim = float(np.dot(q, v))
            scores.append((nid, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def clear(self) -> None:
        if self._use_qdrant:
            client = self._get_qdrant()
            try:
                client.delete_collection(self._collection_name)
            except Exception:
                pass
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )
        else:
            self._in_memory_vectors = []

    def save_to_dir(self, path: str | Path) -> None:
        """Persist in-memory vectors and node_ids to directory (for search in another process)."""
        if self._use_qdrant:
            return
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        ids = [nid for nid, _ in self._in_memory_vectors]
        vecs = [v for _, v in self._in_memory_vectors]
        np.save(path / "vectors.npy", np.array(vecs, dtype=np.float32))
        (path / "vector_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (path / "vector_meta.json").write_text(
            json.dumps({
                "model": self._model_name,
                "dimensions": self.dimensions,
                "embedding_provider": "openai" if self._use_openai else "sentence-transformers",
            }),
            encoding="utf-8",
        )

    def load_from_dir(self, path: str | Path) -> None:
        """Load in-memory vectors from directory (must match embedding model for query embedding)."""
        if self._use_qdrant:
            return
        path = Path(path)
        ids = json.loads((path / "vector_ids.json").read_text(encoding="utf-8"))
        arr = np.load(path / "vectors.npy")
        self._in_memory_vectors = [(nid, row.tolist()) for nid, row in zip(ids, arr)]
        meta_path = path / "vector_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._dimensions = meta.get("dimensions")
            if self._dimensions is None and self._in_memory_vectors:
                self._dimensions = len(self._in_memory_vectors[0][1])
        logger.info("Loaded %d vectors from %s", len(self._in_memory_vectors), path)
