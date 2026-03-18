"""
Cross-encoder re-ranker (BGE-Reranker) for two-stage retrieval.
If FlagEmbedding fails to load (e.g. transformers version mismatch), falls back to vector-search order.
"""

import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    """
    Re-rank (query, document) pairs with a cross-encoder; return top_n by score.
    Falls back to preserving vector-search order if the model cannot be loaded.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._fallback_no_rerank = False

    def _get_model(self) -> Any:
        if self._fallback_no_rerank:
            return None
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self._model_name)
                logger.info("Loaded reranker %s", self._model_name)
            except Exception as e:
                logger.warning(
                    "FlagReranker not available (%s). Re-ranking disabled; using vector-search order.",
                    e,
                )
                self._fallback_no_rerank = True
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        candidates: list of (node_id, text).
        Returns list of (node_id, score) for top_n, sorted by score descending.
        """
        if not candidates:
            return []
        model = self._get_model()
        if model is None:
            return [(c[0], 1.0 - i * 0.01) for i, c in enumerate(candidates[:top_n])]
        pairs = [(query, text) for _id, text in candidates]
        try:
            scores = model.compute_score(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]
            indexed = list(zip([c[0] for c in candidates], scores))
            indexed.sort(key=lambda x: -x[1])
            return indexed[:top_n]
        except Exception as e:
            logger.warning("Reranker compute_score failed: %s; returning by order", e)
            return [(c[0], 1.0 - i * 0.01) for i, c in enumerate(candidates[:top_n])]
