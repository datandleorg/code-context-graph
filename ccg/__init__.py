"""Code Context Graph (CCG) — syntax-aware code indexing and search."""

from .parser import parse_codebase, ScopeNode
from .graph import CodeGraph
from .shadow_index import ShadowIndex
from .vector_store import VectorStore
from .reranker import Reranker
from .collector import get_llm_context
from .runner import ingest_codebase, search_codebase
from .agent import run_agent

__all__ = [
    "parse_codebase",
    "ScopeNode",
    "CodeGraph",
    "ShadowIndex",
    "VectorStore",
    "Reranker",
    "get_llm_context",
    "ingest_codebase",
    "search_codebase",
    "run_agent",
]
