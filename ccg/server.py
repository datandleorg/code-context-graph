"""
FastAPI server: ingest, search, and LLM-summarised search endpoints.
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ccg.runner import ingest_codebase, search_codebase, clear_index, clear_all_indexes

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CCG API",
    description="Code Context Graph — ingest, semantic + graph search, and LLM-summarised search",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    root_path: str = Field(..., description="Absolute or relative path to the repo root to index")
    index_id: Optional[str] = Field(None, description="Named index ID (stored under CCG_INDEX_ROOT/ID)")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key for embeddings (or set OPENAI_API_KEY)")
    embedding_model: Optional[str] = Field(None, description="e.g. text-embedding-3-small")


class IngestResponse(BaseModel):
    files_found: int
    nodes_created: int
    edges_created: int
    index_id: Optional[str] = None
    index_dir: Optional[str] = None
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language search query")
    index_id: Optional[str] = Field(None, description="Named index ID (from ingest with --id)")
    index_dir: Optional[str] = Field(None, description="Path to index directory (if not using index_id)")
    top_k: int = Field(5, ge=1, le=50, description="Number of results after rerank")
    initial_k: int = Field(50, ge=1, le=200, description="Vector search candidates before rerank")
    max_hops: int = Field(1, ge=1, le=5, description="Graph search hops from seeds (1=one-hop expansion; 2+=BFS graph search)")
    max_graph_nodes: int = Field(50, ge=1, le=200, description="Max nodes to add from graph search (when max_hops>=2)")
    openai_api_key: Optional[str] = Field(None, description="For query embedding (or set OPENAI_API_KEY)")
    references_only: bool = Field(False, description="If true, return only file/function/line references (no code content)")


class CodeReference(BaseModel):
    """Full node object for a search hit (same shape as graph nodes, without content)."""
    id: str = Field(..., description="Node id (path::type::name::line)")
    type: str = Field(..., description="Node type: file, class, function, etc.")
    path: str = Field(..., description="Relative file path")
    name: str = Field(..., description="Symbol name")
    class_name: Optional[str] = Field(None, description="Enclosing class if any")
    signature: str = Field("", description="Function/class signature snippet")
    line_start: Optional[int] = Field(None, description="Start line (1-based)")
    line_end: Optional[int] = Field(None, description="End line (1-based)")


class SearchResponse(BaseModel):
    query: str
    context: str = Field("", description="Raw code context (empty when references_only=true)")
    references: List[CodeReference] = Field(default_factory=list, description="File, function, and line range for each hit")
    error: Optional[str] = None


class SearchSummarizeRequest(BaseModel):
    query: str = Field(..., description="Natural-language question about the codebase")
    index_id: Optional[str] = Field(None, description="Named index ID")
    index_dir: Optional[str] = Field(None, description="Path to index directory")
    top_k: int = Field(5, ge=1, le=50)
    initial_k: int = Field(50, ge=1, le=200)
    max_hops: int = Field(1, ge=1, le=5, description="Graph search hops from seeds")
    max_graph_nodes: int = Field(50, ge=1, le=200, description="Max nodes from graph search")
    model: str = Field("gpt-4o-mini", description="OpenAI chat model for summary")
    openai_api_key: Optional[str] = Field(None)
    max_context_chars: int = Field(12000, description="Max context chars sent to LLM (to stay under context window)")


class SearchSummarizeResponse(BaseModel):
    query: str
    summary: str = Field(..., description="LLM-generated summary/answer from the retrieved code context")
    context_preview: Optional[str] = Field(None, description="First N chars of retrieved context")
    error: Optional[str] = None


class ClearIndexRequest(BaseModel):
    index_id: Optional[str] = Field(None, description="Named index ID to clear (under CCG_INDEX_ROOT)")
    index_dir: Optional[str] = Field(None, description="Path to index directory to clear (if not using index_id)")


class ClearIndexResponse(BaseModel):
    cleared: bool = Field(..., description="Whether the index was cleared")
    index_id: Optional[str] = None
    index_dir: Optional[str] = None
    error: Optional[str] = None


class ClearAllIndexesResponse(BaseModel):
    cleared: bool = Field(..., description="Whether the operation succeeded")
    cleared_ids: List[str] = Field(default_factory=list, description="List of index IDs that were removed")
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
def api_ingest(body: IngestRequest) -> IngestResponse:
    """Index a codebase. Use index_id to refer to it later in search and agent."""
    config: dict = {}
    if body.index_id:
        config["index_id"] = body.index_id
    if body.openai_api_key:
        config["openai_api_key"] = body.openai_api_key
    if body.embedding_model:
        config["embedding_model"] = body.embedding_model
    if os.environ.get("QDRANT_URL"):
        config["qdrant_url"] = os.environ.get("QDRANT_URL")

    result = ingest_codebase(body.root_path, config=config)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return IngestResponse(
        files_found=result["files_found"],
        nodes_created=result["nodes_created"],
        edges_created=result["edges_created"],
        index_id=result.get("index_id"),
        index_dir=result.get("index_dir"),
    )


@app.post("/search", response_model=SearchResponse)
def api_search(body: SearchRequest) -> SearchResponse:
    """Semantic + graph search. Returns raw context string (code snippets) for the query."""
    if not body.index_id and not body.index_dir:
        raise HTTPException(status_code=400, detail="Provide index_id or index_dir")
    config: dict = {}
    if body.index_id:
        config["index_id"] = body.index_id
    else:
        config["index_dir"] = body.index_dir
    if body.openai_api_key:
        config["openai_api_key"] = body.openai_api_key
    if os.environ.get("QDRANT_URL"):
        config["qdrant_url"] = os.environ.get("QDRANT_URL")
    config["max_hops"] = body.max_hops
    config["max_graph_nodes"] = body.max_graph_nodes
    config["references_only"] = body.references_only
    result = search_codebase(
        body.query,
        top_k=body.top_k,
        initial_k=body.initial_k,
        config=config,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    refs = [CodeReference(**r) for r in result.get("references", [])]
    return SearchResponse(
        query=result["query"],
        context=result.get("context", ""),
        references=refs,
    )


@app.post("/search/summarize", response_model=SearchSummarizeResponse)
def api_search_summarize(body: SearchSummarizeRequest) -> SearchSummarizeResponse:
    """Search the codebase and return an LLM-summarised answer (summary of the relevant code)."""
    if not body.index_id and not body.index_dir:
        raise HTTPException(status_code=400, detail="Provide index_id or index_dir")
    config: dict = {}
    if body.index_id:
        config["index_id"] = body.index_id
    else:
        config["index_dir"] = body.index_dir
    if body.openai_api_key:
        config["openai_api_key"] = body.openai_api_key
    if os.environ.get("QDRANT_URL"):
        config["qdrant_url"] = os.environ.get("QDRANT_URL")
    config["max_hops"] = body.max_hops
    config["max_graph_nodes"] = body.max_graph_nodes

    result = search_codebase(
        body.query,
        top_k=body.top_k,
        initial_k=body.initial_k,
        config=config,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    context = result.get("context", "")
    if not context.strip():
        return SearchSummarizeResponse(
            query=body.query,
            summary="No relevant code found for this query.",
            context_preview=None,
        )

    api_key = body.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key required for summarization (set OPENAI_API_KEY or pass in body)")

    truncated = context[: body.max_context_chars]
    if len(context) > body.max_context_chars:
        truncated += "\n\n[... context truncated ...]"

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=body.model,
        messages=[
            {
                "role": "system",
                "content": "You are a concise code expert. Given retrieved code context from a codebase search, summarize only what is relevant to the user's question. Be brief and direct; use bullet points if helpful. Do not invent code that is not in the context.",
            },
            {
                "role": "user",
                "content": f"User question: {body.query}\n\nRetrieved code context:\n\n{truncated}\n\nProvide a short summarised answer.",
            },
        ],
        temperature=0.2,
    )
    summary = (response.choices[0].message.content or "").strip()
    return SearchSummarizeResponse(
        query=body.query,
        summary=summary,
        context_preview=context[:2000] + ("..." if len(context) > 2000 else ""),
    )


@app.post("/index/clear", response_model=ClearIndexResponse)
def api_clear_index(body: ClearIndexRequest) -> ClearIndexResponse:
    """Clear a single index by index_id or index_dir. Deletes the index directory and all its contents."""
    if not body.index_id and not body.index_dir:
        raise HTTPException(status_code=400, detail="Provide index_id or index_dir")
    result = clear_index(index_id=body.index_id, index_dir=body.index_dir)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return ClearIndexResponse(
        cleared=result.get("cleared", True),
        index_id=result.get("index_id"),
        index_dir=result.get("index_dir"),
    )


@app.delete("/indexes", response_model=ClearAllIndexesResponse)
@app.post("/indexes/clear", response_model=ClearAllIndexesResponse)
def api_clear_all_indexes() -> ClearAllIndexesResponse:
    """Clear all named indexes under CCG_INDEX_ROOT. Each subdirectory is removed. Use DELETE /indexes or POST /indexes/clear."""
    result = clear_all_indexes()
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return ClearAllIndexesResponse(
        cleared=result.get("cleared", True),
        cleared_ids=result.get("cleared_ids", []),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def run_server(host: str = "0.0.0.0", port: int = 8010) -> None:
    import uvicorn
    uvicorn.run("ccg.server:app", host=host, port=port, reload=False)
