"""
Code Context Graph (CCG) — CLI entry point.

Usage:
  python main.py ingest <root_path> [--id ID] [--full]
  python main.py search "<query>" [--id ID]
  python main.py watch <root_path> [--id ID]  # incremental ingest on file change
  python main.py agent --id ID
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

from ccg.runner import ingest_codebase, search_codebase

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Code Context Graph — ingest, search, or agent")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_p = sub.add_parser("ingest")
    ingest_p.add_argument("root_path", help="Root path to index")
    ingest_p.add_argument("--id", dest="index_id", metavar="ID", help="Named index ID (saved under CCG_INDEX_ROOT/ID; use for agent)")
    ingest_p.add_argument("--index-dir", help="Override index directory (default: <root>/.ccg or CCG_INDEX_ROOT/ID if --id)")
    ingest_p.add_argument("--openai-api-key", help="OpenAI API key for embeddings (or set OPENAI_API_KEY)")
    ingest_p.add_argument("--embedding-model", help="Embedding model (e.g. text-embedding-3-small, or sentence-transformers/all-MiniLM-L6-v2)")
    ingest_p.add_argument("--full", action="store_true", help="Force full ingest (disable incremental / manifest)")
    watch_p = sub.add_parser("watch", help="Watch repo and run incremental ingest on file changes")
    watch_p.add_argument("root_path", help="Root path to watch")
    watch_p.add_argument("--id", dest="index_id", metavar="ID", help="Named index ID (same as ingest --id)")
    watch_p.add_argument("--index-dir", help="Override index directory")
    watch_p.add_argument("--debounce", type=float, default=2.0, help="Seconds to wait after last change before ingest (default 2)")
    watch_p.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY)")
    watch_p.add_argument("--embedding-model", help="Embedding model (must match index)")
    search_p = sub.add_parser("search")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--id", dest="index_id", metavar="ID", help="Named index ID (from ingest --id)")
    search_p.add_argument("--index-dir", default=".ccg", help="Path to index directory (ignored if --id set)")
    search_p.add_argument("--openai-api-key", help="OpenAI API key for query embedding (or set OPENAI_API_KEY)")
    search_p.add_argument("--embedding-model", help="Embedding model (must match index; default from vector_meta)")
    search_p.add_argument("--top-k", type=int, default=5, help="Results after rerank")
    search_p.add_argument("--initial-k", type=int, default=50, help="Vector search candidates")
    search_p.add_argument("--max-hops", type=int, default=1, help="Graph search hops from seeds (1=one-hop; 2+=BFS)")
    search_p.add_argument("--max-graph-nodes", type=int, default=50, help="Max nodes from graph search when max-hops>=2")
    agent_p = sub.add_parser("agent", help="ReAct agent with code RAG tool (interactive)")
    agent_p.add_argument("--id", dest="index_id", metavar="ID", required=True, help="Named index ID to RAG over (from ingest --id)")
    agent_p.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model (default: gpt-4o-mini)")
    agent_p.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY)")
    serve_p = sub.add_parser("serve", help="Run FastAPI server (ingest, search, search/summarize)")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_p.add_argument("--port", type=int, default=8010, help="Port")
    args = parser.parse_args()

    if args.command == "ingest":
        config = {}
        if getattr(args, "index_id", None):
            config["index_id"] = args.index_id
        if getattr(args, "index_dir", None):
            config["index_dir"] = args.index_dir
        if getattr(args, "openai_api_key", None):
            config["openai_api_key"] = args.openai_api_key
        if getattr(args, "embedding_model", None):
            config["embedding_model"] = args.embedding_model
        incremental = not getattr(args, "full", False)
        result = ingest_codebase(args.root_path, config=config or None, incremental=incremental)
        print(result)
    elif args.command == "watch":
        from ccg.watcher import run_watcher
        config = {}
        if getattr(args, "index_id", None):
            config["index_id"] = args.index_id
        if getattr(args, "index_dir", None):
            config["index_dir"] = args.index_dir
        if getattr(args, "openai_api_key", None):
            config["openai_api_key"] = args.openai_api_key
        if getattr(args, "embedding_model", None):
            config["embedding_model"] = args.embedding_model
        run_watcher(
            args.root_path,
            config=config or None,
            debounce_seconds=getattr(args, "debounce", 2.0),
        )
    elif args.command == "search":
        config = {}
        if getattr(args, "index_id", None):
            config["index_id"] = args.index_id
        else:
            config["index_dir"] = args.index_dir
        if getattr(args, "openai_api_key", None):
            config["openai_api_key"] = args.openai_api_key
        if getattr(args, "embedding_model", None):
            config["embedding_model"] = args.embedding_model
        result = search_codebase(
            args.query,
            top_k=getattr(args, "top_k", 5),
            initial_k=getattr(args, "initial_k", 50),
            max_hops=getattr(args, "max_hops", 1),
            max_graph_nodes=getattr(args, "max_graph_nodes", 50),
            config=config,
        )
        if "error" in result:
            print(result["error"])
        else:
            print(result.get("context", result))
    elif args.command == "agent":
        from ccg.agent import run_agent
        index_id = getattr(args, "index_id", None)
        if not index_id:
            print("Error: agent requires --id <index_id> (use an index created with ingest --id)")
            raise SystemExit(1)
        run_agent(
            index_id=index_id,
            model=getattr(args, "model", "gpt-4o-mini"),
            openai_api_key=getattr(args, "openai_api_key", None),
        )
    elif args.command == "serve":
        import uvicorn
        uvicorn.run(
            "ccg.server:app",
            host=getattr(args, "host", "0.0.0.0"),
            port=getattr(args, "port", 8010),
            reload=False,
        )
