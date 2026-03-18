"""
ReAct agent that uses CCG search as a tool for code RAG.
Runs interactively: user input -> agent can call search_code(index_id, query) -> reply.
"""

import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def run_agent(
    index_id: str,
    model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None,
    top_k: int = 5,
    initial_k: int = 50,
    search_codebase_fn: Optional[Callable[..., Any]] = None,
) -> None:
    """
    Run an interactive ReAct agent that can search the codebase (by index_id) via a tool.

    search_codebase_fn: function(query, top_k, initial_k, config) -> dict with "context" or "error".
        If None, will be resolved from the runner module (used when invoked from CLI).
    """
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required for agent (set OPENAI_API_KEY or pass --openai-api-key)")

    if search_codebase_fn is None:
        from ccg.runner import search_codebase as search_codebase_fn

    def _search(query: str) -> str:
        result = search_codebase_fn(
            query,
            top_k=top_k,
            initial_k=initial_k,
            config={"index_id": index_id, "openai_api_key": api_key},
        )
        if "error" in result:
            return f"Error: {result['error']}"
        return result.get("context", "") or "(no results)"

    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    # Tool: RAG over the codebase identified by index_id
    @tool
    def search_code(query: str) -> str:
        """Search the indexed codebase for relevant code. Use this when you need to find functions, classes, or logic related to the user's question. Input should be a natural-language search query (e.g. 'auth login', 'database connection')."""
        return _search(query)

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    agent = create_react_agent(llm, [search_code])

    print(f"CCG ReAct agent (index_id={index_id}, model={model})")
    print("Ask questions about the codebase. The agent can search the index. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        try:
            events = agent.stream(
                {"messages": [("user", user_input)]},
                stream_mode="values",
            )
            final_event = None
            for event in events:
                final_event = event
            if final_event and "messages" in final_event and final_event["messages"]:
                last = final_event["messages"][-1]
                if hasattr(last, "content") and last.content:
                    print(f"Agent: {last.content}")
        except Exception as e:
            logger.exception("Agent error")
            print(f"Agent error: {e}")
