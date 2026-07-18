"""Shared tools used across the three example scripts.

Two tools: one deterministic arithmetic tool, and one network tool that
searches DuckDuckGo. The `@tool` decorator turns each plain function into
an object with the right shape for `.bind_tools()` and `ToolNode`.
"""

from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return their product."""
    return a * b


@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo for a query and return the top three text results."""
    from ddgs import DDGS

    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as exc:
        return f"Search failed: {exc}"

    if not results:
        return "No results."

    return "\n\n".join(
        f"- {r.get('title', '')}\n  {r.get('body', '')}" for r in results
    )


TOOLS = [multiply, web_search]
