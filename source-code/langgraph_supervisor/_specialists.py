"""Two specialist agents, each a compiled ReAct graph.

Nothing in this module is new — every line uses primitives from Chapters 1
through 7. The point is that once you can build a single specialist agent
with `create_react_agent`, you can build several and orchestrate them.
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo. Returns the top three text results."""
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


_model = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False)

research_agent = create_react_agent(_model, [web_search])
math_agent = create_react_agent(_model, [add, multiply])

SPECIALIST_NAMES = ["research", "math"]
