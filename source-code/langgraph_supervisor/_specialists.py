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
    result = a + b
    if _DEBUG:
        print(f"  [DEBUG] add({a}, {b}) = {result}")
    return result


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    result = a * b
    if _DEBUG:
        print(f"  [DEBUG] multiply({a}, {b}) = {result}")
    return result


@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo. Returns the top three text results."""
    from ddgs import DDGS

    if _DEBUG:
        print(f"  [DEBUG] web_search({query!r})")
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as exc:
        msg = f"Search failed: {exc}"
        if _DEBUG:
            print(f"  [DEBUG] web_search error: {msg}")
        return msg
    if not results:
        if _DEBUG:
            print(f"  [DEBUG] web_search: no results")
        return "No results."
    found = "\n\n".join(
        f"- {r.get('title', '')}\n  {r.get('body', '')}" for r in results
    )
    if _DEBUG:
        print(f"  [DEBUG] web_search: {len(results)} results ({len(found)} chars)")
    return found


_DEBUG = False


def _make_model(debug: bool = False):
    global _DEBUG
    _DEBUG = debug
    return ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False, verbose=debug)


_model = _make_model()
research_agent = create_react_agent(_model, [web_search])
math_agent = create_react_agent(_model, [add, multiply])

SPECIALIST_NAMES = ["research", "math"]


def setup_debug(debug: bool) -> None:
    """Recreate specialist agents with debug mode enabled (or disabled)."""
    global _DEBUG, _model, research_agent, math_agent
    _DEBUG = debug
    _model = _make_model(debug=debug)
    research_agent = create_react_agent(_model, [web_search])
    math_agent = create_react_agent(_model, [add, multiply])
