"""Two shared tools: a deterministic multiply and a live web search."""


def multiply(a: int, b: int) -> int:
    """Multiply two integers and return their product."""
    return a * b


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
