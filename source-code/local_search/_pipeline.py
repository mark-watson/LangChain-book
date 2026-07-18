"""A five-node LangGraph pipeline that behaves like a small local Perplexity.

Nodes, in order:

  search      -> DuckDuckGo search for the query. Populates `raw_results`.
  filter      -> One quick LLM call per raw result asks "is this relevant?".
                 Populates `filtered_results` with only the ones that pass.
  fetch       -> Downloads each filtered URL and extracts clean text with
                 `trafilatura`. Populates `pages`.
  summarize   -> One LLM call per page: "summarize this text with only
                 material relevant to the query." Populates `summaries`.
  synthesize  -> One final LLM call combines the summaries into a single
                 multi-paragraph answer. Populates `final_answer`.

Every node is a plain Python function from state to a partial state
update — exactly the same shape as the graphs in Chapters 6 through 10.
"""

from typing import TypedDict

import trafilatura
from ddgs import DDGS
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

model = ChatOllama(model="qwen3.5:4b", temperature=0)

MAX_RESULTS = 5
MAX_PAGE_CHARS = 6000


class State(TypedDict):
    query: str
    raw_results: list[dict]
    filtered_results: list[dict]
    pages: list[dict]
    summaries: list[str]
    final_answer: str


def search_node(state: State) -> dict:
    try:
        results = list(DDGS().text(state["query"], max_results=MAX_RESULTS))
    except Exception as exc:
        results = []
        print(f"  [search failed: {exc}]")
    return {"raw_results": results}


def filter_node(state: State) -> dict:
    """Keep only search results the model judges relevant to the query."""
    kept = []
    for r in state["raw_results"]:
        snippet = r.get("body", "")
        prompt = (
            "Reply with a single character, either Y or N. "
            f"Is the following snippet relevant to the query {state['query']!r}?\n\n"
            f"{snippet}"
        )
        answer = model.invoke(prompt).content.strip().upper()
        if answer.startswith("Y"):
            kept.append(r)
    return {"filtered_results": kept}


def fetch_node(state: State) -> dict:
    """Download the page text for each filtered result."""
    pages = []
    for r in state["filtered_results"]:
        url = r.get("href") or r.get("url")
        if not url:
            continue
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded) or ""
        except Exception:
            text = ""
        if text:
            pages.append({"url": url, "title": r.get("title", ""), "text": text[:MAX_PAGE_CHARS]})
    return {"pages": pages}


def summarize_node(state: State) -> dict:
    """One summary per page, focused on material relevant to the query."""
    summaries = []
    for p in state["pages"]:
        prompt = (
            f"Summarize the following text, including only material relevant to the "
            f"query {state['query']!r}. Keep it to at most three sentences.\n\n"
            f"{p['text']}"
        )
        summaries.append(model.invoke(prompt).content.strip())
    return {"summaries": summaries}


def synthesize_node(state: State) -> dict:
    """Combine the per-page summaries into a final multi-paragraph answer."""
    if not state["summaries"]:
        return {"final_answer": "No usable sources were found."}

    joined = "\n\n---\n\n".join(state["summaries"])
    prompt = (
        f"Using the following per-source summaries, write a clear, multi-paragraph "
        f"answer to the query {state['query']!r}. Do not repeat information across "
        f"paragraphs. Do not include a list of sources.\n\n{joined}"
    )
    return {"final_answer": model.invoke(prompt).content.strip()}


def build_pipeline():
    graph = StateGraph(State)
    graph.add_node("search", search_node)
    graph.add_node("filter", filter_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "filter")
    graph.add_edge("filter", "fetch")
    graph.add_edge("fetch", "summarize")
    graph.add_edge("summarize", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
