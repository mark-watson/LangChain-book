# A Perplexity-style local search agent

I subscribe to [Perplexity](https://www.perplexity.ai) and use it most days. It does one thing very well: given a natural-language question, it searches the web, reads the top pages, and synthesizes a multi-paragraph answer that cites the sources it used. That pattern — search, filter, fetch, summarize, synthesize — is broadly useful and about eighty lines of Python to build for yourself. This chapter builds it as a LangGraph pipeline, running entirely on your laptop, with Ollama for the LLM, DuckDuckGo for search, and `trafilatura` for HTML-to-text extraction.

This is also the last chapter of Part I. Everything that follows in Part II covers LlamaIndex — a different framework with a different mental model, but many of the same underlying ideas.

## The pipeline

Five nodes, one long straight edge, no conditional routing:

```text
START -> search -> filter -> fetch -> summarize -> synthesize -> END
```

Each node takes state in and produces a partial state update — the same shape as every graph in Chapters 6 through 12. The state accumulates as the pipeline runs:

| Field after node | Contents |
|---|---|
| `raw_results` | The top ~5 DuckDuckGo results for the query. |
| `filtered_results` | The subset the model judged relevant to the query. |
| `pages` | Full text of each filtered URL, pulled with `trafilatura`. |
| `summaries` | One per-page summary focused on the query. |
| `final_answer` | The multi-paragraph synthesis. |

Nothing in this graph requires an agent's decision-making. It is a straight pipeline. LangGraph is still the right tool because each stage benefits from being an isolated, streamable, replaceable unit — but no conditional edges, no loops, no ReAct.

Setup:

```console
$ cd source-code/local_search
$ uv sync
$ ollama pull qwen3:8b
```

## The graph

`_pipeline.py`, in full:

```python
from typing import TypedDict

import trafilatura
from duckduckgo_search import DDGS
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

model = ChatOllama(model="qwen3:8b", temperature=0)

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
```

Notes on the individual nodes.

**`search_node`.** DuckDuckGo does not require an API key. Its `.text()` method is rate-limited, which is why we cap at five results. If the endpoint fails (occasional for busy times of day), we return an empty list and let the pipeline continue — the synthesis node handles the "no sources" case.

**`filter_node`.** One LLM call per raw result, each a very short prompt. This is where the pipeline spends most of its tokens per pass; it is also what separates good results from noise. If your model is slow, you can drop this node and take a small quality hit — but for local models where quality is the bottleneck, filtering pays for itself.

**`fetch_node`.** `trafilatura.extract()` handles the whole "HTML to clean text" problem well enough that this node is three lines of real logic. We cap page text at 6000 characters because larger inputs occasionally overwhelm smaller local models, and the model does not usually need more than a page or two of context to summarize accurately.

**`summarize_node`.** One LLM call per fetched page. The prompt explicitly says "only material relevant to the query" — without that, the model tends to summarize the whole page, which then dilutes the synthesis step downstream.

**`synthesize_node`.** One final LLM call combining the per-page summaries into the actual answer. The prompt tells the model not to include a source list — Perplexity does that in its UI, and adding it inline tends to make local models produce noisy citation strings that do not link to anything.

## Running it

`01_search.py`:

```python
from _pipeline import build_pipeline

app = build_pipeline()

query = "What are the main challenges in running large language models on consumer laptops?"

initial: dict = {
    "query": query,
    "raw_results": [],
    "filtered_results": [],
    "pages": [],
    "summaries": [],
    "final_answer": "",
}

final = app.invoke(initial)

print("=== FINAL ANSWER ===")
print(final["final_answer"])
```

A representative run (specific wording will vary):

```console
$ uv run 01_search.py
USER: What are the main challenges in running large language models on consumer laptops?

=== FINAL ANSWER ===
Consumer laptops face three main obstacles when running large language models
locally. The first is memory: even quantized models in the 7-13 billion
parameter range typically need 8-16 GB of RAM just to load, and models beyond
30 B parameters are effectively out of reach for anything under 32 GB.

The second is inference speed. On CPU-only laptops or those without a
compatible GPU, token generation rates for a mid-sized model can be under
five tokens per second, which is too slow for interactive use. Apple Silicon
and NVIDIA GPUs bring this into a usable range, but at the cost of hardware
compatibility ...

... The third challenge is thermal management. Sustained inference at high
GPU utilization pushes laptop cooling systems close to their limits, causing
throttling that can cut throughput in half after a few minutes of continuous
generation.
```

The exact sources and phrasing will differ every run — DuckDuckGo returns different results at different times, and the model has some non-determinism even at `temperature=0`. That is inherent to the pattern, not a bug.

Total run time is typically 10-60 seconds. The dominant cost is the five per-source summarize calls; if you want it faster, drop `MAX_RESULTS` to three.

## Watching each stage

`02_stream_search.py` streams the same query and shows one line per stage:

```python
for step in app.stream(initial):
    for node_name, node_output in step.items():
        print(f"=== node: {node_name} ===")
        if node_name == "search":
            for r in node_output["raw_results"]:
                print(f"  - {r.get('title', '')}")
                print(f"    {r.get('href') or r.get('url')}")
        elif node_name == "filter":
            print(f"  kept {len(node_output['filtered_results'])} results")
        elif node_name == "fetch":
            for p in node_output["pages"]:
                print(f"  fetched {len(p['text'])} chars from {p['url']}")
        elif node_name == "summarize":
            for i, s in enumerate(node_output["summaries"]):
                snippet = s if len(s) < 200 else s[:200] + "..."
                print(f"  [{i}] {snippet}")
        elif node_name == "synthesize":
            print(node_output["final_answer"])
        print()
```

Streaming makes it very easy to see which stage is slow (usually summarize, occasionally fetch when a page is huge) and which stage is dropping quality. If the filter keeps everything, it is broken; if the summaries all say the same thing, the corpus is bad; if the synthesis contradicts individual summaries, the model is too small.

## Extending this

Everything from Chapters 8, 9, and 10 composes with this pipeline:

- **Add a checkpointer** and a `thread_id` so follow-up questions can reference prior searches ("of those, which is the cheapest option?"). The current graph has no memory between invocations.
- **Add an approval interrupt** before `fetch` if you want a human to prune the URL list before you spend the time on downloads.
- **Swap `synthesize` for a supervisor graph** that routes to different downstream specialists based on the question ("if the summaries are about code, delegate to the code specialist; if factual, to the KG specialist").

You can also swap the search backend. `duckduckgo-search` is the free default; if you want more consistent results, [Brave Search](https://api.search.brave.com/) offers a generous free tier and drops in with a two-line change to `search_node`.

## Wrapping up Part I

Part I has been a tour of what a solo developer can build with LangChain 1.0 and LangGraph 1.0 as open source libraries, without any of the commercial services LangChain Inc. sells on top. The primitives covered:

- Chat models, `.invoke`/`.stream`/`.batch`, LCEL, prompts, structured output, tool binding (Chapters 1-5).
- Retrieval patterns for RAG (Chapter 4).
- LangGraph state machines: state, nodes, reducers, conditional edges (Chapter 6).
- ReAct agents built on those primitives (Chapter 7).
- Durability via checkpointers (Chapter 8), HITL via interrupts and state editing (Chapter 9), multi-agent supervisor patterns (Chapter 10).
- Applied to a SQL database (Chapter 11), knowledge graphs (Chapter 12), and web search (this chapter).

Everything above runs on your laptop with the packages listed in "The Stack We're Building On." No LangSmith, no LangGraph Cloud, no LangSmith Deployment, no LlamaCloud, no LlamaParse.

Part II covers the same design space with LlamaIndex — starting with its own quick tour, then RAG patterns, then the Workflows API that plays a role similar to LangGraph in the LlamaIndex ecosystem.
