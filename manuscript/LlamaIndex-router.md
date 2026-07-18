# Multi-index query pipelines

Every example so far in Part II has used a single index over a single corpus. Real projects rarely look like that. A support-desk assistant has a docs index, a runbooks index, and a changelog index. A research assistant has one index per paper set. A personal knowledge base has one index per notebook, project, or year.

LlamaIndex has two prebuilt patterns for querying across multiple indices: **`RouterQueryEngine`** and **`SubQuestionQueryEngine`**. Both wrap several per-corpus query engines and use an LLM to decide how to combine them.

Everything lives in `source-code/llama_index_router/`. For teaching purposes each of the four `../data/*.txt` files becomes its own tiny index — chemistry, economics, health, sports. In a real project the same code would drive four indices over four folders of hundreds of documents each.

Setup:

```console
$ cd source-code/llama_index_router
$ uv sync
$ ollama pull qwen3:8b
```

## `RouterQueryEngine`: pick one index

For queries that clearly belong to one corpus, you want the router to send the whole query to that one index and get a synthesized answer back. `01_router_query_engine.py`:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

from _indices import build_per_topic_engines

engines = build_per_topic_engines()

tools = [
    QueryEngineTool.from_defaults(
        query_engine=engines["chemistry"],
        description="Questions about chemistry, matter, substances, and lab work.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=engines["economics"],
        description="Questions about economics, markets, and schools of economic thought.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=engines["health"],
        description="Questions about human health, disease, exercise, and well-being.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=engines["sports"],
        description="Questions about sports, athletic activity, and physical competition.",
    ),
]

router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools,
)

for q in [
    "What is chemistry?",
    "What is the Austrian School of Economics?",
    "How does exercise affect the body?",
]:
    print(f"USER: {q}")
    print(f"AGENT: {router.query(q)}\n")
```

The mechanism is simple. Each per-topic engine gets wrapped in a `QueryEngineTool` with an English description. The `LLMSingleSelector` reads the query plus all the descriptions and picks one tool. The router forwards the query to that tool and returns the tool's response.

The quality of routing depends entirely on the quality of the tool descriptions. Vague descriptions produce wrong routing; overlapping descriptions produce coin-flip routing. This is the same discipline as writing good tool docstrings for a ReAct agent.

`LLMMultiSelector.from_defaults()` is the sibling class that picks *multiple* tools and combines their responses — useful when questions might legitimately touch two or three indices.

## `SubQuestionQueryEngine`: decompose and combine

For compound questions that no single index can answer alone — "compare A and B," "how do X, Y, and Z relate?" — you want the engine to plan a series of subquestions, run each against the appropriate index, and synthesize a combined answer.

`02_subquestion_query_engine.py`:

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from _indices import build_per_topic_engines

engines = build_per_topic_engines()

tools = [
    QueryEngineTool(
        query_engine=engines["chemistry"],
        metadata=ToolMetadata(
            name="chemistry_index",
            description="Corpus about chemistry, matter, substances, and lab work.",
        ),
    ),
    QueryEngineTool(
        query_engine=engines["economics"],
        metadata=ToolMetadata(
            name="economics_index",
            description="Corpus about economics, markets, and schools of economic thought.",
        ),
    ),
    QueryEngineTool(
        query_engine=engines["health"],
        metadata=ToolMetadata(
            name="health_index",
            description="Corpus about human health, disease, exercise, and well-being.",
        ),
    ),
    QueryEngineTool(
        query_engine=engines["sports"],
        metadata=ToolMetadata(
            name="sports_index",
            description="Corpus about sports, athletic activity, and physical competition.",
        ),
    ),
]

engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    use_async=False,
)

query = "How do sports, health, and chemistry relate to one another?"
print(engine.query(query))
```

Behind the scenes: an LLM planner reads the compound query and the tool descriptions, produces a list of subquestions (typically one per relevant tool), each subquestion runs against its target index, and a final LLM call synthesizes the subquestion answers into a coherent response.

Costs to be aware of. With four tools and a compound query, you may end up with three or four subquestion runs plus two synthesis calls — six or seven LLM calls per user query. On a local model this adds latency; on a hosted model it adds a real bill. `SubQuestionQueryEngine` is powerful but not the tool to reach for on every question.

## When to reach for which

- **`RouterQueryEngine` with `LLMSingleSelector`** — most queries clearly belong to one corpus. Cheap: one selector call plus one downstream query engine call.
- **`RouterQueryEngine` with `LLMMultiSelector`** — most queries belong to one or two corpora. Slightly more expensive; useful when your corpora overlap.
- **`SubQuestionQueryEngine`** — compound "compare / relate / synthesize" queries that no single index can answer. Most expensive; reach for it when you have evidence users actually ask these questions.

You can also build routing yourself with the Workflows API from Chapters 18-19: a classify step, a routing step, per-corpus engines behind separate steps. That is what you would do when your routing logic is deterministic (based on user role, request metadata, or a fixed classification) rather than LLM-driven.

## What we covered

- `RouterQueryEngine` sends each query to one (or a few) of several per-corpus engines via LLM-driven routing.
- `SubQuestionQueryEngine` decomposes compound queries into per-corpus subquestions and synthesizes the results.
- Both patterns depend heavily on the English descriptions attached to each `QueryEngineTool`. Treat descriptions like production API contracts.

Chapter 21 covers structured extraction — using `PydanticProgram` to force LLM output into a validated data schema.
