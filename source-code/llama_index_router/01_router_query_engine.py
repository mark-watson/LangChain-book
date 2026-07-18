"""RouterQueryEngine: LLM picks the single best index for each query.

Each per-topic engine is wrapped in a QueryEngineTool with an English
description. The router LLM reads the query and the tool descriptions
and picks which tool to forward to.
"""

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
