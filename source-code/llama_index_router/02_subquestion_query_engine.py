"""SubQuestionQueryEngine: decompose a compound query into per-index subquestions.

Where RouterQueryEngine picks one tool, SubQuestionQueryEngine plans a
series of subquestions — potentially one per index — runs them in
parallel, and synthesizes the results into a single answer.

The right choice for compound "compare A and B" questions that no single
index can answer alone.
"""

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
print(f"USER: {query}\n")
print(f"AGENT: {engine.query(query)}")
