"""Ask the DBpedia agent one demo question."""
# Make sure to run with a large context:
#  OLLAMA_CONTEXT_LENGTH=32768 ollama serve

from langchain_core.messages import HumanMessage

from _dbpedia import build_dbpedia_agent

agent = build_dbpedia_agent()

question = "What countries border Germany?"

print(f"USER: {question}")

result = agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config={"recursion_limit": 30},
)

print(f"AGENT: {result['messages'][-1].content.strip()}")
