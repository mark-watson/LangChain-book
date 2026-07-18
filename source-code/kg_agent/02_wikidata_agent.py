"""Ask the Wikidata agent one demo question."""

from langchain_core.messages import HumanMessage

from _wikidata import build_wikidata_agent

agent = build_wikidata_agent()

question = "When was Bill Clinton president of the United States?"

print(f"USER: {question}")

result = agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config={"recursion_limit": 30},
)

print(f"AGENT: {result['messages'][-1].content.strip()}")
