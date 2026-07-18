"""Ask the SQL agent four English questions about the sample company database."""

from langchain_core.messages import HumanMessage

from _agent import build_sql_agent

agent = build_sql_agent()

QUESTIONS = [
#    "How many employees are there?",
    "Which employee has the highest salary?",
    "Which customer has generated the most total revenue?",
    "What is the total revenue per department?",
]

for q in QUESTIONS:
    print(f"USER: {q}")
    result = agent.invoke(
        {"messages": [HumanMessage(content=q)]},
        config={"recursion_limit": 30},
    )
    final = result["messages"][-1]
    print(f"AGENT: {final.content.strip()}\n")
