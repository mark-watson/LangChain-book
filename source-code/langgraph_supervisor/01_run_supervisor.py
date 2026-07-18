"""Invoke the supervisor graph on three test questions.

The third question ("population of Canada times 2") is the interesting
one — it forces the supervisor to route to research first, then math,
then finish.
"""

from langchain_core.messages import HumanMessage

from _supervisor import build_supervisor

app = build_supervisor()

QUESTIONS = [
    "What is 137 times 24?",
    "What is the population of Canada?",
    "What is the population of Canada times 2?",
]

for q in QUESTIONS:
    print(f"USER: {q}")
    result = app.invoke(
        {"messages": [HumanMessage(content=q)], "next": ""},
        config={"recursion_limit": 25},
    )
    final = result["messages"][-1]
    print(f"FINAL: {final.content.strip()[:300]}\n")
