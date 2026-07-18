"""Stream one question so you can watch every tool call the agent makes.

For debugging SQL agents, `.stream()` is essential — it shows every query
the model writes, every schema lookup, and every checker pass, in order.
"""

from langchain_core.messages import HumanMessage

from _agent import build_sql_agent

agent = build_sql_agent()

question = "What is the total revenue per department?"

print(f"USER: {question}\n")

for step in agent.stream(
    {"messages": [HumanMessage(content=question)]},
    config={"recursion_limit": 30},
):
    for node_name, node_output in step.items():
        print(f"=== node: {node_name} ===")
        for m in node_output.get("messages", []):
            if getattr(m, "tool_calls", None):
                for call in m.tool_calls:
                    args = str(call["args"])
                    if len(args) > 200:
                        args = args[:200] + "..."
                    print(f"  tool_call: {call['name']}({args})")
            if m.content:
                snippet = m.content if len(m.content) < 300 else m.content[:300] + "..."
                print(f"  {type(m).__name__}: {snippet}")
        print()
