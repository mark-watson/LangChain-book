"""Stream the supervisor graph so you can watch each routing decision.

Each yielded item is {node_name: partial_state_update}. Watching the
supervisor node emit `{"next": ...}` values makes the pattern very legible.
"""

from langchain_core.messages import HumanMessage

from _supervisor import build_supervisor

DEBUG = True
app = build_supervisor(debug=DEBUG)

question = "What is the population of Canada times 2?"

print(f"USER: {question}\n")

try:
    for step in app.stream(
        {"messages": [HumanMessage(content=question)], "next": ""},
        config={"recursion_limit": 25},
    ):
        for node_name, node_output in step.items():
            print(f"=== node: {node_name} ===")
            if "next" in node_output:
                print(f"  next -> {node_output['next']!r}")
            for m in node_output.get("messages", []):
                snippet = (
                    m.content if len(m.content) < 250 else m.content[:250] + "..."
                )
                print(f"  {type(m).__name__}: {snippet}")
            print()
except Exception as exc:
    print(f"ERROR: {exc}")
    if DEBUG:
        import traceback

        traceback.print_exc()
    print()
