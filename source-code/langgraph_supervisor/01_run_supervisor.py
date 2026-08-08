"""Invoke the supervisor graph on three test questions.

The third question ("population of Canada times 2") is the interesting
one — it forces the supervisor to route to research first, then math,
then finish.
"""

import time

from langchain_core.messages import HumanMessage

from _supervisor import build_supervisor

DEBUG = True
app = build_supervisor(debug=DEBUG)

QUESTIONS = [
    "What is 137 times 24?",
    "What is the population of Canada?",
    "What is the population of Canada times 2?",
]

for q in QUESTIONS:
    print(f"USER: {q}")
    t0 = time.monotonic()
    try:
        result = app.invoke(
            {"messages": [HumanMessage(content=q)], "next": ""},
            config={"recursion_limit": 25},
        )
        elapsed = time.monotonic() - t0
        final = result["messages"][-1]
        if DEBUG:
            print(f"DEBUG: graph invoke took {elapsed:.2f}s")
            print(f"DEBUG: result has {len(result['messages'])} message(s), "
                  f"types: {[type(m).__name__ for m in result['messages']]}")
            print(f"DEBUG: final message type = {type(final).__name__}, "
                  f"content repr = {final.content!r}")
        print(f"FINAL: {final.content.strip()[:300]}\n")
    except Exception as exc:
        print(f"ERROR after {time.monotonic() - t0:.2f}s: {exc}")
        if DEBUG:
            import traceback

            traceback.print_exc()
        print()
