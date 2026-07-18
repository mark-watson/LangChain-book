"""Walk every checkpoint recorded for a thread.

`agent.get_state(config)` returns the current state.
`agent.get_state_history(config)` yields every checkpoint the graph has
recorded for the thread, newest first. Each item carries the values,
the metadata (which node produced it, when), and a `config` you can pass
back to `invoke` to time-travel to that checkpoint.

Run 02_sqlite_first_run.py and 03_sqlite_second_run.py first so there is
some history to inspect.
"""

from langgraph.checkpoint.sqlite import SqliteSaver

from _graph import build_graph

CHECKPOINT_DB = "checkpoints.db"
THREAD_ID = "sqlite-demo"

with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
    agent = build_graph().compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": THREAD_ID}}

    print(f"=== Current state for thread {THREAD_ID!r} ===")
    current = agent.get_state(config)
    for m in current.values["messages"]:
        print(f"  {type(m).__name__}: {m.content[:100]}")

    print(f"\n=== Checkpoint history (newest first) ===")
    for i, snapshot in enumerate(agent.get_state_history(config)):
        n_messages = len(snapshot.values.get("messages", []))
        meta = snapshot.metadata or {}
        source = meta.get("source", "?")
        step = meta.get("step", "?")
        print(f"  [{i}] step={step} source={source} messages_len={n_messages}")
