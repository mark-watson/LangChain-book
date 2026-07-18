# Chapter 9 — Human-in-the-loop patterns

Three scripts that show how to pause a LangGraph run, get information (or approval, or an edit) from an outside caller, and then resume from exactly where the graph stopped.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

Only `03_edit_draft.py` calls an LLM. Scripts 1 and 2 are pure Python state-machine demos so the HITL mechanics are visible without any model noise.

## Scripts

| Script | What it shows |
|---|---|
| `01_interrupt_basic.py` | The minimum viable `interrupt()` demo — pause a node, resume with `Command(resume=value)`. No LLM. |
| `02_approval_gate.py` | A ReAct-style agent with a "dangerous" tool (`send_email`) that is gated by a human approval interrupt. The safe tool (`multiply`) runs without asking. |
| `03_edit_draft.py` | `interrupt_after=["propose"]` + `update_state()` — the graph pauses after producing a draft, a human edits it, then the graph runs the "refine" node on the edited draft. |

Run any one with:

```console
$ uv run 01_interrupt_basic.py
```
