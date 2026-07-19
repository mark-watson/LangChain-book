# Appendix C. Observability without LangSmith

"Observability" for an LLM app usually means one thing: "when a user reports a bug, can I reconstruct what the model actually saw and what it actually returned?" That is a small enough problem that you do not need a paid platform to solve it. This appendix covers three layers of local observability that between them are enough for solo-dev-scale projects.

## Layer 1: `set_debug(True)`

For LangChain, one line at the top of your script gets you verbose logging of every LLM call, every prompt, every tool invocation:

```python
from langchain.globals import set_debug
set_debug(True)
```

For LlamaIndex, the equivalent is:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This is the "print statement" of observability. It is not pretty and not searchable, but it is instant and it costs nothing. For a five-minute investigation of "why did the agent do that?", it is usually enough.

## Layer 2: `.stream()` for structured tracing

Both frameworks make traces first-class via streaming. Every LangGraph agent's `.stream()` yields one item per node execution, showing exactly which state field each node updated. LlamaIndex Workflows have a similar mechanism via the `Context.write_event_to_stream()` API. Chapters 4, 7, and 8 all showed streaming in action.

For interactive development, `.stream()` output is more informative than dashboards. You see the shape of what happened, in order, in a form you can grep, save, and diff against yesterday's run. When I hit a bug in an agent I have written, my first move is almost always to re-run it with `.stream()` and read the trace.

## Layer 3: OpenInference + Phoenix

For everything more structured than "read the print output" — collecting traces from multiple users, comparing them across runs, dashboarding aggregate metrics — the open standard is [**OpenInference**](https://github.com/Arize-ai/openinference), an OpenTelemetry-based schema for LLM traces. Both LangChain and LlamaIndex ship OpenInference exporters.

The receiver most people pair with them is [**Phoenix**](https://github.com/Arize-ai/phoenix) from Arize. It runs as a single Python process locally:

```console
$ pip install arize-phoenix
$ phoenix serve
```

Point your LangChain / LlamaIndex OpenInference exporter at `http://localhost:6006` and every trace shows up in the Phoenix UI: prompts, responses, timings, tool calls, retrieval hits. Similar shape to LangSmith, no bill, runs on your laptop.

If you outgrow Phoenix — you want a shared service across a team, or you want long retention — the same OpenInference traces can be shipped to any OpenTelemetry-compatible backend: Jaeger, Grafana Tempo, self-hosted or managed. This is one of the concrete arguments for staying on open standards: your instrumentation code does not change when your storage does.

## Layer 4: what LangSmith gives you that Phoenix does not

To be honest about the tradeoff. LangSmith's differentiators over an OpenInference + Phoenix setup:

- **Prompt versioning UI**, with diff and rollback. You can approximate this with Git.
- **A "Playground" for iterating on prompts** against real traces. You can approximate this by pasting the prompt into whatever tool you use for interactive iteration.
- **Managed retention and search across long time periods** without you running any infrastructure. You can approximate this by running Phoenix on a small VPS (Appendix D covers the practical setup).
- **LangGraph Studio integration** — a graphical debugger for LangGraph runs. Free for local use; paid tiers for hosted. If you love graphical debuggers, worth trying even if you skip the rest of LangSmith.

For a solo developer building small-to-medium projects, none of the above justify the subscription cost. For a five-person team building on top of LangGraph, it is a conversation worth having.

## Practical setup

If you follow one recommendation from this appendix, make it this one:

1. Install Phoenix locally (`pip install arize-phoenix`).
2. Run `phoenix serve` in a terminal.
3. Add the OpenInference exporter to your LangChain / LlamaIndex app (three lines each; see their docs).
4. Bookmark `http://localhost:6006`.

That gets you traces for every run of every app you build for the rest of the project's life. When a bug arrives, you have the evidence to debug it.
