# The Stack We're Building On

Before writing a single line of application code it is worth spelling out exactly which packages this book uses, and which packages I have deliberately left out. The goal is that after reading this chapter you can look at any `pyproject.toml` in the `source-code/` directory and know why every dependency is there.

## The whole book, in one dependency list

Across every chapter in this book, the total set of Python packages you will install is small. Here it is, grouped by purpose:

**Core frameworks**

- `langchain` — the LangChain 1.0 core library.
- `langchain-core` — the primitives (`Runnable`, `BaseMessage`, `PromptTemplate`, etc.) that `langchain` and `langgraph` share.
- `langchain-community` — first-party integrations that live outside the core (SQL agents, some vector stores, some tools).
- `langgraph` — the LangGraph 1.0 stateful-agent framework.
- `llama-index-core` — the LlamaIndex 0.14+ core library.

**LLM providers (pick as needed per chapter)**

- `langchain-ollama` and `llama-index-llms-ollama` — the default LLM provider for the book. Local models via Ollama.
- `langchain-openai` and `llama-index-llms-openai` — for readers who want to use OpenAI or an OpenAI-compatible endpoint such as Fireworks.ai.
- `langchain-google-genai` and `llama-index-llms-gemini` — for readers who want to use Google's Gemini API.

**Embeddings and vector stores**

- `langchain-huggingface` and `llama-index-embeddings-huggingface` — local embedding models (BGE, nomic-embed).
- `chromadb` — a local vector store that persists to disk in a single directory.
- `faiss-cpu` — used in a couple of chapters where an in-memory FAISS index is a better fit than Chroma.
- `sentence-transformers` — pulled in transitively for the reranker examples.

**Data-source and tool packages**

- `ddgs` — the free web search backend used throughout.
- `SPARQLWrapper` and `rdflib` — for the DBpedia and Wikidata chapters.
- `trafilatura` — for pulling clean text out of arbitrary web pages.
- `pydantic` — for structured-output schemas.

**Utility**

- `python-dotenv` — for loading API keys out of a `.env` file if you use hosted models.
- `sqlite-utils` — occasionally handy in the SQLite chapter.

Every chapter's `source-code/<chapter>/pyproject.toml` pins the exact versions used when the chapter was written, so a chapter that works today will keep working tomorrow.

## What we are deliberately not installing

These are the packages that a search for "LangChain" or "LlamaIndex" will turn up first, and that we do not use anywhere in this book. It is worth naming them explicitly so you know why they are absent.

- **`langsmith`** — the client for LangSmith observability, evaluation, and prompt-hub features. LangSmith is a paid SaaS product ($39/seat/month for the Plus tier, plus per-trace usage). Everything the book does for observability we do with local tools like `set_debug(True)`, callbacks, and (optionally) a self-hosted OpenInference collector.
- **`langgraph-sdk`** (cloud client) — the client for the managed LangGraph deployment platform. We use `langgraph` itself, which is MIT-licensed and runs anywhere Python runs.
- **`langserve`** (in its managed form) — where we serve LangGraph or LlamaIndex apps, we use plain FastAPI plus the framework's own primitives.
- **`llama-cloud-services`** and **`llama-parse`** — the SDKs for LlamaCloud and LlamaParse. Both are credits-based commercial services. Where we need document parsing we use `pypdf`, `unstructured`, or Python's own text handling, plus the OSS ingestion pipeline built into `llama-index-core`.
- **`llama-index-agent-openai`** as the *primary* agent story — LlamaIndex now recommends the Workflows API for new agent development, and that is what Part II uses.

If you already pay for one or more of these platforms and want to use them alongside the code in this book, nothing here will stop you. But nothing here will require you to.

## A note on `uv` and `pyproject.toml`

Every chapter's example directory looks like this:

```text
source-code/<chapter-slug>/
  pyproject.toml
  README.md
  example_1.py
  example_2.py
  ...
```

To run any example, `cd` into its directory and do:

```console
$ uv sync
$ uv run example_1.py
```

`uv sync` reads `pyproject.toml`, creates a `.venv` in the chapter directory, and installs exactly the pinned versions. `uv run` executes a script inside that venv. You never need to `source .venv/bin/activate` manually, and you never need to think about which Python interpreter is active.

If you prefer plain `pip`, each chapter's `pyproject.toml` is a normal PEP 621 file and works with `pip install -e .` in a manually created virtual environment. The book uses `uv` for its snippets because it is faster and it avoids the "which environment am I in" class of bugs that used to eat an hour of every new reader's weekend.

## The one-time setup

If you do these three things once, every chapter in the book will work:

1. Install `uv` (`brew install uv` on macOS, or `curl -LsSf https://astral.sh/uv/install.sh | sh` on Linux).
2. Install Ollama from [ollama.com](https://ollama.com) and pull one general-purpose model that supports tool calling. As of mid-2026 the models I use most often for the book examples are `qwen3.5:4b` for tool-calling work, `llama3.2:3b` when I want something small and fast, and `gemma3:12b-it-qat` when I want more headroom on a 16 GB machine. Appendix A discusses model selection in more depth.
3. Optionally, put a `.env` file in your home directory with any hosted-model API keys you want to use:

```text
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
FIREWORKS_API_KEY=...
```

That is the whole setup. You are ready for Part I.
