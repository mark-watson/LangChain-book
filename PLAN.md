# Rewrite Plan: "LangChain and LlamaIndex for Solo Developers"

*A plan document for the next edition. Nothing in the manuscript changes until this plan is approved.*

---

## 1. Why a rewrite is needed

Since the previous edition (May 2025), both frameworks have pivoted noticeably toward commercial platforms:

- **LangChain Inc.** now positions LangSmith (observability, evals, deployment, sandboxes, Engine, Fleet, LLM Gateway) as the paid layer on top of the MIT-licensed `langchain` and `langgraph` libraries. Seat-based pricing scales from $39/seat/month to enterprise contracts in the $200k+ range. LangChain 1.0 (Oct 2025) and LangGraph 1.0 (Oct 2025) are the current stable OSS releases.
- **LlamaIndex** now brands itself as "the leading document agent and OCR platform," fronted by LlamaCloud and LlamaParse (credits-based, $1.25 per 1,000 credits, 1–90 credits per page). The OSS framework (`llama-index-core` + 300+ integration packages) is still MIT-licensed and fully self-hostable, but it is no longer the marketing centerpiece.

The existing edition mixes coverage of the OSS libraries with references to paid services (SerpAPI, Zapier, Google KG API), a chapter mentioning LangSmith / LangChain Hub, and outdated `initialize_agent` / `LLMChain` / `GPTSimpleVectorIndex` APIs that have been superseded.

This rewrite refocuses the book on **what a solo developer or small team can build on a single laptop with MIT-licensed code and open endpoints**, using LangChain 1.0 and LlamaIndex 0.14+ as they exist in 2026.

---

## 2. Target reader

The new book is for **one specific reader profile**:

- **Solo developers and small teams** (2–5 people) building LLM-powered apps for themselves, their consulting clients, or small products.
- **Comfortable with Python** but not necessarily deep AI/ML researchers. Want working examples they can extend.
- **Value control over convenience.** They want to be able to run the whole stack on a laptop or a small VPS, understand every dependency, and not be a captive customer of any vendor.
- **Willing to pay per-token for LLM inference** (Gemini, Fireworks.ai, OpenAI) or run models locally via Ollama — but **not** willing to pay a per-seat or per-trace SaaS bill for LangSmith / LangGraph Cloud / LlamaCloud / LlamaParse.
- **Skeptical of platform lock-in.** Want abstractions that let them swap the LLM, the vector store, or the framework itself.

Explicit non-goals:
- Not written for enterprise teams that already run LangSmith or need SOC-2 compliant managed deployment.
- Not a JavaScript / TypeScript book (Python only, matching the current edition).
- Not a survey of every framework — LangChain and LlamaIndex are chosen because they are the two most widely used OSS abstractions and are the ones the reader is most likely to encounter in the wild.

---

## 3. What to keep (with modernization)

These chapters have durable value; the writing stays, but code is rewritten against current APIs (LangChain 1.0 / LlamaIndex 0.14+) and stripped of any paid-platform references.

| Existing chapter | Modernization needed |
|---|---|
| `Preface.md` | Rewrite. New "target reader" framing. Remove the "commercial integrations we don't cover" section — it's now the whole point of the book. |
| `LLM-overview.md` | Light refresh: 2026 model landscape (Llama 4, Qwen3, Gemma 3, Mistral 3.1, DeepSeek R1 for local; Gemini 2.5, GPT-4.1, Claude Sonnet 4.6 for API). Keep the "big tech vs. small dev" framing — it's now more relevant. |
| `LangChain-overview.md` | Rewrite for LangChain 1.0. Show install via `uv add langchain langchain-ollama langchain-openai`. Replace `LLMChain` / `initialize_agent` with `ChatOllama`, `.invoke()`, and LCEL runnables. |
| `Ollama.md` | Update model list (drop deprecated model refs, add Qwen3, Gemma 3, GPT-OSS). Replace deprecated `langchain.llms.Ollama` with `langchain_ollama.ChatOllama`. Add tool-calling model guidance. |
| `RAG.md` | Rewrite as the flagship "RAG on your laptop" chapter. Show both a LangChain path and a LlamaIndex path over the same corpus. Cover chunking, embeddings, and reranking with open models. |
| `KG.md` (DBpedia + Wikidata) | Keep. Rewrite queries and update `llama_index` imports (the current code uses `GPTSimpleVectorIndex` / `GPTTreeIndex` which no longer exist). Add a LangGraph agent that uses SPARQL as a tool. |
| `Agents.md` | Rewrite around LangGraph 1.0 (`create_react_agent`, `StateGraph`, `ToolNode`, `SqliteSaver`). Move DBpedia custom-tool example here. |
| `Extraction.md` | Keep the prompt-driven approach but rewrite examples with Pydantic + `.with_structured_output()` on both `ChatOllama` and `ChatOpenAI`. |
| `Sumarization.md` | Modernize: show map-reduce and refine chains against a local model; drop the raw OpenAI SDK example. |
| `StructuredData.md` | Fold into `Extraction.md`. Show CSV → JSON via `.with_structured_output(Pydantic)` — one page, not a chapter. |
| `SqLite.md` | Rewrite. `langchain_experimental.sql` is deprecated. Use `create_sql_agent` from `langchain_community` or a LangGraph SQL agent. Emphasize local SQLite; keep the "NLP-over-DB is now a solved problem" wrap-up. |
| `HuggingFace.md` | Trim heavily. Drop the LangChain HuggingFaceHub wrapper (rate limits + login issues make it a poor first example). Keep a short "run a small HF model locally via `transformers` pipeline" and hand off to Ollama for the main story. |
| `LlamaIndex_case_study.md` | Rewrite as the Part II opening chapter. Show `VectorStoreIndex` + `SimpleDirectoryReader` + local `HuggingFaceEmbedding` + `Ollama` LLM — zero OpenAI dependency. |
| `llm_search.md` (Perplexity-style multi-prompt search) | Keep and update. This is a genuinely useful pattern. Update model, and add a LangGraph variant. |

---

## 4. What to delete

These come out of the manuscript entirely. Move the removed prose + code into `manuscript/DEPRECATED/` (as this repo already does with the sibling examples repo) so nothing is truly lost.

| Chapter / section | Reason for removal |
|---|---|
| `LangChain-overview.md` — SerpAPI / Google Serper example | Requires paid API key. Replace with DuckDuckGo Search (still MIT + free). |
| `SearchKG.md` (Google Knowledge Graph API) | Paid Google Cloud API. Public DBpedia/Wikidata cover the same use case for free. |
| `GoogleDrive.md` | Requires OAuth setup, `pydrive` is unmaintained, and it locks the reader into Google. **Keep the personal narrative** (the story about the author's earlier custom Clojure Dropbox project and the "gentleman scientist" framing — this is exactly the kind of solo-dev voice the new edition wants to preserve) but replace the Google Drive code with a "load documents from a local folder / iCloud / Dropbox folder" recipe. Rename the chapter to something like "Indexing your own documents." |
| `Recipes.md` (Zapier and old cooking recipes example) | Zapier integration is proprietary. The recipe example is a light demo that no longer earns its chapter slot. |
| `MiniChain.md` | Referenced but not in `Book.txt`. Delete file. |
| `AdvancedExperiments.md` | All three sub-chapters are `TBD` placeholders. Delete unless the user wants to actually write them (see §5). |
| `OtherUsefulLibraries.md` — EmbedChain + Kor | EmbedChain is now `mem0` and has pivoted commercially; Kor is essentially unmaintained. Replace with a short section on **Instructor** and **Outlines** for structured extraction. |
| `LangChain-overview.md` — LangSmith / LangChain Hub mentions | Out of scope for this edition. Cite them in one paragraph in the Preface as "not covered" and move on. |
| Any references to LangChain 0.2/0.3 module paths (`langchain.llms.*`, `langchain.chains.RetrievalQA`, `langchain.vectorstores.*`) | These are legacy. Rewrite against `langchain_core`, `langchain_ollama`, `langchain_openai`, `langchain_community.vectorstores.*`. |
| The `Llamacpp.md` chapter | Ollama supersedes it for the target reader. Delete entirely — no side note in the Ollama chapter. |

---

## 5. Proposed new chapters and sections

New material to add, grouped by the two parts.

### Front matter
- **New Preface**: one page on why "no platform, no lock-in" matters in 2026, and one page on what tools the reader needs installed (Python 3.12, `uv`, Ollama, and a text editor — nothing else is mandatory).
- **New chapter — "The stack we're building on"**: a single-page dependency map showing exactly which packages the book uses (`langchain`, `langgraph`, `langchain-ollama`, `langchain-openai`, `llama-index-core`, `llama-index-llms-ollama`, `llama-index-embeddings-huggingface`, `chromadb`, `duckduckgo-search`, `SPARQLWrapper`) and which packages we deliberately don't touch (`langsmith`, `langgraph-sdk` cloud client, `llama-cloud-services`, `llama-parse`).

### Part I — LangChain
1. LangChain 1.0 in one hour (LCEL, runnables, `.invoke`/`.stream`/`.batch`, chat models, tool binding)
2. Prompt templates, few-shot, structured output with Pydantic
3. Embeddings and vector stores you can run locally (Chroma, FAISS, sqlite-vec)
4. RAG patterns with LangChain (naive → reranked → hybrid → multi-query)
5. Tools and tool-calling with local models (which Ollama models actually support tools in 2026, and why the wrong one silently fails)
6. **LangGraph 1.0 fundamentals** (StateGraph, nodes, edges, conditional routing, reducers)
7. **Building a ReAct agent with LangGraph + Ollama**
8. **Persistent, durable agents with SqliteSaver** (the killer OSS feature of LangGraph 1.0 — restart-safe agents without any cloud service)
9. **Human-in-the-loop patterns** (interrupts, approval nodes, editing agent state)
10. **Multi-agent supervisor pattern** in pure OSS LangGraph
11. Natural-language SQLite queries with a LangGraph SQL agent
12. DBpedia + Wikidata SPARQL as agent tools (rewritten `KG.md` example)
13. Perplexity-style multi-prompt web search agent using DuckDuckGo + Ollama (rewritten `llm_search.md`)

### Part II — LlamaIndex
14. LlamaIndex 0.14 in one hour (documents, nodes, indices, query engines — and why `GPTSimpleVectorIndex` is gone)
15. `SimpleDirectoryReader` + local embeddings via `HuggingFaceEmbedding` (BGE, nomic-embed) — zero OpenAI
16. Vector, tree, keyword, and hybrid indices — when to use each
17. RAG with reranking using `SentenceTransformerRerank` (all local)
18. **The Workflows API** (event-driven step composition — the recommended 2026 pattern)
19. **Building an agent as a Workflow** (ReAct with `FunctionAgent`, tool calling with local models)
20. Query pipelines across multiple indices (per-corpus routing)
21. Structured data extraction with `PydanticProgram` on a local model
22. Serving a Workflow with `llama-deploy` on a laptop (Redis + FastAPI, no cloud)

### Cross-cutting appendices
- **A. A discussion of model sizes and what they're good for** — deliberately *not* a "best model of the month" recommendation. Instead, an evergreen discussion of what small (1–4B), medium (7–14B), and large (30B+) local models tend to be good and bad at, when a hosted model (Gemini, Fireworks.ai, OpenAI) is worth the token cost, and how to think about tool-calling vs. chat-only vs. embedding vs. reranking model roles. Written to age gracefully as specific model names change.
- **B. Evaluation without LangSmith** — how to score RAG and agent outputs using a local model as judge, plus a note on the open-source alternatives (Langfuse, MLflow, Phoenix/Arize OpenInference).
- **C. Observability without LangSmith** — one page on `set_debug(True)`, callbacks, and pointing Workflows' auto-instrumentation at a local OpenInference collector.
- **D. Deployment on a $5/month VPS** — how to actually put a LangGraph or LlamaIndex Workflow app in front of users without any managed service.

---

## 6. Table of contents (as actually shipped)

This was originally proposed with slots 2, 3, and 5 reserved for standalone
prompts/structured-output, embeddings/vector-store, and tool-calling
deep-dive chapters. Those three were never written — the material was
folded into Chapters 1 and 4 instead, per the decision recorded in
`NOTES_from_Claude.md` — and the book was renumbered to close the gaps
rather than ship with three dangling chapter numbers. Every internal
"Chapter N" cross-reference in the manuscript and every `source-code/*/`
`pyproject.toml`/`README.md` "Chapter N —" label uses this numbering.

```
Preface
The stack we're building on

PART I — LangChain
  1. LangChain 1.0 in one hour
  2. RAG patterns with LangChain
  3. LangGraph 1.0 fundamentals
  4. Building a ReAct agent
  5. Durable, restart-safe agents
  6. Human-in-the-loop
  7. Multi-agent supervisor
  8. Natural-language SQLite
  9. DBpedia and Wikidata as agent tools
 10. A Perplexity-style local search agent

PART II — LlamaIndex
 11. LlamaIndex 0.14 in one hour
 12. Local documents and local embeddings
 13. Choosing an index type
 14. RAG with reranking
 15. The Workflows API
 16. Agents as Workflows
 17. Multi-index query pipelines
 18. Structured extraction with PydanticProgram
 19. Serving Workflows with FastAPI (the `llama-deploy` plan below didn't
     pan out — see `M.md` for why)

Appendices
  A. Model sizes and what they're good for (a discussion)
  B. Evaluation without LangSmith
  C. Observability without LangSmith
  D. Deploying on a $5/month VPS

Wrap-up
```

---

## 7. Cross-cutting rewrite rules

Applied everywhere, not just in one chapter:

- **No LangSmith, LangGraph Cloud/Deployment, LangChain Hub, LangServe managed, LlamaCloud, LlamaParse, LlamaAgents, or `llama-cloud-services` package.** If a topic requires them, it doesn't go in the book.
- **No paid third-party APIs required.** SerpAPI, Google KG, Zapier, and Google Drive are out. DuckDuckGo, Brave Search (free tier is generous), and public SPARQL endpoints are in. If an example calls Gemini / Fireworks.ai / OpenAI, that must be **for the LLM only**, and there must always be a working Ollama alternative shown in the same chapter.
- **Every code example runs on the reader's laptop** with `uv run examples/<name>.py`. No Colab notebooks required (Colab can be mentioned as a "if you don't have a GPU" fallback for a couple of heavier examples).
- **Pin versions.** Each chapter's code directory has a `pyproject.toml` with pinned `langchain`, `langgraph`, `llama-index-core`, etc. This is already partly done in the current edition — extend it to every chapter.
- **Prefer `uv` over `pip`.** `uv` is now the de facto Python packaging tool and matches the "small, self-contained, low-friction" ethos.
- **Migrate off deprecated imports.** No `langchain.llms.*`, `langchain.chains.LLMChain`, `initialize_agent`, `GPTSimpleVectorIndex`, `GPTTreeIndex`, `download_loader`, or `SQLDatabaseChain`.
- **All code lives in this repo.** The old external examples repo (`github.com/mark-watson/langchain-book-examples`) is being deprecated. Every example in the new edition lives in `./source-code/<chapter-slug>/` alongside the manuscript, with its own `pyproject.toml`. Book text references code by relative path only — no external repo URLs to go stale.
- **Preserve the author's voice.** The current edition's "solo dev / gentleman scientist / here's why I built this for myself" narrative is a differentiator, not fluff. Modernize the tech in those chapters but keep the personal framing.
- **No head-to-head "LangChain vs LlamaIndex" chapter.** Each framework gets its own Part; the reader learns each on its own terms and picks based on the individual chapters, not a comparison matrix.

---

## 8. Working title

**Confirmed title:** *LangChain and LlamaIndex: Open Source Recipes for Solo Developers*

---

## 9. Suggested rewrite order

If the user wants to write incrementally rather than all at once:

1. Preface + "The stack we're building on" (defines the contract with the reader).
2. Part I chapters 1, 4, 6, 7 (LangChain basics → RAG → LangGraph → ReAct agent). This alone is a shippable "v1" of the new edition.
3. Part II chapters 14, 15, 17, 18, 19 (LlamaIndex basics → local RAG → reranking → Workflows → agent).
4. The specialty chapters (SQLite, KG, search agent, deployment).
5. Appendices last.

---

## 10. Author decisions locked in (from review on 2026-07-16)

- **Title:** *LangChain and LlamaIndex: Open Source Recipes for Solo Developers*.
- **`Llamacpp.md`:** deleted outright, no side note in the Ollama chapter.
- **No head-to-head LangGraph vs. LlamaIndex Workflows chapter.**
- **Personal author's-story framing is preserved** (see the updated Google Drive row in §3 and the new rewrite rule in §7).
- **Model-sizes appendix stays discussion-style**, not a "best model" table.
- **Code examples now live in `./source-code/` inside this repo.** The old external `langchain-book-examples` repo is being deprecated. Book text references code by relative path only.
