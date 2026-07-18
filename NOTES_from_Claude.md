# Session Notes — Book Rewrite

Summary of the session that rewrote *LangChain and LlamaIndex: Open Source Recipes for Solo Developers* (2026 edition). Reference document for picking the work back up.

## The core reframing

The book has been rewritten around one thesis: **use LangChain and LlamaIndex as MIT-licensed open source libraries, avoid every commercial platform grown up around them**. Specifically avoided everywhere:

- LangSmith, LangSmith Deployment, LangSmith Engine, Sandboxes, Fleet, LLM Gateway
- LangChain Hub, LangGraph Cloud, managed LangServe
- LlamaCloud, LlamaParse, `llama-cloud-services`

Allowed for LLM inference (author's decision): local Ollama (default), plus Gemini / Fireworks.ai / OpenAI APIs where a hosted call adds real value. Also avoided: any paid third-party API where a free self-hostable alternative exists (SerpAPI, Google KG, Zapier).

## Structural changes locked in during planning

- **New title:** *LangChain and LlamaIndex: Open Source Recipes for Solo Developers*
- **Structure:** one combined book, **Part I — LangChain** (13 chapters) and **Part II — LlamaIndex** (9 chapters).
- **Agent framework:** LangGraph 1.0 is the primary agent story (five full chapters cover it).
- **KG chapters preserved and modernized:** DBpedia + Wikidata (Google KG dropped).
- **Author's personal-narrative voice preserved** — this is a differentiator vs. competing books.
- **Model discussion (Appendix A) is deliberately not opinionated** — a size/role taxonomy rather than a "best model of the month" table.
- **Code lives in `./source-code/` inside this repo.** Old external `langchain-book-examples` repo is being deprecated.
- **No head-to-head LangGraph vs. Workflows comparison chapter** — each framework gets its own Part.

Full plan captured in `PLAN.md` at repo root.

## What got written this session

### Front matter
- `manuscript/Preface.md` — rewritten around the new target reader + explicit list of what's not covered
- `manuscript/TheStack.md` — new chapter listing every package the book uses, every package deliberately avoided, and the `uv sync` / `uv run` convention

### Part I — LangChain / LangGraph (13 chapters)

| # | Manuscript file | Source-code dir | Focus |
|---|---|---|---|
| 1 | `LangChain-overview.md` | `langchain_getting_started/` | LangChain 1.0 primitives: chat models, `.invoke`/`.stream`/`.batch`, LCEL, prompts, structured output, `bind_tools` |
| 4 | `RAG.md` | `rag_langchain/` | Naive → reranked → hybrid → multi-query RAG patterns |
| 6 | `LangGraph-fundamentals.md` | `langgraph_fundamentals/` | StateGraph, nodes, reducers, conditional edges. Includes the "what LangGraph is / is not" paragraph |
| 7 | `Agents.md` | `langgraph_react_agent/` | `create_react_agent` prebuilt + same agent built manually with StateGraph + `ToolNode` |
| 8 | `LangGraph-durable.md` | `langgraph_durable/` | `MemorySaver` and `SqliteSaver` checkpointers; two-script demo for cross-process persistence |
| 9 | `LangGraph-hitl.md` | `langgraph_hitl/` | `interrupt()` / `Command(resume=...)`, `interrupt_after` + `update_state()` |
| 10 | `LangGraph-supervisor.md` | `langgraph_supervisor/` | Multi-agent supervisor with structured-output routing over `Literal[...]` |
| 11 | `SqLite.md` | `sql_agent/` | NL-to-SQL with `SQLDatabaseToolkit` + `create_react_agent`; self-contained sample DB |
| 12 | `KG.md` | `kg_agent/` | ReAct agents over DBpedia and Wikidata SPARQL endpoints |
| 13 | `llm_search.md` | `local_search/` | Perplexity-style pipeline as a 5-node LangGraph (search → filter → fetch → summarize → synthesize); ends with Part I recap |

**Not written (intentionally deferred):** Chapters 2, 3, and 5 slots from the plan (prompts deep-dive, embeddings/vector stores standalone, tool-calling deep-dive). Reason: material is largely covered inside Chapters 1 and 4 already. Author asked about this and left decision open — mentioned in earlier turn as "Note on the two open slots."

### Part II — LlamaIndex (9 chapters)

| # | Manuscript file | Source-code dir | Focus |
|---|---|---|---|
| 14 | `LlamaIndex_case_study.md` | `llama_index_intro/` | LlamaIndex 0.14 primitives: Document / Node / Index / QueryEngine, `Settings` config, persist/reload |
| 15 | `LlamaIndex-ingest.md` | `llama_index_ingest/` | SimpleDirectoryReader deep dive, custom Documents, embedding-model comparison, IngestionPipeline chunking |
| 16 | `LlamaIndex-indices.md` | `llama_index_indices/` | `SummaryIndex`, `SimpleKeywordTableIndex`, `QueryFusionRetriever` (BM25 + dense) |
| 17 | `LlamaIndex-rerank.md` | `llama_index_rerank/` | `SentenceTransformerRerank` as node postprocessor |
| 18 | `LlamaIndex-workflows.md` | `llama_index_workflows/` | Workflows API: `Workflow`, `@step`, `Event`, type-based wiring |
| 19 | `LlamaIndex-agent.md` | `llama_index_agent/` | `FunctionAgent` prebuilt + manual `Workflow` version |
| 20 | `LlamaIndex-router.md` | `llama_index_router/` | `RouterQueryEngine`, `SubQuestionQueryEngine` |
| 21 | `LlamaIndex-extract.md` | `llama_index_extract/` | `llm.structured_predict(SchemaClass, prompt)` for one-shot and batch extraction |
| 22 | `LlamaIndex-deploy.md` | `llama_index_deploy/` | `llama-deploy` local three-process setup (control plane + workflow service + client) with Redis |

### Appendices + wrap-up

- `AppendixA-models.md` — evergreen discussion of small/medium/large local models, when hosted is worth the cost, roles a model can play
- `AppendixB-eval.md` — LLM-as-judge, reference-based `assert` tests, open-source alternatives (Langfuse, MLflow, Phoenix)
- `AppendixC-observability.md` — `set_debug(True)`, `.stream()`, OpenInference + Phoenix; honest comparison with LangSmith
- `AppendixD-vps.md` — $5/month VPS setup: Ubuntu + `uv` + Caddy + systemd, cost math
- `WrapUp.md` — rewritten around the OSS-first thesis; explicit acknowledgement of what the book skipped

### Source-code layout convention

Each chapter directory contains:
- `pyproject.toml` with pinned dependencies (uv-managed)
- `README.md` with setup + script descriptions
- Numbered scripts (`01_*.py`, `02_*.py`, ...) each demonstrating one concept
- Shared helper modules prefixed with `_` (e.g. `_tools.py`, `_supervisor.py`)

## Files preserved, not deleted

Nothing has been thrown away. Old-edition files live in:

- `manuscript/DEPRECATED/` — `AdvancedExperiments.md`, `AgentsLangGraph.md`, `Llamacpp.md`, `MiniChain.md`, `Preface.old.md`, `Recipes.md`, `SearchKG.md`
- `source-code/DEPRECATED/` — `kg_search/`, `langchain_dbpedia_agent/`, `langchain_getting_started/` (old), `llama-index_case_study/`, `llm_enhanced_search_ddg_ollama/`, `sqlite/` (old)

Old chapters still in `manuscript/` that are in `Book.txt` but **not yet modernized** — these are placeholders and will need attention in a follow-up pass:

- `LLM-overview.md`
- `Ollama.md`
- `Extraction.md`
- `Sumarization.md`
- `StructuredData.md`
- `GoogleDrive.md`
- `HuggingFace.md`
- `OtherUsefulLibraries.md`

Also present but not part of the book build: `UPDATES.md`.

## Book.txt final order

```
Preface, TheStack, LLM-overview,
LangChain-overview, RAG, Ollama, Extraction, Sumarization, StructuredData,
LangGraph-fundamentals, Agents, LangGraph-durable, LangGraph-hitl, LangGraph-supervisor,
SqLite, KG, llm_search,
LlamaIndex_case_study, LlamaIndex-ingest, LlamaIndex-indices, LlamaIndex-rerank,
LlamaIndex-workflows, LlamaIndex-agent, LlamaIndex-router, LlamaIndex-extract, LlamaIndex-deploy,
GoogleDrive, HuggingFace, OtherUsefulLibraries,
AppendixA-models, AppendixB-eval, AppendixC-observability, AppendixD-vps,
WrapUp
```

## Cross-cutting rewrite rules applied

- **No LangSmith / LangGraph Cloud / LlamaCloud** anywhere. If a topic would require them, it doesn't go in.
- **No paid third-party APIs required.** DuckDuckGo, Brave Search free tier, public SPARQL endpoints are in; SerpAPI, Google KG, Zapier are out.
- **Every code example runs on the reader's laptop** with `uv run examples/<name>.py`.
- **Pin versions** in every `pyproject.toml`.
- **Prefer `uv` over `pip`.**
- **No deprecated imports:** `langchain.llms.*`, `LLMChain`, `initialize_agent`, `GPTSimpleVectorIndex`, `GPTTreeIndex`, `download_loader`, `SQLDatabaseChain` — all gone.
- **Author's voice preserved** — first-person moments where natural, especially in KG and Google Drive-style chapters.

## Things to verify during your test-run pass

I did not run any of the code (author will test and report back). Most likely issues to watch for:

- **LangChain 1.0 import paths** — a few chapters (Ch 4's `MultiQueryRetriever`, Ch 4's `CrossEncoderReranker`) use import paths that were stable in 0.3 but 1.0 has reorganized. If imports fail, look in `langchain_community` first.
- **`langgraph-checkpoint-sqlite` version pin** — the package has had major version bumps; `>=2.0,<3` is my guess for mid-2026.
- **`SqliteSaver.from_conn_string(...)` as context manager** — API has been stable but syntax matters (`with` block required).
- **`langgraph.types.interrupt` / `Command`** — Chapter 9 uses these; earlier LangGraph versions expose them at different paths.
- **LlamaIndex `Settings` API** — Chapter 14+ assumes the current Settings-based config; the older `ServiceContext` pattern is fully removed in 0.14.
- **LlamaIndex Workflows API** (Chapters 18-19) — `FunctionAgent`, `Context.get/set`, `Workflow(timeout=...)`. Class names may have moved between minor releases.
- **`llama-deploy` API** (Chapter 22) — this package has had several rewrites; the `ControlPlaneConfig` / `WorkflowServiceConfig` / `Client.core.sessions.create()` shape reflects the 0.7 line as I understood it.
- **`QueryFusionRetriever` in Ch 16** — `BM25Retriever.from_defaults(nodes=...)` vs `docstore=...` differs between versions.
- **`SubQuestionQueryEngine` in Ch 20** — some versions require explicit `question_gen` argument.
- **`llm.structured_predict` in Ch 21** — the exact method name (`structured_predict` vs `apredict_and_call`) has varied.

Model behavior to watch for:

- Tool-calling reliability on `qwen3:8b` — the book's default. Silent failures (model narrates instead of calling) are the biggest gotcha; documented in Chapters 7 and 10.
- `.with_structured_output()` reliability on small models — Chapter 10's supervisor routing depends on this.
- DuckDuckGo occasional rate limiting — affects Chapters 7, 10, 13, 19.

## Where to pick up

Roughly in priority order for the next work session:

1. **Test-run pass.** Author plans to `uv sync` + `uv run` each chapter's scripts and report issues back. Fix as they come in.
2. **Modernize the still-old chapters** that are in `Book.txt` but haven't been touched: `LLM-overview.md`, `Ollama.md`, `Extraction.md`, `Sumarization.md`, `StructuredData.md`, `GoogleDrive.md`, `HuggingFace.md`, `OtherUsefulLibraries.md`. Guidance for each is in `PLAN.md` §3.
3. **Decide on Chapters 2, 3, 5 slots** — the plan lists them (prompts deep-dive, embeddings deep-dive, tool-calling deep-dive) but I folded most of the material into Chapters 1 and 4. Options:
   - Leave folded (current state, works fine)
   - Extract into standalone chapters
   - Renumber the book to reflect what's actually there
4. **Front-matter polish** — the new Preface and TheStack chapter were the first things written; may want a fresh pass after all the other chapters are in place.
5. **Cross-references and forward pointers** — many chapters end with "Chapter X covers..." pointers. Some point at chapter numbers that may need updating if the TOC shifts.
6. **Book cover and marketing copy** — separate from the manuscript work.

## What was not written

Called out in the WrapUp chapter but worth restating:

- Advanced retrieval patterns beyond hybrid + reranking (HyDE, graph-augmented, filter-by-metadata)
- Fine-tuning local models
- Multi-modal (vision, audio, video)
- A deeper eval/observability chapter (only touched in appendices)

If author wants any of these as follow-up chapters or a follow-up book, that's a separate scope.

## Sanity-check numbers

- **New manuscript files this session:** 24 (Preface, TheStack, 17 new chapters, 4 appendices, WrapUp)
- **New source-code directories:** 15 (langchain_getting_started + rag_langchain + 5 langgraph dirs + sql_agent + kg_agent + local_search + 8 llama_index dirs)
- **Old files preserved to DEPRECATED:** 7 manuscript files, 6 source directories
- **Total lines of manuscript written:** roughly 5,000+ across all new chapters
- **Total lines of example code written:** roughly 2,500+ across all new source dirs

## Files that don't fit any category

- `PLAN.md` (repo root) — the reference document the whole rewrite worked from. If a future decision seems unclear, check what §3, §4, §5, or §7 say.
- `NOTES_from_Claude.md` (this file) — session summary for the author.
