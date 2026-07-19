# Deprecated chapters and code

Material removed from the current edition of the book, kept here rather than deleted. None of it is maintained or guaranteed to run against current package versions. The corresponding old manuscript chapters live in `manuscript/DEPRECATED/`.

| Source directory | Old manuscript chapter | Why it was dropped |
|---|---|---|
| `langchain_getting_started/` | — (superseded) | Pre-1.0 LangChain getting-started examples; replaced by the current `source-code/langchain_getting_started/`. |
| `sqlite/` | — (superseded) | Used the deprecated `langchain_experimental.sql` / `SQLDatabaseChain`; replaced by the current `source-code/sql_agent/`. |
| `llama-index_case_study/` | — (superseded) | Pre-`llama-index-core` API (`GPTSimpleVectorIndex` etc.); replaced by the current `source-code/llama_index_intro/`. |
| `llm_enhanced_search_ddg_ollama/` | — (superseded) | Early DuckDuckGo + Ollama search prototype; replaced by the current `source-code/local_search/`. |
| `langchain_dbpedia_agent/` | — (superseded) | Early DBpedia-only agent; replaced by the current `source-code/kg_agent/`, which also covers Wikidata. |
| `kg_search/` | `SearchKG.md` | Used Google's Knowledge Graph API, a paid Google Cloud service. Public DBpedia/Wikidata via SPARQL cover the same use case for free — see `source-code/kg_agent/`. |

Old manuscript chapters with no surviving source directory (the code either never had its own folder or was cut along with the chapter):

- `AdvancedExperiments.md` — three `TBD` placeholder sub-chapters, never written.
- `AgentsLangGraph.md` — an early, pre-LangGraph-1.0 agent experiment; superseded by the current `Agents.md` / `source-code/langgraph_react_agent/`.
- `Llamacpp.md` — running local models via `llama.cpp`; Ollama covers this need for the book's target reader now.
- `MiniChain.md` — a chapter on the (now largely inactive) MiniChain library.
- `Preface.old.md` — the previous edition's preface.
- `Recipes.md` — a recipe-generation example built around Zapier, a proprietary integration; the source code lives on, unreferenced, in `source-code/cooking_recipes/`.
- `StructuredData.md` — folded into `Extraction.md` per the current edition's plan; its own code example was fictional and never had a real source directory.
