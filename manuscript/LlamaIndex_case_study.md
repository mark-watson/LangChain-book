# LlamaIndex

Dear reader, welcome to Part II of this book. Everything so far has been about LangChain and LangGraph. The rest of the book covers LlamaIndex, which occupies overlapping-but-distinct territory. LlamaIndex has always been strongest at retrieval, indexing, and document-centric applications. LangChain has always been strongest at orchestration, chains, and general-purpose agent plumbing. In 2026 both frameworks have widened their scope enough to cover most of the same problems, but the shapes of their APIs — and the shapes of the applications that fall out most naturally from those APIs — remain different.

There is nothing to unlearn from Part I. Prompts are still prompts, embeddings are still embeddings, RAG is still RAG. What changes is the vocabulary and the choreography.

## What changed since the previous edition

If you have used LlamaIndex before and stopped a year or two ago, be prepared for most of the class names you remember to have moved or disappeared. As of 2026:

- **`GPTSimpleVectorIndex`, `GPTTreeIndex`, `GPTKeywordTableIndex`** — all removed. The generic `VectorStoreIndex` (and a handful of other index types) took their place. The `GPT` prefix is gone; embeddings are pluggable and no longer tied to OpenAI.
- **`LLMPredictor`, `PromptHelper`, `ServiceContext`** — all removed. Configuration lives in a global `Settings` object, or is passed directly to the components that need it.
- **`download_loader`** — removed. Loaders are now regular pip-installable packages under `llama-index-readers-*`.
- **The one-package install** has become a monorepo of ~300 packages. `pip install llama-index` still works but pulls in everything; the recommended pattern is to install just `llama-index-core` plus the specific integration packages you need.

The mental model, though, is the same as it always was: **Documents** get chunked into **Nodes**, Nodes get organized into an **Index**, and an Index exposes a **QueryEngine** (or a lower-level **Retriever**) for answering questions. Get comfortable with those four concepts and everything else in Part II is a variation on the theme.

## The four primitives

**Document.** A chunk of text plus metadata. `Document(text="...", metadata={"source": "..."})`. In practice you rarely construct them by hand — a reader like `SimpleDirectoryReader` produces them from files.

**Node.** A Document after chunking. The atomic unit the retriever returns. In vanilla setups, one Document becomes one Node; with a `SentenceSplitter` transformation in your ingestion pipeline, one Document becomes many Nodes.

**Index.** A queryable data structure built over Nodes. `VectorStoreIndex` is the one you will use 90% of the time — it stores each Node's embedding and does cosine-similarity lookup at query time. Others include `SummaryIndex`, `KeywordTableIndex`, and `TreeIndex`; we will look at when to reach for each in Chapter 13.

**QueryEngine / Retriever.** The two ways to *use* an Index. `.as_query_engine()` gives you `engine.query(text)` which returns a synthesized answer from an LLM after retrieval. `.as_retriever()` gives you `retriever.retrieve(text)` which returns the raw Nodes without the LLM step. Both are useful; you pick based on whether you want an answer or the ingredients for one.

## Setup

All of Part II's examples share the same base install. For this chapter's directory:

```console
$ cd source-code/llama_index_intro
$ uv sync
$ ollama pull qwen3.5:4b
```

The four scripts in this chapter share the four-file corpus in `source-code/data/` — the same one Chapter 2 used. Reusing the corpus lets you compare LlamaIndex's behavior directly against Chapter 2's LangChain RAG chapter.

## Your first LlamaIndex program

The smallest useful program that exercises Documents → Index → QueryEngine end-to-end. `01_hello_llamaindex.py`:

```python
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
print(f"Loaded {len(documents)} documents from ../data")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is the Austrian School of Economics?")
print(f"\nAnswer:\n{response}")
```

Four things to notice.

**`Settings` is the LlamaIndex global config.** Set `Settings.llm` and `Settings.embed_model` at the top of your script, and every downstream component (`VectorStoreIndex`, query engines, retrievers, extractors) picks them up automatically. You can also pass `llm=` and `embed_model=` directly to individual constructors when you want overrides, but the global default handles most cases.

**Provider packages match Part I's convention.** `Ollama` from `llama_index.llms.ollama`, `HuggingFaceEmbedding` from `llama_index.embeddings.huggingface`. Every LlamaIndex integration lives in its own package under the `llama-index-*` prefix, and you install only the ones you use.

**`SimpleDirectoryReader` is the reader you will use most often.** It scans a directory, dispatches each file to the right parser based on extension (.txt, .md, .pdf, .docx, .csv, .json), and returns a list of Documents. It has options for recursion, exclusion patterns, and metadata extraction — but the default of "everything in this directory as text" is what you want most of the time.

**`.query()` returns a Response object**, which prints as its text but also carries `.source_nodes` (the Nodes the retriever picked) and `.metadata`. In production you often want the source nodes for citations or debugging; a plain `print(response)` is fine for exploration.

Expected output:

```console
$ uv run 01_hello_llamaindex.py
Loaded 4 documents from ../data

Answer:
The Austrian School of Economics is a school of economic thought that
emphasizes the spontaneous organizing power of the price mechanism,
advocates a laissez-faire approach to the economy, and holds that commercial
transactions should be subject to minimal government intervention.
```

## Swapping in a hosted model

The provider-swap in LlamaIndex is one line: swap what `Settings.llm` points at. `02_hosted_swap.py`:

```python
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

Everything downstream — documents, index, query engine, response — is identical to script 1. Notice that even in the "hosted" script, the embedding model stays local. Embeddings are cheap to run on your own hardware, they run in milliseconds, and there is no reason to pay a provider for something an open model can do fine. I keep hosted embedding providers in the same "avoid unless you have a specific reason" bucket as LlamaCloud.

## Persist an index, reload it later

Real projects almost never build an index every time they answer a query. Building is expensive (embed every document, potentially thousands of API calls or minutes of local GPU); querying is cheap (embed one query, do a nearest-neighbor lookup, one LLM call). The standard pattern is: build once, persist to disk, reload in the query script.

`03_persist_and_reload.py`:

```python
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

PERSIST_DIR = Path(__file__).parent / "storage"

# Build and persist.
documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=str(PERSIST_DIR))

# Later — potentially in a different script — reload and query.
storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
reloaded = load_index_from_storage(storage_context)

response = reloaded.as_query_engine().query("What is the definition of sport?")
print(response)
```

`index.storage_context.persist(persist_dir)` writes three JSON files into the target directory: `docstore.json` (the raw Documents and Nodes), `index_store.json` (the index metadata), and `vector_store.json` (the embeddings). For small corpora these are a few dozen KB total; for large corpora with millions of vectors you will want a real vector store (Chroma, Qdrant, LanceDB) instead of the default in-memory one — LlamaIndex integrates with all of them via `llama-index-vector-stores-*` packages.

`load_index_from_storage(StorageContext.from_defaults(persist_dir=...))` is the reverse trip. The returned object behaves exactly like the freshly-built one — same `.as_query_engine()`, same `.as_retriever()`.

## Retrievers, without the LLM

`.as_query_engine().query(...)` bundles two steps: retrieve the top-k Nodes for the query, then send them to the LLM along with the query for synthesis. Sometimes you only want the first step. `04_retriever_only.py`:

```python
retriever = index.as_retriever(similarity_top_k=3)

nodes = retriever.retrieve("What is the Austrian School of Economics?")

for i, node_with_score in enumerate(nodes, 1):
    node = node_with_score.node
    source = node.metadata.get("file_name", "?")
    snippet = node.text[:120].replace("\n", " ")
    print(f"[{i}] score={node_with_score.score:.3f}  source={source}")
    print(f"    {snippet}...\n")
```

`.retrieve()` returns a list of `NodeWithScore` objects. `.score` is the cosine similarity, `.node.text` is the chunk text, `.node.metadata` is the metadata dict (which for `SimpleDirectoryReader`-loaded documents includes `file_name`, `file_path`, and a few others).

I reach for `.as_retriever()` whenever LlamaIndex is going to be one step in a larger pipeline whose synthesis step is not a straightforward LLM call — feeding the nodes into a LangGraph workflow, or into a custom multi-step prompt, or into a downstream reranker.

Representative output:

```console
$ uv run 04_retriever_only.py
[1] score=0.612  source=economics.txt
    The Austrian School of Economics is a school of economic thought that ...

[2] score=0.318  source=health.txt
    Human health depends on many factors, including body chemistry and ...

[3] score=0.281  source=sports.txt
    Sport is generally recognised as activities based in physical athleticism ...
```

The score gap tells the story: only the first result is meaningfully relevant. A production retriever would filter on a minimum score, apply a reranker (Chapter 14 covers this), or expand the query — techniques that translate one-to-one from the Chapter 2 LangChain patterns.

## What we covered

Four primitives, four scripts:

1. **Documents → Index → QueryEngine** with `Settings.llm` and `Settings.embed_model` for provider config.
2. **Provider swap** as a one-line change to `Settings.llm`.
3. **`.persist()` / `load_index_from_storage`** for saving and reloading indices.
4. **`.as_retriever()`** to get raw Nodes without an LLM synthesis step.

Everything in the rest of Part II is combinations and elaborations of those four. Chapter 12 goes deeper on document loading and local embedding models. Chapter 13 covers the different index types and when to reach for each. Chapter 14 adds reranking. Chapters 15-16 introduce the Workflows API — LlamaIndex's answer to LangGraph — and use it to build a real agent. Chapters 17-19 cover multi-index query pipelines, structured extraction with `PydanticProgram`, and serving a workflow with FastAPI.
