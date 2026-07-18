# Local documents and local embeddings

Chapter 14 covered the LlamaIndex primitives in the abstract. This chapter is about the two concrete, everyday operations you will do more than any others: getting your data into the framework as `Document` objects, and picking an embedding model to convert them into vectors. Both operations run entirely on your laptop.

Everything here lives in `source-code/llama_index_ingest/` and shares the four-file corpus in `source-code/data/` (chemistry, economics, health, sports) that the previous two chapters used.

## `SimpleDirectoryReader` deep dive

Chapter 14 showed `SimpleDirectoryReader("../data").load_data()` and moved on. It is worth going a little deeper, because for 90% of "load some files from disk" use cases, this single class is the right answer.

`01_directory_loading.py`:

```python
from pathlib import Path

from llama_index.core import SimpleDirectoryReader


def add_custom_metadata(file_path: str) -> dict:
    p = Path(file_path)
    return {
        "topic": p.stem,          # "sports.txt" -> topic="sports"
        "collection": "book_data",
    }


reader = SimpleDirectoryReader(
    input_dir="../data",
    recursive=True,
    exclude=["*.png", "*.jpg", ".DS_Store"],
    file_metadata=add_custom_metadata,
)

documents = reader.load_data()

print(f"Loaded {len(documents)} documents\n")
for d in documents:
    meta = d.metadata
    print(f"file_name : {meta.get('file_name')}")
    print(f"topic     : {meta.get('topic')}")
    print(f"file_size : {meta.get('file_size')} bytes")
    print(f"first 80c : {d.text[:80]!r}")
    print()
```

Four options worth internalizing.

**`input_dir`.** A directory path. `SimpleDirectoryReader` walks it and processes each file it recognizes. For a fixed list of files instead of a directory, use `input_files=[...]`.

**`recursive=True`.** Include subdirectories. Off by default; you almost always want it on for real corpora.

**`exclude=[...]`.** Glob patterns to skip. This is where you filter out `.DS_Store`, `.git`, images, source code, or anything else you don't want indexed.

**`file_metadata=callable`.** A function that runs once per file and returns a dict. Whatever you return here gets merged into the Document's metadata alongside the fields the reader adds automatically (`file_name`, `file_path`, `file_type`, `file_size`, `creation_date`, `last_modified_date`). Use it for anything not derivable from the file path — a category, a project ID, an author.

The metadata a Document carries is not decorative. It flows through chunking so every resulting Node has the same metadata, and it survives all the way to retrieval time as `node.metadata`. That means you can use it for filtered queries ("only search in topic='sports' notes"), for citations ("this answer came from `sports.txt`"), or for post-retrieval routing in a supervisor graph.

Beyond `.txt`, `SimpleDirectoryReader` also handles Markdown, CSV, JSON, PDF, DOCX, PowerPoint, images (with OCR), and audio — but each format needs an extra reader package installed (`llama-index-readers-file` covers most of them). The book stays with `.txt` throughout because it keeps the setup light; a real project just does `uv add llama-index-readers-file` and points the reader at whatever mix of formats you have.

## Building Documents by hand

When your source is anything other than a filesystem — a REST API, a database query, a Kafka topic, a directory listing that lives inside a zip file — you construct `Document` objects yourself. There is no ceremony to this; the rest of LlamaIndex has no idea (and no need to know) where the Documents came from.

`02_custom_documents.py`:

```python
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Imagine these came from a REST API or a database query.
records = [
    {
        "id": "note-001",
        "author": "Mark",
        "title": "Sedona hiking",
        "body": "Cathedral Rock and Bell Rock are the most popular hikes near Sedona, Arizona.",
    },
    {
        "id": "note-002",
        "author": "Carol",
        "title": "Sedona food",
        "body": "The Mexican restaurants in Sedona serve especially good chile relleno.",
    },
    {
        "id": "note-003",
        "author": "Mark",
        "title": "Prescott hiking",
        "body": "Thumb Butte in Prescott, Arizona is a short but steep hike with a great summit view.",
    },
]

documents = [
    Document(
        text=r["body"],
        metadata={"id": r["id"], "author": r["author"], "title": r["title"]},
    )
    for r in records
]

index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=2)

nodes = retriever.retrieve("What hikes are near Sedona?")

for i, n in enumerate(nodes, 1):
    print(f"[{i}] score={n.score:.3f}  author={n.node.metadata['author']}  title={n.node.metadata['title']!r}")
    print(f"    {n.node.text}\n")
```

Every custom `Document` carries whatever metadata you want. In the example above I put `id`, `author`, and `title` on each one; a real project might also carry `created_at`, `department`, `visibility`, or anything else the app needs downstream. As long as the values are JSON-serializable, LlamaIndex is happy.

Representative output:

```console
[1] score=0.671  author=Mark  title='Sedona hiking'
    Cathedral Rock and Bell Rock are the most popular hikes near Sedona, Arizona.

[2] score=0.412  author=Mark  title='Prescott hiking'
    Thumb Butte in Prescott, Arizona is a short but steep hike with a great summit view.
```

The retriever picked the Sedona hiking record first and the Prescott one second (Prescott is a similar concept but a different city). The Sedona food record was correctly filtered out by the embedding similarity score.

## Comparing embedding models

The embedding model is arguably the most consequential choice in a RAG pipeline. It determines which Nodes come back for a given query and in what order. It is also the choice most people default on and never revisit. This script makes the difference visible.

`03_embedding_comparison.py`:

```python
MODELS = [
    ("BGE-small (BAAI/bge-small-en-v1.5)", "BAAI/bge-small-en-v1.5"),
    ("MiniLM (sentence-transformers/all-MiniLM-L6-v2)", "sentence-transformers/all-MiniLM-L6-v2"),
]

QUERY = "How does the body process energy during exercise?"

documents = SimpleDirectoryReader("../data").load_data()

for label, model_name in MODELS:
    print(f"=== {label} ===")
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=3)

    nodes = retriever.retrieve(QUERY)
    for i, n in enumerate(nodes, 1):
        source = n.node.metadata.get("file_name", "?")
        print(f"  [{i}] score={n.score:.3f}  source={source}")
    print()
```

Both are small (under 150 MB), both are free, both run locally in tens of milliseconds. On the four-file corpus, on the query "How does the body process energy during exercise?", they usually pick the same top-1 (`health.txt`) but disagree on the ranking of the remaining files and on absolute score.

A rough taxonomy of the local embedding models worth knowing about in 2026:

- **`BAAI/bge-small-en-v1.5`** (~130 MB) — my default. Fast, high quality for its size, English-only.
- **`BAAI/bge-base-en-v1.5`** (~440 MB) — a step up in quality. Roughly 3× slower to embed but the ranking is noticeably tighter on hard queries.
- **`sentence-transformers/all-MiniLM-L6-v2`** (~90 MB) — a classic. Older than the BGE models, slightly less accurate on modern benchmarks, still perfectly usable.
- **`nomic-ai/nomic-embed-text-v1.5`** (~550 MB) — trained on longer contexts (up to 8k tokens), useful when your Nodes are large and you don't want to chunk aggressively. Requires `trust_remote_code=True`.
- **`BAAI/bge-m3`** (~2.3 GB) — multilingual, multi-vector, top-of-the-line quality. Overkill for most projects; reach for it when you actually need any of those things.

I default to BGE-small for prototypes and move up to BGE-base when I see the small model missing obvious matches. Below that threshold, effort is better spent on retrieval quality (Chapter 17's reranker) than on a bigger embedding model.

## Chunking with the ingestion pipeline

The four text files in `../data/` are each small enough to fit in one Node. Real corpora rarely look like that. A 40-page PDF wants to be broken into ~50 smaller Nodes so retrieval returns tight, focused context instead of the whole document at once.

The mechanism is `IngestionPipeline` with a `SentenceSplitter`. `04_ingestion_pipeline.py`:

```python
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
print(f"Loaded {len(documents)} documents")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=200, chunk_overlap=20),
    ]
)

nodes = pipeline.run(documents=documents)
print(f"Produced {len(nodes)} nodes after splitting\n")

# Nodes can be handed directly to an index instead of Documents.
index = VectorStoreIndex(nodes)
```

`SentenceSplitter(chunk_size=200, chunk_overlap=20)` breaks documents at sentence boundaries into chunks of at most 200 tokens each, with a 20-token overlap between consecutive chunks. The overlap is there to prevent chunks from splitting mid-thought — if a key sentence lands right at a chunk boundary, both chunks will contain it. For the small corpus here, `chunk_size=200` produces about 10 Nodes from the 4 Documents.

Standard chunk sizes in my own projects:

- **200-300 tokens** for narrow, precise Q&A. Small chunks mean the retriever picks focused context, which small local LLMs handle better.
- **500-1000 tokens** for anything summarization-adjacent. Bigger chunks mean the model sees more surrounding context and produces more coherent output.
- **Overlap** of 10-20% of chunk size. Below 10% you get chunk-boundary artifacts; above 20% is diminishing returns.

`IngestionPipeline` accepts a list of transformations, and the ones useful beyond `SentenceSplitter` include:

- **`TitleExtractor`** — asks an LLM to generate a title for each chunk and adds it as metadata. Improves retrieval on queries that mention high-level topics.
- **`KeywordExtractor`** — extracts N keywords per chunk. Cheap way to give queries a keyword-matching path even in a vector store.
- **`SummaryExtractor`** — adds a per-chunk summary. Useful for retrieval interfaces that show a preview of each result.

Each of these costs one LLM call per chunk at ingestion time, so on a real corpus you would only pick the transformations that actually help your queries.

## What we covered

- `SimpleDirectoryReader` with `recursive`, `exclude`, and a `file_metadata` callback covers most on-disk loading.
- Building `Document` objects by hand covers everything else: APIs, databases, custom sources. Metadata on the Document flows all the way through to `node.metadata` at retrieval time.
- Local embedding models are small, fast, and free. BGE-small is my default; BGE-base if I need more quality; MiniLM as a classic backup.
- `IngestionPipeline` with `SentenceSplitter` is the standard way to chunk. Chunk sizes in the 200-1000 token range with 10-20% overlap cover most needs.

Chapter 16 covers the other end of the ingestion story: given a corpus of Nodes, which index type should you actually build?
