# Choosing an index type

`VectorStoreIndex` is the right answer roughly 90% of the time. This chapter is about the other 10%, and about the one hybrid pattern that pushes the 90% number higher when you need to.

LlamaIndex has always shipped several index types. In the previous edition I only mentioned them in passing because most were rarely worth the extra complexity. In 2026 that is still mostly true — but two of the alternatives (`SummaryIndex` and the BM25 + vector hybrid) have concrete use cases where they clearly outperform a naked `VectorStoreIndex`. The third one covered here (`SimpleKeywordTableIndex`) is worth knowing about even if you never ship it, because it makes the "why do we need embeddings at all" question concrete.

All three scripts live in `source-code/llama_index_indices/` and read from `source-code/data/`. Setup:

```console
$ cd source-code/llama_index_indices
$ uv sync
$ ollama pull qwen3.5:4b
```

## `SummaryIndex` — every query touches every Node

`VectorStoreIndex` is optimized for "find the top-k Nodes most similar to this query." That is the wrong shape for questions like "summarize the whole corpus," "what themes recur across these documents," or "give me an overview of everything in this folder." Those queries want the LLM to see every Node, in order, and reason across all of them.

`SummaryIndex` (the current name for what used to be called `ListIndex`) does exactly this. It maintains the Nodes in insertion order and, at query time, feeds all of them plus the query to the LLM.

`01_summary_index.py`:

```python
from llama_index.core import Settings, SimpleDirectoryReader, SummaryIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=180.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query(
    "Give me a one-paragraph overview of the topics covered by these documents."
)
print(response)
```

Representative output:

```console
The documents cover four distinct topics: chemistry (the scientific
study of matter and its transformations), economics (schools of thought
and their approaches to markets and government), health (factors
affecting human well-being and disease prevention), and sports
(the definition of athletic activity and its cultural and physical
dimensions). Together they span the natural sciences, social sciences,
health sciences, and cultural studies.
```

`VectorStoreIndex.as_query_engine().query(...)` on the same question would pick the top few Nodes by embedding similarity and miss most of the corpus, because "give me an overview" is a semantic query about *the corpus itself*, not about any specific concept the corpus contains.

The tradeoff is obvious: `SummaryIndex` is O(n) per query, where n is the number of Nodes. Fine for corpora with dozens of Nodes, painful with thousands, unusable with millions. In practice I use it either for small curated collections (a research folder, a single project's docs, a book's chapters) or as a downstream tool for questions where the router has already narrowed the corpus to a small subset.

## `SimpleKeywordTableIndex` — retrieval by exact keyword match

Embeddings are not always the right retrieval mechanism. If your corpus is dominated by proper nouns, product identifiers, function names, drug names, or legal citations, semantic similarity is often *worse* than exact-string matching. The classic case: your query mentions "GPT-4o" and the relevant document is one of the few that also mentions "GPT-4o" verbatim. A dense retriever will happily return semantically-adjacent documents about "large language models," "OpenAI," or "GPT-4," pushing your actual match down or off the list.

`SimpleKeywordTableIndex` builds an inverted index of the corpus using regex keyword extraction. No LLM, no embedding model, no ML dependency at all — just Python string processing. At query time, it extracts keywords from the query the same way and retrieves Nodes whose keyword sets overlap.

`02_keyword_table.py`:

```python
from llama_index.core import SimpleDirectoryReader, SimpleKeywordTableIndex

documents = SimpleDirectoryReader("../data").load_data()
index = SimpleKeywordTableIndex.from_documents(documents)

retriever = index.as_retriever()

for query in [
    "What is chemistry?",
    "Tell me about the Austrian School",
    "What is a laboratory?",
]:
    print(f"QUERY: {query}")
    nodes = retriever.retrieve(query)
    for i, n in enumerate(nodes, 1):
        src = n.node.metadata.get("file_name", "?")
        print(f"  [{i}] source={src}")
    print()
```

Where this actually earns its keep in real projects:

- **Factoid corpora full of specific terms.** Product catalogs, drug databases, legal statute collections, API references.
- **Environments where you cannot install ML dependencies.** Air-gapped systems, minimal Docker images, edge devices.
- **As a cheap first-pass filter** before a more expensive stage — pull a wide keyword-based set of candidates, then rerank them with an LLM or a cross-encoder.

Where it fails: any query where the important word does not appear literally in the relevant Node. Paraphrases are invisible. That is why the hybrid pattern in the next section usually beats either dense-only or sparse-only in isolation.

The `KeywordTableIndex` variant (no "Simple") uses an LLM to extract richer keyword lists including synonyms and related concepts. More accurate for hard queries, slower to build, needs a model. In practice I have not shipped the LLM version in a while — the LLM cost at ingestion time is high enough that if I can afford it, I would rather spend it on a reranker at query time.

## `QueryFusionRetriever` — the practical hybrid pattern

The one hybrid pattern I use in almost every real LlamaIndex project. Run both a dense (embedding) retriever and a sparse (BM25) retriever over the same corpus, then merge their ranked lists with reciprocal rank fusion.

`03_hybrid_fusion.py`:

```python
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()

vector_index = VectorStoreIndex.from_documents(documents)
dense = vector_index.as_retriever(similarity_top_k=3)

sparse = BM25Retriever.from_defaults(
    nodes=list(vector_index.docstore.docs.values()),
    similarity_top_k=3,
)

fusion = QueryFusionRetriever(
    retrievers=[dense, sparse],
    similarity_top_k=3,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=False,
)

for query in [
    "How does body chemistry affect exercise?",
    "Austrian School",
]:
    print(f"QUERY: {query}")
    nodes = fusion.retrieve(query)
    for i, n in enumerate(nodes, 1):
        src = n.node.metadata.get("file_name", "?")
        print(f"  [{i}] score={n.score:.3f}  source={src}")
    print()
```

The BM25 retriever wants a list of Nodes, not a full index — hence the slightly odd `list(vector_index.docstore.docs.values())` pattern. In a production setup where you have already run an ingestion pipeline, you would pass the Node list directly.

Two `QueryFusionRetriever` parameters worth knowing about:

- **`mode="reciprocal_rerank"`** — reciprocal rank fusion. Documents ranked highly by either retriever get a good final score; documents ranked highly by both get an excellent final score. This is the default and almost always what you want.
- **`num_queries=1`** — the user's query is used as-is. If you set this to a higher number, the fusion retriever asks the LLM to generate that many query rewrites and runs *each* retriever on *each* rewrite. Similar to Chapter 4's `MultiQueryRetriever` from LangChain but built into the fusion retriever itself. Costs LLM calls, buys recall.

On the two-query test in the script, the fusion retriever handles both cleanly: the "body chemistry / exercise" query gets a strong dense-retrieval boost, and the "Austrian School" query (a proper noun that appears verbatim in `economics.txt`) gets a strong BM25 boost. Neither retriever on its own would rank both queries as well as the fusion does.

## Decision tree

A concrete decision procedure I use in my own projects:

1. **Start with `VectorStoreIndex`** and see how retrieval quality holds up on real user queries. This is the right answer most of the time.
2. **If overview-style queries do badly**, add a `SummaryIndex` over the same corpus and route those queries to it via a router (Chapter 20).
3. **If exact-term queries do badly** (product names, function names, proper nouns), swap the `VectorStoreIndex` retriever for a `QueryFusionRetriever` over both dense and BM25. This is close to free — no extra models, no meaningful latency increase — and it fixes a wide class of retrieval failures.
4. **If ingestion-time budget is nonexistent** and you cannot install ML dependencies at all, use `SimpleKeywordTableIndex`. Ship it, measure, revisit.

Reach for `TreeIndex` (not covered here) if you have thousands of Nodes and need a hierarchical retrieval that traverses top-down. Reach for `KnowledgeGraphIndex` if you want the framework to extract entities and relationships for you and query them as a graph — but at that point you may find Chapter 12's DBpedia/Wikidata SPARQL agents a cleaner fit.

## What we covered

- `VectorStoreIndex` is the default, but not the only tool.
- `SummaryIndex` handles overview / cross-corpus queries by touching every Node — O(n) per query, worth it when the shape of the question demands it.
- `SimpleKeywordTableIndex` gives you keyword-based retrieval with zero ML dependencies — a fallback and a first-pass filter.
- `QueryFusionRetriever` fuses BM25 and dense retrieval with reciprocal rank fusion. This is the hybrid pattern most projects should use once a vanilla vector retriever starts missing obvious matches.

Chapter 17 covers reranking — the last piece of the retrieval-quality puzzle before we move on to LlamaIndex's Workflows API in Chapter 18.
