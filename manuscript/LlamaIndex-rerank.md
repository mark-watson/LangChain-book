# RAG with reranking

Chapter "RAG patterns with LangChain" made the same argument I am about to make here, and it applies verbatim in LlamaIndex: cosine similarity over sentence embeddings is fast but coarse; a cross-encoder is slow but much more accurate; the trick is to retrieve wide with the cheap tool and rerank narrow with the expensive one.

In LlamaIndex the mechanism is a **node postprocessor**. You pass one to `.as_query_engine(node_postprocessors=[...])` and it runs between the retriever's `.retrieve()` call and the LLM synthesis step. The framework ships several (`SimilarityPostprocessor` for score thresholding, `MetadataReplacementPostProcessor` for retrieved-content rewriting, `LongContextReorder` for reordering), but the one that matters most for retrieval quality is `SentenceTransformerRerank`.

Everything lives in `source-code/llama_index_rerank/` and reads from `source-code/data/`. Setup:

```console
$ cd source-code/llama_index_rerank
$ uv sync
$ ollama pull qwen3.5:4b
```

The BGE reranker model (~275 MB) downloads on first run.

## Baseline: no reranker

`01_no_reranker.py`:

```python
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=180.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=5)

query = "How does body chemistry affect exercise performance?"
response = query_engine.query(query)

print("=== Retrieved nodes ===")
for i, sn in enumerate(response.source_nodes, 1):
    src = sn.node.metadata.get("file_name", "?")
    print(f"  [{i}] score={sn.score:.3f}  source={src}")

print(f"\n=== Answer ===\n{response}")
```

Retrieves the top 5 nodes by embedding cosine similarity, sends them to the LLM for synthesis. On the four-file corpus this returns all four files (there is nothing to filter out) plus one that scored low but happens to be present.

## Same query, with a reranker

`02_with_reranker.py` differs from the baseline in three lines:

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=3,
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
```

`similarity_top_k=10` pulls a wider set of candidates from the vector store. `node_postprocessors=[reranker]` runs each candidate through the cross-encoder against the query. `top_n=3` keeps the three highest-scoring candidates after reranking.

Compare the two outputs. The baseline typically hands the LLM some low-scoring nodes that add noise to the context. The reranked version hands the LLM only the three the cross-encoder considers most relevant, and the resulting answer is usually tighter.

## What the reranker actually costs

For each reranker call, the cross-encoder runs once per candidate. On the tiny corpus here, 10 candidates cost about 100 ms of extra latency on a modern laptop. On a real corpus where you might retrieve 50-100 candidates for reranking, the cost is 500 ms - 2 seconds, still cheap relative to a local LLM call, and usually the single highest-value quality improvement you can make to a RAG pipeline.

Rules of thumb from my own projects:

- **`similarity_top_k` should be 3-5× your final `top_n`.** If you want three passages for the LLM, pull 10-15 candidates for the reranker. Below 3× you are not giving the reranker much to work with; above 5× is diminishing returns.
- **`BAAI/bge-reranker-base` is my default reranker model.** `BAAI/bge-reranker-large` is more accurate and 3-4× slower; not usually worth the trade unless quality is your bottleneck. `mixedbread-ai/mxbai-rerank-base-v1` is a strong open alternative I sometimes reach for.
- **Reranker + cheap embedding beats no reranker + expensive embedding.** If forced to choose between BGE-small + reranker versus BGE-base with no reranker, take the former every time.

## What we covered

- Node postprocessors run between the retriever and the LLM.
- `SentenceTransformerRerank` is the reranker most projects should reach for.
- Pull a wider candidate set for the reranker to consider (5× your final desired count).
- Reranking is nearly always the highest-value quality upgrade for a RAG pipeline.

Chapter "The Workflows API" changes gears from retrieval to orchestration, introducing the Workflows API, LlamaIndex's answer to LangGraph for multi-step LLM applications.
