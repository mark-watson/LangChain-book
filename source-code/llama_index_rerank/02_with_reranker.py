"""The same query engine, but with a cross-encoder reranker attached.

Pattern: pull more candidates from the vector store than you need (top
10 instead of top 3), then let a cross-encoder rescore them all against
the query. The top 3 after reranking are what the LLM actually sees.

The cross-encoder model is much more accurate at ranking than the
embedding cosine similarity, at the cost of running one small model
per (query, candidate) pair. For 10 candidates on a laptop, that is
50-200 ms of extra latency — nearly always worth it.
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=180.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=3,
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)

query = "How does body chemistry affect exercise performance?"
response = query_engine.query(query)

print("=== Retrieved nodes (after reranking) ===")
for i, sn in enumerate(response.source_nodes, 1):
    src = sn.node.metadata.get("file_name", "?")
    print(f"  [{i}] score={sn.score:.3f}  source={src}")

print(f"\n=== Answer ===\n{response}")
