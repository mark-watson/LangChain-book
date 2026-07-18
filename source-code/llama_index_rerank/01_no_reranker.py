"""Baseline: vector retrieval with no reranking."""

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
