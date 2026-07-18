"""The smallest useful LlamaIndex 0.14 program: load, index, query.

Configuration in LlamaIndex is centralized in `Settings`. Set the LLM and
the embedding model once, at the top of your script, and every downstream
component (VectorStoreIndex, query engines, retrievers) picks them up
automatically.
"""

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
