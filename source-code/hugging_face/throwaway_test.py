"""Minimal LlamaIndex 0.14 index-and-query test.

Uses the Settings API with a local Ollama LLM and local HuggingFace
embeddings. No external API keys required.
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load documents from a directory
documents = SimpleDirectoryReader("../data").load_data()

# Create a new index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Query the index
results = query_engine.query("what is the history of economics?")
print(f"results: {results}")
