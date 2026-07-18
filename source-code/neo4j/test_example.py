"""Neo4j knowledge-graph query engine with LlamaIndex 0.14.

Uses llama-index's Neo4j graph store integration. Requires Neo4j
credentials in credentials.json.
"""

import json

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.graph_stores.neo4j import Neo4jGraphStore

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Get Neo4j credentials
with open("credentials.json") as f:
    creds = json.load(f)

graph_store = Neo4jGraphStore(
    username=creds["username"],
    password=creds["password"],
    url=creds["url"],
    database=creds["database"],
)

# Load documents (example: from a local directory)
documents = SimpleDirectoryReader("../data").load_data()

# Build an index using the graph store
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What are the benefits of a paleo diet?")
print(f"{response}\n\n")

response = query_engine.query("What kinds of food should I buy for a paleo diet?")
print(response)
