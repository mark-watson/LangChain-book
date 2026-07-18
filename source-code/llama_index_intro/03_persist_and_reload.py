"""Persist an index to disk and reload it in the same script.

Real projects almost always split index building from index querying —
building is expensive (embed every document, write to a vector store),
querying is cheap (embed one query, look up neighbors, call the LLM).
LlamaIndex's storage layer handles this cleanly with `.persist(dir)` on
the way out and `load_index_from_storage(StorageContext.from_defaults(...))`
on the way back in.
"""

from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

PERSIST_DIR = Path(__file__).parent / "storage"

# --- Build and persist ---
print("Building index and persisting to disk...")
documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=str(PERSIST_DIR))
print(f"Persisted to {PERSIST_DIR}")

# --- Reload and query ---
print("\nReloading index from disk...")
storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
reloaded = load_index_from_storage(storage_context)

query_engine = reloaded.as_query_engine()
response = query_engine.query("What is the definition of sport?")
print(f"\nAnswer:\n{response}")
