"""Embedchain replacement: simple RAG with LlamaIndex 0.14.

The embedchain library is unmaintained. This example replaces it with
LlamaIndex's SimpleDirectoryReader + VectorStoreIndex, which provides
the same add-and-query functionality using local models.
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Build index from PDF data directory
import os

data_dir = "./data/"
if os.path.isdir(data_dir):
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
else:
    # Fallback: use a simple in-memory document
    from llama_index.core import Document

    index = VectorStoreIndex.from_documents([Document(text="No data directory found.")])

query_engine = index.as_query_engine()


def test(q):
    print(q)
    print(query_engine.query(q), "\n")


test("How can I iterate over a list in Haskell?")
test("How can I edit my Common Lisp files?")
test("How can I scrape a website using Common Lisp?")
