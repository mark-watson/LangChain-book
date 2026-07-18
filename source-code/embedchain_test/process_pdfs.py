"""Process PDFs with LlamaIndex 0.14 (replaces embedchain).

Uses LlamaIndex's SimpleDirectoryReader to load PDF files and build
a vector index. Local models only — no external API keys required.
"""

import os

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

my_books_dir = "./data/"

if os.path.isdir(my_books_dir):
    documents = SimpleDirectoryReader(my_books_dir).load_data()
    print(f"Loaded {len(documents)} documents from {my_books_dir}")

    index = VectorStoreIndex.from_documents(documents)
    print("Index built successfully.")
else:
    print(f"Directory {my_books_dir} does not exist. Nothing to process.")
