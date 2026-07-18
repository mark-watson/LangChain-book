"""Index Google Drive text files and answer questions with LlamaIndex 0.14.

Uses local HuggingFace embeddings and a local Ollama LLM.
Run fetch_txt_files.py first to download the .txt files into ./data/.
"""

import sys
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

data_dir = Path("data")
if not data_dir.exists() or not any(data_dir.iterdir()):
    print(
        "No files found in ./data/\n\n"
        "Run fetch_txt_files.py first to download your Google Drive .txt files.\n"
        "That script requires Google OAuth credentials (client_secrets.json).\n"
        "Setup guide: https://developers.google.com/drive/api/quickstart/python"
    )
    sys.exit(0)

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

print(query_engine.query("What is the definition of sport?"))
