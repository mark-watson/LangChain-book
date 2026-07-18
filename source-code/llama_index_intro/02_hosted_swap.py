"""The same program as 01_hello_llamaindex.py, but pointed at OpenAI.

Only the `Settings.llm` line changes. Everything else — documents,
index, query engine, response — is identical. Requires OPENAI_API_KEY
in the environment.

Note: even here we still use a local HuggingFace embedding model. There
is no reason to pay OpenAI for embeddings when a good open one runs on
your laptop in a few milliseconds. Match the LLM to the task; embeddings
almost always stay local.
"""

import os

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY in your environment to run this example.")

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is the Austrian School of Economics?")
print(response)
