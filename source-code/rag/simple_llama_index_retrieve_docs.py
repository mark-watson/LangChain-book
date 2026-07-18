"""Retrieve documents (without LLM synthesis) using LlamaIndex 0.14.

Shows the difference between as_query_engine() and as_retriever().
"""

from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

text_list = [
    "LlamaIndex is a powerful tool for LLM applications.",
    "It helps in structuring and retrieving data efficiently.",
]
documents = [Document(text=t) for t in text_list]

index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()

retrieved_docs = retriever.retrieve("What is LlamaIndex?")
print(retrieved_docs)
