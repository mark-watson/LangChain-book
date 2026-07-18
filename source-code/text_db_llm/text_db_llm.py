"""Text database supporting search and chat-based exploration.

Uses LlamaIndex 0.14 with a Chroma vector store and local HuggingFace
embeddings. No external API keys required.
"""

from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("temp")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=Settings.embed_model
)
query_engine = index.as_query_engine()
print(query_engine.query("effect of body chemistry on exercise?"))
