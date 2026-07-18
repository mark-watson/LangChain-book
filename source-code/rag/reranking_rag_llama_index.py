"""RAG with reranking using LlamaIndex 0.14.

Loads documents, builds a vector index, and uses a cross-encoder
reranker as a node postprocessor to improve retrieval quality.
Uses local models only.
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load documents
documents = SimpleDirectoryReader("../data").load_data()

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Initialize the reranker with a specific model
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_n=3,
)

# Set up the query engine with the reranker as a postprocessor
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)

# Perform a query
response = query_engine.query("Compare sports with the study of health issues")

print(response)
