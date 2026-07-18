"""as_retriever() vs as_query_engine().

`.as_query_engine()` returns an object whose `.query()` runs retrieval AND
sends the retrieved context plus the query to an LLM for synthesis.

`.as_retriever()` returns just the retrieval half. `.retrieve(query)`
returns the raw `NodeWithScore` objects that the LLM would have seen,
without calling the LLM at all. Useful when you want to plug LlamaIndex
retrieval into your own custom prompt, or feed it into a different chain
entirely.
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)

retriever = index.as_retriever(similarity_top_k=3)

nodes = retriever.retrieve("What is the Austrian School of Economics?")

print(f"Retrieved {len(nodes)} nodes:\n")
for i, node_with_score in enumerate(nodes, 1):
    node = node_with_score.node
    source = node.metadata.get("file_name", "?")
    snippet = node.text[:120].replace("\n", " ")
    print(f"[{i}] score={node_with_score.score:.3f}  source={source}")
    print(f"    {snippet}...\n")
