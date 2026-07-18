"""SummaryIndex: every query touches every Node.

The name is slightly misleading — SummaryIndex is not specifically about
summarization. It is about queries where you want the LLM to see every
document, in order, and reason across all of them. Common uses:

- "Summarize this whole corpus."
- "What are the common themes across these documents?"
- "Give me an overview of everything in this folder."

Because every query hits every Node, this index is O(n) per query. Fine
for small corpora, unusable for large ones. But when it is the right
tool, no vector retriever will match it — vector retrieval is designed
to find the top few, not to reason across all.
"""

from llama_index.core import Settings, SimpleDirectoryReader, SummaryIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=180.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query(
    "Give me a one-paragraph overview of the topics covered by these documents."
)
print(response)
