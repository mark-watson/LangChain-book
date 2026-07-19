"""Hybrid retrieval with QueryFusionRetriever.

Two retrievers over the same corpus:

- BM25 (sparse, keyword-based). Great with exact terms and proper nouns.
- VectorStoreIndex (dense, embedding-based). Great with paraphrases and
  semantic similarity.

`QueryFusionRetriever` runs both, then merges the ranked lists with
reciprocal rank fusion. Documents that score highly on either retriever
end up highly in the final list; documents that score highly on both
end up at the very top.

This is the practical hybrid pattern in LlamaIndex — same idea as the
reciprocal-rank-fusion retriever from Chapter 2's LangChain examples,
different framework.
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()

# Dense retriever from a normal VectorStoreIndex.
vector_index = VectorStoreIndex.from_documents(documents)
dense = vector_index.as_retriever(similarity_top_k=3)

# Sparse retriever from a BM25 index over the same corpus.
sparse = BM25Retriever.from_defaults(
    nodes=list(vector_index.docstore.docs.values()),
    similarity_top_k=3,
)

# Fusion.
fusion = QueryFusionRetriever(
    retrievers=[dense, sparse],
    similarity_top_k=3,
    num_queries=1,   # 1 = use the user's query as-is; >1 = LLM query rewriting
    mode="reciprocal_rerank",
    use_async=False,
    verbose=False,
)

for query in [
    "How does body chemistry affect exercise?",
    "Austrian School",
]:
    print(f"QUERY: {query}")
    nodes = fusion.retrieve(query)
    for i, n in enumerate(nodes, 1):
        src = n.node.metadata.get("file_name", "?")
        print(f"  [{i}] score={n.score:.3f}  source={src}")
    print()
