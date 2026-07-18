"""Two embedding models, same corpus, same query.

The choice of embedding model changes which Nodes come back and in
what order. Both models used here are small and free and run locally;
the point of the script is not "which is better in general" (that
question doesn't have a single answer) but "how much does the choice
actually matter for my corpus and my queries."

Try re-running with different queries to see when the two models
disagree.
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MODELS = [
    ("BGE-small (BAAI/bge-small-en-v1.5)", "BAAI/bge-small-en-v1.5"),
    ("MiniLM (sentence-transformers/all-MiniLM-L6-v2)", "sentence-transformers/all-MiniLM-L6-v2"),
]

QUERY = "How does the body process energy during exercise?"

documents = SimpleDirectoryReader("../data").load_data()

for label, model_name in MODELS:
    print(f"=== {label} ===")
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=3)

    nodes = retriever.retrieve(QUERY)
    for i, n in enumerate(nodes, 1):
        source = n.node.metadata.get("file_name", "?")
        print(f"  [{i}] score={n.score:.3f}  source={source}")
    print()
