"""Building Documents by hand.

When your source is a REST API, a database, a message queue, or anything
else that SimpleDirectoryReader doesn't understand, you construct
Document objects yourself. The rest of LlamaIndex — indices, query
engines, retrievers — doesn't care where the Documents came from.

Metadata that you put on a Document flows through chunking and shows up
on each resulting Node, so it survives all the way to retrieval time.
Use it for anything you might want to filter or cite by.
"""

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Imagine these came from a REST API or a database query.
records = [
    {
        "id": "note-001",
        "author": "Mark",
        "title": "Sedona hiking",
        "body": "Cathedral Rock and Bell Rock are the most popular hikes near Sedona, Arizona.",
    },
    {
        "id": "note-002",
        "author": "Carol",
        "title": "Sedona food",
        "body": "The Mexican restaurants in Sedona serve especially good chile relleno.",
    },
    {
        "id": "note-003",
        "author": "Mark",
        "title": "Prescott hiking",
        "body": "Thumb Butte in Prescott, Arizona is a short but steep hike with a great summit view.",
    },
]

documents = [
    Document(
        text=r["body"],
        metadata={"id": r["id"], "author": r["author"], "title": r["title"]},
    )
    for r in records
]

index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=2)

nodes = retriever.retrieve("What hikes are near Sedona?")

for i, n in enumerate(nodes, 1):
    print(f"[{i}] score={n.score:.3f}  author={n.node.metadata['author']}  title={n.node.metadata['title']!r}")
    print(f"    {n.node.text}\n")
