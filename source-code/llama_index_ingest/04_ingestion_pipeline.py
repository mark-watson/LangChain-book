"""Chunking documents with an IngestionPipeline.

The four text files in ../data are small enough that each one is a
single Node. Real corpora rarely look like that — a 40-page PDF wants
to be broken into ~50 smaller Nodes so retrieval returns tight, focused
context instead of the whole document at once.

`IngestionPipeline` runs an ordered list of transformations. The most
common one is `SentenceSplitter`, which breaks documents into ~1000-char
chunks at sentence boundaries. You can chain other transforms too:
`TitleExtractor` to add a title metadata field, `SummaryExtractor` to
add a per-chunk summary, and so on.

The pipeline output is a list of Nodes, ready to be handed to any index.
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
print(f"Loaded {len(documents)} documents")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=200, chunk_overlap=20),
    ]
)

nodes = pipeline.run(documents=documents)
print(f"Produced {len(nodes)} nodes after splitting\n")

# Inspect the first few nodes.
for i, n in enumerate(nodes[:5], 1):
    src = n.metadata.get("file_name", "?")
    snippet = n.text[:120].replace("\n", " ")
    print(f"[{i}] source={src}  chars={len(n.text)}")
    print(f"    {snippet}...")
    print()

# Nodes can be handed directly to an index instead of Documents.
index = VectorStoreIndex(nodes)
retriever = index.as_retriever(similarity_top_k=3)

hits = retriever.retrieve("What is the definition of sport?")
print(f"=== top-3 retrieval on the chunked corpus ===")
for i, h in enumerate(hits, 1):
    src = h.node.metadata.get("file_name", "?")
    print(f"[{i}] score={h.score:.3f}  source={src}  chars={len(h.node.text)}")
