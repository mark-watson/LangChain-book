"""SimpleKeywordTableIndex: retrieval by keyword match, no embeddings.

The index extracts keywords from each Node using a plain regex (no LLM,
no embedding model). At query time, keywords are extracted from the
query the same way; Nodes whose keyword sets overlap the query's are
retrieved.

Where this shines:

- Factoid corpora full of specific terms (product SKUs, function names,
  legal citations, drug names). Semantic similarity is often confused
  by these; exact-term matching is not.
- Environments where you cannot install any ML dependencies. This index
  needs no embedding model and no LLM to build.
- As a cheap first-pass filter before a more expensive stage.

Where it fails:

- Any query where the important word does not appear literally in the
  relevant Node. Paraphrases are invisible.

The `KeywordTableIndex` variant (no "Simple") uses an LLM to extract
richer keyword lists. More accurate, slower to build, needs a model.
"""

from llama_index.core import SimpleDirectoryReader, SimpleKeywordTableIndex

documents = SimpleDirectoryReader("../data").load_data()
index = SimpleKeywordTableIndex.from_documents(documents)

# The retriever returns Nodes whose extracted keywords overlap the query's.
retriever = index.as_retriever()

for query in [
    "What is chemistry?",
    "Tell me about the Austrian School",
    "What is a laboratory?",
]:
    print(f"QUERY: {query}")
    nodes = retriever.retrieve(query)
    for i, n in enumerate(nodes, 1):
        src = n.node.metadata.get("file_name", "?")
        print(f"  [{i}] source={src}")
    print()
