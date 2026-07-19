# RAG patterns with LangChain

Retrieval-Augmented Generation is the workhorse pattern of applied LLM development. The idea is one paragraph long: before you send a user's question to the model, look up the passages in your corpus that are most likely to contain the answer, and paste them into the prompt as context. The model gets to condition its answer on your data, not just on what it saw during training. Hallucinations drop, answers stay grounded, and you can add or remove documents from the corpus without retraining anything.

![RAG System Overview](RAG_diagram.png)

Everything I wrote in the previous edition about *why* RAG works still applies. What has changed is that in 2026, the parts that used to be interesting — chunking, embedding, storing — are commoditized and take about six lines each. The interesting choice now is which *retrieval pattern* to use, because a naive retriever is often not good enough, and the differences between the good patterns are meaningful and worth understanding.

This chapter walks through four patterns over the same tiny four-file corpus. Everything runs locally on your laptop with `qwen3.5:4b` from Ollama as the LLM and BGE from Hugging Face as the embedding model.

| Script | Pattern |
|---|---|
| `01_naive_rag.py` | Dense-embedding vector search, straight into the prompt. |
| `02_reranked_rag.py` | Retrieve top 10, rerank to top 3 with a cross-encoder. |
| `03_hybrid_rag.py` | BM25 keyword search + dense embeddings ensembled. |
| `04_multi_query_rag.py` | Let an LLM rephrase the question, union the retrievals. |

The four scripts live in `source-code/rag_langchain/` and share the corpus in `source-code/data/` (four short text files on chemistry, economics, health, and sports). Set up:

```console
$ cd source-code/rag_langchain
$ uv sync
$ ollama pull qwen3.5:4b
```

The first script will also download two small models from Hugging Face — the BGE embedding model and, in the reranker example, a BGE cross-encoder. Roughly 350 MB total, cached under `~/.cache/huggingface/` after the first run.

## A shared corpus loader

All four scripts pull documents through the same helper, `_corpus.py`:

```python
from pathlib import Path

from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent.parent / "data"


def load_documents() -> list[Document]:
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


TEST_QUESTIONS = [
    "Who tried to define what chemistry is?",
    "What is the Austrian School of Economics?",
    "How does body chemistry affect exercise?",
]
```

Two things to notice. First, `Document` is the core type LangChain retrievers pass around; `page_content` is the text and `metadata` is an arbitrary dict you can filter and sort on. Second, the four .txt files are short enough that each one comfortably fits into a single `Document` — I am deliberately skipping the chunking step for this chapter so the retrieval strategy is the only variable that changes between the four scripts. For a larger corpus you would run the loaded documents through a `RecursiveCharacterTextSplitter` before indexing.

## Pattern 1: naive RAG

The baseline. Every later pattern is a modification of this shape. Here is `01_naive_rag.py` end-to-end:

```python
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from _corpus import TEST_QUESTIONS, load_documents

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(
    documents=load_documents(),
    embedding=embeddings,
    collection_name="ch4_naive",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


model = ChatOllama(model="qwen3.5:4b", temperature=0)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

for q in TEST_QUESTIONS:
    print(f"Q: {q}")
    print(f"A: {chain.invoke(q).strip()}\n")
```

The interesting line is the chain definition. LCEL lets us wire a dict, a prompt, a model, and a parser into a single `Runnable`. The dict is where the "R" of RAG happens: for each incoming question we run the retriever, format the resulting `Document` list into a string with `format_docs`, and pass the question through unchanged. The prompt template consumes both variables, the model produces an `AIMessage`, and the parser hands back a plain string. The whole pipeline supports `.invoke`, `.stream`, and `.batch` just like the model does on its own.

The retriever asks the vector store for the top *k* = 2 most similar documents by cosine similarity over the BGE embeddings. That is enough for our four-file corpus. In a larger corpus you would want `k` in the neighborhood of 4-10 combined with a reranker, which is exactly what the next script does.

## Pattern 2: retrieve wide, rerank narrow

Sentence embeddings are fast but coarse. Every document gets encoded once, offline; every query gets encoded once, online; you compare with a cosine similarity that treats the query and each document independently. A cross-encoder is the opposite: it sees the query and one candidate document together in a single forward pass and scores their relevance directly. Cross-encoders are much more accurate and much slower, so you cannot use them on your whole corpus. The trick is to use the cheap embedding retriever to pull a wider set of candidates, then let the cross-encoder rerank them.

`02_reranked_rag.py` differs from the naive version in exactly the retriever setup:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
```

`ContextualCompressionRetriever` is a general wrapper: it takes a base retriever plus a "compressor" that filters or reorders the base retriever's output. `CrossEncoderReranker` is one such compressor. The base retriever now returns *k* = 10 candidates instead of 2, and the reranker keeps only the three that the cross-encoder scores highest.

On this four-file corpus the reranker's effect is subtle because the corpus is small. On a real corpus of thousands of chunks, adding a reranker is often the single biggest single quality improvement you can make to a RAG system for the smallest amount of code, and I add it early in almost every project.

## Pattern 3: hybrid dense + sparse retrieval

Dense embeddings are good at "what does this mean" and can miss documents that share the exact rare word or proper noun with the query. BM25 is the opposite: it is the classical bag-of-words scoring function from information retrieval, brilliant with exact term matches, blind to paraphrases. Neither one dominates the other across all queries.

`03_hybrid_rag.py` runs both and ensembles them:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

docs = load_documents()

vectorstore = Chroma.from_documents(
    documents=docs, embedding=embeddings, collection_name="ch4_hybrid",
)

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 3
dense = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
```

`EnsembleRetriever` uses reciprocal rank fusion under the hood: each source retriever produces a ranked list, each document gets a score based on where it appears in each list, and the fused ranking is the sum of those scores weighted by the numbers you pass in. Equal weights are a reasonable default. In production I usually push a bit toward dense (say `[0.4, 0.6]`) for prose-heavy corpora and toward BM25 (`[0.6, 0.4]`) for corpora full of identifiers, product SKUs, or legal citations.

Notable: BM25 has no embedding step, so `BM25Retriever.from_documents` returns immediately regardless of corpus size. That makes hybrid retrieval more or less free to add on the retrieval side; the only cost is that you do two lookups per query instead of one.

## Pattern 4: multi-query rewriting

Users phrase questions in one specific way. The passage that answers their question in your corpus may be phrased another. Multi-query retrieval addresses this by asking an LLM to generate three or four rephrasings of the user's question, running a dense retrieval for each, and unioning the results.

`04_multi_query_rag.py`:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

rewriter = ChatOllama(model="qwen3.5:4b", temperature=0)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=rewriter,
)
```

That's the whole difference. The chain composition, prompt, and answering model are identical to the naive script. The tradeoff is one extra LLM call per user query (the rewriter's call), traded for meaningfully better recall on questions whose surface form is unlike the corpus.

You can also read the generated queries by turning on LangChain's logging — helpful when debugging why a particular question is or isn't finding the right documents.

## Which pattern for which situation

A rough decision tree from projects I have shipped:

- Start with **naive** if your corpus is under a few hundred documents. Anything more complex is over-engineering until you have real query traffic to measure against.
- Add **reranking** the moment your corpus exceeds a few thousand chunks or the moment you notice the model answering from irrelevant passages. This is nearly always the highest-return upgrade.
- Add **hybrid** if your corpus is heavy in proper nouns, product identifiers, code snippets, or legal/medical citations — anything where exact string matches matter more than semantic similarity.
- Add **multi-query** last, when you have evidence that user queries and corpus phrasing are systematically different. It is the most expensive of the four and the improvement is the hardest to predict.

You can also stack them. A reasonable production retriever is "hybrid BM25 + dense, then cross-encoder rerank," and that is what I default to for new projects when I have no other information.

## What we covered

RAG has boiled down to two concrete choices in 2026: which retrieval pattern (this chapter) and which retriever+reranker+chunker stack (LangChain gives you the pieces, LlamaIndex will give you an even richer set in Part II). The chain composition, the prompt template, and the model call are essentially fixed shapes at this point. Learn the four patterns above and you can compose a strong retriever for almost any application without leaving the boundaries of `langchain-core`, `langchain-community`, `langchain-ollama`, and `langchain-huggingface`.

Chapter 5 covers tool calling in more depth — the primitive from Chapter 1 § "Tool binding" — and Chapter 6 introduces LangGraph, which is where we start building agents that use RAG as one of several tools rather than as the whole app.
