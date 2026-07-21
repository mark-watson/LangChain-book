# RAG patterns with LangChain

Retrieval-Augmented Generation was the workhorse pattern of early applied LLM development, and still has large practical use. The idea is one paragraph long: before you send a user's question to the model, look up the passages (we will call these samples of text "chunks")in your corpus that are most likely to contain the answer, and paste these matching chunks of text into the prompt as context. The model gets to condition its answer on your data, not just on what it saw during training. Hallucinations drop, answers stay grounded, and you can add or remove documents from the corpus without retraining anything.

Please note, dear reader, that commercial chat apps like ChatGPT and Gemini became much more useful when they started using web search to gather chunks of text to add to context. We will also see several examples of integrating data from web search in later examples in this book.

![RAG System Overview](RAG_diagram.png)

Everything I wrote in the previous edition about *why* RAG works still applies. What has changed is that in 2026, the parts that used to be interesting — chunking, embedding, storing — are commoditized and take about six lines each. The interesting choice now is which *retrieval pattern* to use, because a naive retriever is often not good enough, and the differences between the good patterns are meaningful and worth understanding.

This chapter walks through four patterns over the same tiny four-file corpus. Everything runs locally on your laptop with `qwen3.5:4b` served by Ollama as the LLM and use BGE from Hugging Face as the embedding model.

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

Two things to notice. First, `Document` is the core type LangChain retrievers pass around; `page_content` is the text and `metadata` is an arbitrary dict you can filter and sort on. Second, the four .txt files are short enough that each one comfortably fits into a single `Document`. I am deliberately skipping the chunking step for this chapter so the retrieval strategy is the only variable that changes between the four scripts. For a larger corpus you would run the loaded documents through a `RecursiveCharacterTextSplitter` before indexing.

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

`02_reranked_rag.py` differs from the naive version in the retriever setup — plus two small classes that would not have existed in earlier editions of this chapter. `langchain_community.cross_encoders.HuggingFaceCrossEncoder` and `langchain.retrievers.ContextualCompressionRetriever`/`CrossEncoderReranker` are all retired; neither the `langchain-community` package nor a `langchain.retrievers` module exists to import them from anymore. Both are short enough to reimplement directly on top of `sentence-transformers` and `langchain_core`:

```python
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import CrossEncoder


class HuggingFaceCrossEncoder(BaseModel, BaseCrossEncoder):
    """sentence-transformers cross encoder (langchain_community.cross_encoders is retired)."""

    model_name: str = "BAAI/bge-reranker-base"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    client: Any = None

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = CrossEncoder(self.model_name, **self.model_kwargs)

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.client.predict(text_pairs)
        if len(scores.shape) > 1:
            scores = [s[1] for s in scores]
        return scores


class RerankRetriever(BaseRetriever):
    """Retrieve wide from a base retriever, then rerank narrow with a cross-encoder."""

    base_retriever: BaseRetriever
    cross_encoder: BaseCrossEncoder
    top_n: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.score(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]
```

With those in place, building the retriever is exactly the two-step shape the old classes used to give you for free:

```python
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
retriever = RerankRetriever(base_retriever=base_retriever, cross_encoder=reranker_model, top_n=3)
```

`RerankRetriever` takes a base retriever plus the cross-encoder above: it calls the base retriever, scores every candidate against the query with the cross-encoder, and keeps the top 3. The base retriever now returns *k* = 10 candidates instead of 2, and the reranker keeps only the three that score highest.

Representative output:

```console
$ uv run 02_reranked_rag.py
Q: Who tried to define what chemistry is?
A: Based on the context provided, several individuals are mentioned who defined or characterized chemistry:

*   **Georg Ernst Stahl** (1730) used a definition referring to resolving mixed bodies into their principles and composing them from those principles.
*   **Jean-Baptiste Dumas** (1837) considered the word "chemistry" to refer to the science concerned with the laws and effects of molecular forces.
*   **Linus Pauling** accepted a characterization in 1947 regarding the science of substances, their structure, properties, and reactions.
*   **Professor Raymond Chang** (1998) phrased the definition as "the study of matter and the changes it undergoes."

Q: What is the Austrian School of Economics?
A: Based on the provided text, the Austrian School is a school of economic thought that emphasizes the spontaneous organizing power of the price mechanism. It is also known as the Vienna School or the Psychological School...

Q: How does body chemistry affect exercise?
A: Based on the provided context, there is no information regarding how body chemistry affects exercise.

The text discusses general chemical principles in **[chemistry.txt]**... but does not mention human physiology or exercise performance. The **sports** section in **[sports.txt]** defines sport and physical activity based on athleticism, dexterity, and rules for competition, without linking these concepts to chemical processes within the body.
```

On this four-file corpus the reranker's effect is subtle because the corpus is small — notice the third answer still comes up empty, which the hybrid pattern next fixes. On a real corpus of thousands of chunks, adding a reranker is often the single biggest single quality improvement you can make to a RAG system for the smallest amount of code, and I add it early in almost every project.

## Pattern 3: hybrid dense + sparse retrieval

Dense embeddings are good at "what does this mean" but is not likely to miss documents that share the exact rare word or proper noun with the query. BM25 is the opposite: it is the classical bag-of-words scoring function from information retrieval, brilliant with exact term matches, blind to paraphrases. Neither one dominates the other across all queries.

`03_hybrid_rag.py` runs both and ensembles them. Same story as the reranker: `langchain_community.retrievers.BM25Retriever` and `langchain.retrievers.EnsembleRetriever` are both retired, so this script hand-rolls a BM25 retriever on top of `rank_bm25` and its own reciprocal-rank-fusion ensembler:

```python
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi


class BM25Retriever(BaseRetriever):
    """Keyword retriever built on rank_bm25 (langchain_community.retrievers is retired)."""

    vectorizer: Any
    docs: list[Document] = Field(repr=False)
    k: int = 4

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(cls, documents: list[Document], **kwargs: Any) -> "BM25Retriever":
        docs = list(documents)
        corpus = [doc.page_content.split() for doc in docs]
        return cls(vectorizer=BM25Okapi(corpus), docs=docs, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.vectorizer.get_top_n(query.split(), self.docs, n=self.k)


class RRFEnsembleRetriever(BaseRetriever):
    """Fuse several retrievers with reciprocal rank fusion."""

    retrievers: list[BaseRetriever]
    weights: list[float]
    k: int = 60

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scores: dict[str, float] = {}
        docs_by_key: dict[str, Document] = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            for rank, doc in enumerate(retriever.invoke(query), start=1):
                key = doc.metadata.get("source", doc.page_content)
                scores[key] = scores.get(key, 0.0) + weight / (self.k + rank)
                docs_by_key.setdefault(key, doc)
        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        return [docs_by_key[key] for key in ranked_keys]
```

Wiring them together:

```python
docs = load_documents()

vectorstore = Chroma.from_documents(
    documents=docs, embedding=embeddings, collection_name="ch4_hybrid",
)

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 3
dense = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = RRFEnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
```

`RRFEnsembleRetriever` is reciprocal rank fusion, spelled out: each source retriever produces a ranked list, each document earns `weight / (k + rank)` from every list it appears in (`k = 60` is the standard RRF constant — it flattens out the difference between, say, rank 1 and rank 2 so one retriever's top pick cannot completely dominate), and the fused ranking is the sum of those scores, highest first. Equal weights are a reasonable default. In production I usually push a bit toward dense (say `[0.4, 0.6]`) for prose-heavy corpora and toward BM25 (`[0.6, 0.4]`) for corpora full of identifiers, product SKUs, or legal citations.

Notable: BM25 has no embedding step, so `BM25Retriever.from_documents` returns immediately regardless of corpus size. That makes hybrid retrieval more or less free to add on the retrieval side; the only cost is that you do two lookups per query instead of one.

Representative output — notice the third question, which came up empty for both the naive and reranked patterns, gets a real answer here because BM25's exact-term matching pulls in `health.txt`'s ATP/anaerobic-exercise passage that dense retrieval alone was missing:

```console
$ uv run 03_hybrid_rag.py
Q: Who tried to define what chemistry is?
A: Based on the provided context, several individuals are mentioned who defined or characterized what chemistry is:

*   **Georg Ernst Stahl** (1730): Defined "chemistry" as the art of resolving mixed, compound, or aggregate bodies into their principles and composing such bodies from those principles.
*   **Jean-Baptiste Dumas** (1837): Considered the word to refer to the science concerned with the laws and effects of molecular forces.
*   **Linus Pauling**: Accepted a characterization in 1947 that came to mean "the science of substances: their structure, their properties, and the reactions that change them into other substances."
*   **Professor Raymond Chang** (1998): Phrased the definition as broadened to mean "the study of matter and the changes it undergoes."

Q: What is the Austrian School of Economics?
A: Based on the provided context, the Austrian School is a school of economic thought (also known as the Vienna School or the Psychological School) that emphasizes the spontaneous organizing power of the price mechanism...

Q: How does body chemistry affect exercise?
A: Based on the provided context, body chemistry relates to exercise through energy storage and production systems:

*   **Adenosine triphosphate (ATP):** This is described as an energy-storing molecule that is found in all human cells ([health.txt]).
*   **Anaerobic Exercise:** The text defines this as exercise that improves the efficiency of energy-producing systems that do not rely on oxygen, such as sprinting and weight lifting.

Additionally, general chemistry is defined in [chemistry.txt] as the science concerned with substances: their structure, properties, and reactions that change them into other substances, involving interactions between atoms leading to rearrangements of chemical bonds.
```

## Pattern 4: multi-query rewriting

Users phrase questions in one specific way. The passage that answers their question in your corpus may be phrased another. Multi-query retrieval addresses this by asking an LLM to generate three or four rephrasings of the user's question, running a dense retrieval for each, and unioning the results.

`langchain.retrievers.multi_query.MultiQueryRetriever` is retired along with the rest of `langchain.retrievers` (the same story as Patterns 2 and 3), so `04_multi_query_rag.py` hand-rolls the same generate-then-union shape:

```python
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


class MultiQueryRetriever(BaseRetriever):
    """Ask an LLM to rephrase the query, retrieve for each phrasing, union the results."""

    retriever: BaseRetriever
    llm: BaseChatModel
    num_queries: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _generate_queries(self, query: str) -> list[str]:
        prompt = (
            f"Generate {self.num_queries} different rephrasings of the question below, "
            "each capturing the same information need from a different angle. "
            "Return one rephrasing per line, with no numbering or extra commentary.\n\n"
            f"Question: {query}"
        )
        response = self.llm.invoke(prompt)
        lines = (line.strip("-• \t") for line in response.content.splitlines())
        return [line for line in lines if line]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        seen: dict[str, Document] = {}
        for q in [query, *self._generate_queries(query)]:
            for doc in self.retriever.invoke(q):
                key = doc.metadata.get("source", doc.page_content)
                seen.setdefault(key, doc)
        return list(seen.values())
```

Wired up:

```python
rewriter = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False)

retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=rewriter,
)
```

The chain composition, prompt, and answering model below this are identical to the naive script. The tradeoff is one extra LLM call per user query (the rewriter's call), traded for meaningfully better recall on questions whose surface form is unlike the corpus.

You can also read the generated queries yourself — `retriever._generate_queries(q)` — which is helpful when debugging why a particular question is or isn't finding the right documents.

Representative output:

```console
$ uv run 04_multi_query_rag.py
Q: Who tried to define what chemistry is?
A: Based on the context provided, the following individuals tried to define what chemistry is (or characterized it):

*   **Georg Ernst Stahl** (1730) defined "chemistry" as the art of resolving mixed bodies into their principles and composing them from those principles.
*   **Jean-Baptiste Dumas** (1837) considered "chemistry" to refer to the science concerned with the laws and effects of molecular forces.
*   **Linus Pauling** accepted a characterization in 1947 that defined chemistry as the science of substances: their structure, properties, and reactions.
*   **Professor Raymond Chang** (1998) phrased "chemistry" to mean the study of matter and the changes it undergoes.

Q: What is the Austrian School of Economics?
A: Based on the provided context, the Austrian School (also known as the Vienna School or the Psychological School) is a school of economic thought that emphasizes the spontaneous organizing power of the price mechanism. Austrians hold that mathematical modelling of the evolving market is extremely difficult due to subjective human choices and advocate a "laissez faire" approach to the economy...

Q: How does body chemistry affect exercise?
A: 
```

That third answer is not a typo: it is genuinely empty, reproducibly, on this corpus. With four short files and `k = 3`, unioning four query variants' retrievals pulls in nearly the entire corpus (all three non-sports files, in this run), and the resulting context pushes `qwen3.5:4b` into a degenerate generation for this particular question: `done_reason` comes back `"length"` after thousands of tokens, with nothing usable in `.content` even at `thinking=False` and with an explicit `num_predict` cap. The naive, reranked, and hybrid patterns above never hit this because they hand the model a much smaller, more targeted context. It is a genuine limitation of multi-query on a tiny corpus with a small local model, not a bug in the retriever logic above — and a good reminder that "retrieve more, union it all" is not free: past some point, more context can make a small model's job harder, not easier. Watching for exactly this kind of silent failure is why every RAG chain in this book prints the answer instead of assuming one arrived.

## Which pattern for which situation

A rough decision tree from projects I have shipped:

- Start with **naive** if your corpus is under a few hundred documents. Anything more complex is over-engineering until you have real query traffic to measure against.
- Add **reranking** the moment your corpus exceeds a few thousand chunks or the moment you notice the model answering from irrelevant passages. This is nearly always the highest-return upgrade.
- Add **hybrid** if your corpus is heavy in proper nouns, product identifiers, code snippets, or legal/medical citations — anything where exact string matches matter more than semantic similarity.
- Add **multi-query** last, when you have evidence that user queries and corpus phrasing are systematically different. It is the most expensive of the four and the improvement is the hardest to predict.

You can also stack them. A reasonable production retriever is "hybrid BM25 + dense, then cross-encoder rerank," and that is what I default to for new projects when I have no other information.

## What we covered

RAG has boiled down to two concrete choices in 2026: which retrieval pattern (this chapter) and which retriever+reranker+chunker stack (LangChain gives you the pieces, LlamaIndex will give you an even richer set in Part II). The chain composition, the prompt template, and the model call are essentially fixed shapes at this point. Learn the four patterns above and you can compose a strong retriever for almost any application without leaving the boundaries of `langchain-core`, `langchain-ollama`, and `langchain-huggingface` — `langchain-community` is not a dependency anywhere in this chapter; every retriever this book needed but the framework no longer ships is about forty lines of `langchain_core.retrievers.BaseRetriever` away.

Tool binding — the primitive from Chapter 1 (section )"Tool binding", example file 06_tool_binding.py) is as far as this book goes for low-level tool calling by itself. Chapter 3 introduces LangGraph, which is where we start building agents that use RAG as one of several tools rather than as the whole app.
