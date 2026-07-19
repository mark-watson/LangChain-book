"""Hybrid retrieval: dense embeddings for meaning + BM25 for exact terms.

Dense embeddings excel at "what does this mean" but can miss documents that
share the exact rare word or proper noun with the query. BM25 is the opposite:
great with exact matches, blind to paraphrases. Ensembling them with reciprocal
rank fusion gives you the best of both, and often changes the answer on queries
with proper nouns or technical terms.
"""

from typing import Any

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

from _corpus import TEST_QUESTIONS, load_documents


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
    """Fuse several retrievers with reciprocal rank fusion.

    langchain.retrievers.EnsembleRetriever is no longer importable from the
    langchain package (the whole langchain.retrievers namespace was retired,
    same story as BM25Retriever above) — this reimplements standard RRF:
    each retriever contributes weight / (k + rank) to a document's combined
    score, documents are deduplicated by source, and the fused ranking is
    the sum of those scores, highest first.
    """

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


docs = load_documents()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="ch4_hybrid",
)

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 3
dense = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = RRFEnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n\n"
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
