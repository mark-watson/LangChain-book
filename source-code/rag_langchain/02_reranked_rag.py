"""Retrieve wide, rerank narrow.

Cosine similarity over sentence embeddings is fast but coarse. A cross-encoder
sees the query and each candidate together and can score them much more
accurately — at the cost of running the model once per candidate. The usual
pattern is: pull 10 candidates cheaply from the vector store, then rerank to
the top 3 with the cross-encoder before showing anything to the LLM.
"""

from typing import Any

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import CrossEncoder

from _corpus import TEST_QUESTIONS, load_documents


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


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(
    documents=load_documents(),
    embedding=embeddings,
    collection_name="ch4_reranked",
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

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
