"""Multi-query retrieval: let the LLM rewrite the question several ways.

Users phrase questions in one specific way; the relevant passage in your
corpus may be phrased another. MultiQueryRetriever asks an LLM to generate
three or four rephrasings of the user's question, runs a dense retrieval for
each, and unions the results. It costs one extra LLM call per query but
noticeably improves recall on questions that are phrased "unlike" the corpus.
"""

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import ConfigDict

from _corpus import TEST_QUESTIONS, load_documents


class MultiQueryRetriever(BaseRetriever):
    """Ask an LLM to rephrase the query, retrieve for each phrasing, union the results.

    langchain.retrievers.multi_query.MultiQueryRetriever is no longer
    importable from the langchain package (the whole langchain.retrievers
    namespace was retired — see 02_reranked_rag.py and 03_hybrid_rag.py for
    the same story) — this reimplements the same generate-then-union shape.
    """

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


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(
    documents=load_documents(),
    embedding=embeddings,
    collection_name="ch4_multiquery",
)

rewriter = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False)

retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=rewriter,
)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


answer_model = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | answer_model
    | StrOutputParser()
)

for q in TEST_QUESTIONS:
    print(f"Q: {q}")
    print(f"A: {chain.invoke(q).strip()}\n")
