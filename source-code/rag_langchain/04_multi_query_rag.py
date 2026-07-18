"""Multi-query retrieval: let the LLM rewrite the question several ways.

Users phrase questions in one specific way; the relevant passage in your
corpus may be phrased another. MultiQueryRetriever asks an LLM to generate
three or four rephrasings of the user's question, runs a dense retrieval for
each, and unions the results. It costs one extra LLM call per query but
noticeably improves recall on questions that are phrased "unlike" the corpus.
"""

from langchain.retrievers.multi_query import MultiQueryRetriever
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
    collection_name="ch4_multiquery",
)

rewriter = ChatOllama(model="qwen3.5:4b", temperature=0)

retriever = MultiQueryRetriever.from_llm(
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


answer_model = ChatOllama(model="qwen3.5:4b", temperature=0)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | answer_model
    | StrOutputParser()
)

for q in TEST_QUESTIONS:
    print(f"Q: {q}")
    print(f"A: {chain.invoke(q).strip()}\n")
