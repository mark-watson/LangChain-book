"""Hybrid retrieval: dense embeddings for meaning + BM25 for exact terms.

Dense embeddings excel at "what does this mean" but can miss documents that
share the exact rare word or proper noun with the query. BM25 is the opposite:
great with exact matches, blind to paraphrases. Ensembling them with reciprocal
rank fusion gives you the best of both, and often changes the answer on queries
with proper nouns or technical terms.
"""

from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from _corpus import TEST_QUESTIONS, load_documents

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

retriever = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])

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
