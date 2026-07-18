"""Recipe generation using RAG with LangChain 1.0.

Indexes cooking recipe text files and uses a local Ollama LLM to
generate new recipes based on the indexed ingredients.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

documents = [
    Document(page_content=p.read_text(), metadata={"source": str(p)})
    for p in Path("./text_data/").glob("**/*.txt")
]

vector = Chroma.from_documents(documents, embeddings, collection_name="recipes")

llm = ChatOllama(model="qwen3.5:4b", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)

retriever = vector.as_retriever()


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke("Create a new recipe using both Broccoli")
print(response)

response = chain.invoke("Create a recipe using Beans, Rice, and Chicken")
print(response)
