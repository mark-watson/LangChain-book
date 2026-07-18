"""RAG with a local Ollama model via LangChain 1.0.

Uses the 1.0 integration packages: langchain_ollama for the LLM and
embeddings, langchain_chroma for the vector store, and LCEL for the
retrieval chain. Requires 'ollama serve' to be running.

The old RetrievalQA chain is replaced by a compose-and-pipe LCEL chain.
"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

model = "mistral:v0.3"

# --- Build the index (can be reused) ---

loader = DirectoryLoader("../data/", glob="**/*.txt")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=100
)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OllamaEmbeddings(model=model),
    collection_name="ollama_rag",
)

# --- Reload from the vector store and query ---

retriever = vectorstore.as_retriever()

llm = ChatOllama(
    base_url="http://localhost:11434",
    model=model,
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("Ask a question: ")
    response = qa_chain.invoke(query)
    print(response)
