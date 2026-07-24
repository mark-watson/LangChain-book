# Running Local LLMs Using Ollama

Dear reader, I often use the Ollama app (download, documentation, and list of supported models at [https://ollama.ai](https://ollama.ai)). Ollama has a good command line interface and also runs a REST service that the examples in this chapter use.

Ollama works very well with Apple Silicon, systems with an NVIDIA GPU, and high end CPU-only systems. My Mac has an M-series SOC with 32G of internal memory which is suitable for running fairly large LLMs efficiently, but every example in this book runs fine with far less than that and I tested using a 16G MacBook Air.

Most of this chapter involves Python code examples using Ollama to run local LLMs. However the Ollama command line interface is useful for interactive experiments. Another useful development technique is to write prompts in individual text files like **p1.txt**, **p2.txt**, etc. and run a prompt (on macOS and Linux) using:

```console
$ ollama run qwen3.5:4b --think=false < p1.txt
```

And after the response is printed either stay in the Ollama REPL or type **/bye** to exit. `--think=false` is worth knowing about: `qwen3.5` is a "thinking" model, and without that flag the CLI prints several paragraphs of visible reasoning before the actual answer.

## Simple Use of a Local Model Using LangChain

We look at a simple example for asking questions and text completions using a local model. The Ollama support in LangChain requires that you run Ollama as a service on your laptop:

    ollama serve

I usually have several LLMs installed to experiment with, for example:

```console
 $ ollama list
NAME                         ID              SIZE      MODIFIED     
nomic-embed-text:latest      0a109f422b47    274 MB    2 days ago      
gemma4:e2b-it-qat            07ea59a47401    4.3 GB    4 days ago      
qwen3.5:4b                   2a654d98e6fb    3.4 GB    4 days ago      
gemma4:26b-mlx               c8656f50f0a6    17 GB     4 days ago      
laguna-xs-2.1:latest         a8562dfd0cad    20 GB     2 weeks ago     
gemma4:12b-mlx               117d0d84cf2a    7.7 GB    2 weeks ago     
gemma4:12b-it-qat            38044be4f923    7.2 GB    2 weeks ago     
qwen3.6:35b-a3b-nvfp4-48k    8c4e86c1307e    21 GB     3 weeks ago     
deepseek-v4-flash:cloud      ea027821675c    -         2 months ago    
qwen3.5:9b                   6488c96fa5fa    6.6 GB    4 months ago    
```

`qwen3.5:4b` is the model used throughout this book: small enough to run comfortably on a laptop, and it reliably supports tool calling, which several later chapters depend on.

Setup:

```console
$ cd source-code/ollama_langchain
$ uv sync
$ ollama pull qwen3.5:4b
```

Here is the file **ollama_langchain/qwen3.5-4b.py**:

```python
"""Chat with a local Ollama model via LangChain 1.0.

Uses langchain_ollama.ChatOllama (the 1.0 integration package).
Requires 'ollama serve' to be running in another terminal.
"""

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0,
    thinking=False,
)

s = llm.invoke("How much is 1 + 2?")
print(s.content)

s = llm.invoke("If Sam is 27, Mary is 42, and Jerry is 33, what are their age differences?")
print(s.content)
```

`langchain_ollama.ChatOllama` is the current integration package; the older `langchain.llms.Ollama` shown in previous editions of this book is gone from LangChain 1.0. `.invoke()` returns an `AIMessage`; `.content` is the text.

Here is the output:

```console
$ uv run qwen3.5-4b.py
1 + 2 = 3
Here are the age differences between each pair:

*   **Mary is older than Sam by 15 years** ($42 - 27 = 15$).
*   **Jerry is older than Sam by 6 years** ($33 - 27 = 6$).
*   **Mary is older than Jerry by 9 years** ($42 - 33 = 9$).
```

## Minimal Example Using Ollama for Retrieval Augmented Queries Against Local Documents

The following listing of file **ollama_langchain/rag_test.py** demonstrates creating a persistent embeddings datastore and reusing it. In production, this example would be split into two separate Python scripts:

- Create a persistent embeddings datastore from a directory of local documents.
- Open a persisted embeddings datastore and use it for queries against local documents.

One wrinkle that was not an issue in earlier editions: a chat model is not automatically an embedding model. `ollama show qwen3.5:4b` lists its capabilities as `completion`, `vision`, `tools`, and `thinking`, with no `embedding`, and asking it for a vector fails with "This server does not support embeddings." Embeddings come from a small dedicated model instead, `nomic-embed-text` (274 MB):

```console
$ ollama pull nomic-embed-text
```

```python
"""RAG with a local Ollama model via LangChain 1.0.

Uses the 1.0 integration packages: langchain_ollama for the LLM and
embeddings, langchain_chroma for the vector store, and LCEL for the
retrieval chain. Requires 'ollama serve' to be running.

The old RetrievalQA chain is replaced by a compose-and-pipe LCEL chain.

Chat models are not embedding models: qwen3.5:4b's declared Ollama
capabilities are completion/vision/tools/thinking, not embedding, so
asking it for a vector raises "This server does not support embeddings".
Embeddings come from a small dedicated model, nomic-embed-text, instead.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

model = "qwen3.5:4b"
embedding_model = "nomic-embed-text"

# --- Build the index (can be reused) ---


def load_text_documents(path: str, glob: str = "**/*.txt") -> list[Document]:
    """Load every matching text file under `path` into a Document."""
    return [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": str(p)})
        for p in sorted(Path(path).glob(glob))
    ]


data = load_text_documents("../data/")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=100
)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OllamaEmbeddings(model=embedding_model),
    collection_name="ollama_rag",
)

# --- Reload from the vector store and query ---

retriever = vectorstore.as_retriever()

llm = ChatOllama(
    base_url="http://localhost:11434",
    model=model,
    temperature=0,
    thinking=False,
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
```

`load_text_documents` is a small stand-in for the old `DirectoryLoader` (removed along with the rest of `langchain.document_loaders` in 1.0): read every `.txt` file under a path into a `Document`, nothing more. Everything downstream of that is the same LCEL shape you have seen since Chapter "RAG Patterns with LangChain": split, embed, store, retrieve, format, prompt, generate.

Here is an example using this script, against the same **../data/** corpus used elsewhere in the book. The first question is answered from general document content; the second uses a detail (a fictional economist's name) that was deliberately added to **economics.txt** and exists nowhere in `qwen3.5:4b`'s training data, so a correct answer proves the response came from retrieval, not from the model's memory:

```console
$ uv run rag_test.py
Ask a question: What is the Austrian School of Economics?
Based on the context provided, the Austrian School (also known as the Vienna School or Psychological School) is a school of economic thought that emphasizes the spontaneous organizing power of the price mechanism. Its economists advocate for strict enforcement of voluntary contractual agreements between agents and hold that commercial transactions should be based on subjective human choices because their complexity makes mathematical modeling difficult. The name derives from its predominantly Austrian founders and early supporters, including Carl Menger, Eugen von Böhm-Bawerk and Ludwig von [Mises].
Ask a question: Who says that economics is bullshit?
Pauli Blendergast
```

The second answer is terse, but it is the right name, and it could only have come from the local document: proof the retrieval half of the chain is doing real work rather than the model coasting on what it already knew.

## Wrap Up for Running Local LLMs Using Ollama

Most of my personal LLM experiments involve running models locally, even though hosted models available through Gemini, Fireworks.ai, OpenAI, and similar APIs are sometimes more capable. Ollama remains the easiest and most convenient way to run local models, either as a REST service or embedded directly in Python scripts, as in the two examples in this chapter.
