# Running Local LLMs Using Ollama


Dear reader, I often use the Ollama app (download, documentation, and list of supported models at [https://ollama.ai](https://ollama.ai)). Ollama has a good command line interface and also runs a REST service that the examples in this chapter use.

 Ollama works very well with Apple Silicon, systems with an NVIDIA GPU, and high end CPU-only systems. My Mac has a M2 SOC with 32G of internal memory which is suitable for running fairly large LLMs efficiently but most of the examples here run fine with 16G memory.

Most of this chapter involves Python code examples using Ollama to run local LLMs. However the Ollama command line interface is useful for interactive experiments. Another useful development technique is to write prompts in individual text files like **p1.txt**, **p2.txt**, etc. and run a prompt (on macOS and Linux) using:

```console
$ ollama run llama3:instruct < p1.txt
```
And after the response is printed either stay in the Ollama REPL or type **/bye** to exit.

## Simple Use of a local Mistral Model Using LangChain

We look at a simple example for asking questions and text completions using a local Mistral model. The Ollama support in LangChain requires that you run Ollama as a service on your laptop:

    ollama serve

Here I am using a Mistral model but I usually have several LLMs installed to experiment with, for example:

```console
 $ ollama list
NAME                       ID              SIZE      MODIFIED    
phi4-reasoning:plus        f0ad3edce8e4    11 GB     4 days ago     
qwen3:8b                   e4b5fd7f8af0    5.2 GB    6 days ago     
qwen3:30b                  2ee832bc15b5    18 GB     6 days ago     
gemma3:12b-it-qat          5d4fa005e7bb    8.9 GB    2 weeks ago    
gemma3:4b-it-qat           d01ad0579247    4.0 GB    2 weeks ago    
gemma3:27b-it-qat          29eb0b9aeda3    18 GB     2 weeks ago    
openthinker:latest         4e61774f7d1c    4.7 GB    3 weeks ago    
deepcoder:latest           12bdda054d23    9.0 GB    3 weeks ago    
llava:7b                   8dd30f6b0cb1    4.7 GB    3 weeks ago    
qwq:latest                 38ee5094e51e    19 GB     3 weeks ago    
llama3.2:1b                baf6a787fdff    1.3 GB    3 weeks ago    
mistral-small:latest       8039dd90c113    14 GB     3 weeks ago    
granite3-dense:latest      5c2e6f3112f4    1.6 GB    3 weeks ago    
phi4-mini:latest           78fad5d182a7    2.5 GB    3 weeks ago    
reader-lm:latest           33da2b9e0afe    934 MB    3 weeks ago    
smollm2:latest             cef4a1e09247    1.8 GB    3 weeks ago    
unsloth_AZ_1B:latest       0b3006d8395a    807 MB    3 weeks ago    
nomic-embed-text:latest    0a109f422b47    274 MB    3 weeks ago    
cogito:32b                 0b4aab772f57    19 GB     3 weeks ago    
deepseek-r1:32b            38056bbcbb2d    19 GB     3 weeks ago    
deepseek-r1:8b             28f8fd6cdc67    4.9 GB    3 weeks ago    
llama3.2:latest            a80c4f17acd5    2.0 GB    3 weeks ago    
qwen2.5:14b                7cdf5a0187d5    9.0 GB    3 weeks ago    
```

Here is the file **ollama_langchain/test.py**:

```python
# requires "ollama serve" to be running in a terminal

from langchain.llms import Ollama

llm = Ollama(
    model="qwen3:8b",
    verbose=False,
)

s = llm("how much is 1 + 2?")
print(s)

s = llm("If Sam is 27, Mary is 42, and Jerry is 33, what are their age differences?")
print(s)
```

Here is the output:

```console
$ python test.py
1 + 2 = 3.

To calculate their age differences, we simply subtract the younger person's age from the older person's age. Here are the calculations:
- Sam is 27 years old, and Mary is 42 years old, so their age difference is 42 - 27 = 15 years.
- Mary is 42 years old, and Jerry is 33 years old, so their age difference is 42 - 33 = 9 years.
- Jerry is 33 years old, and Sam is 27 years old, so their age difference is 33 - 27 = 6 years.
```


## Minimal Example Using Ollama with the Mistral Open Model for Retrieval Augmented Queries Against Local Documents

The following listing of file **ollama_langchain/rag_test.py** demonstrates creating a persistent embeddings datastore and reusing it. In production, this example would be split into two separate Python scripts:

- Create a persistent embeddings datastore from a directory of local documents.
- Open a persisted embeddings datastore and use it for queries against local documents.

Creating a local persistent embeddings datastore for the example text files in **../data/*.txt** takes about 90 seconds on my Mac Mini.
 
```python
# requires "ollama serve" to be running in another terminal

from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.directory import DirectoryLoader

# Create index (can be reused):

loader = DirectoryLoader('../data', glob='**/*.txt')

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

persist_directory = 'cache'

vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OllamaEmbeddings(model="mistral:instruct"),
                               persist_directory=persist_directory)

vectorstore.persist()

# Try reloading index from disk and using for search:

persist_directory = 'cache'

vectorstore = Chroma(
  persist_directory=persist_directory,
  embedding_function=OllamaEmbeddings(model="mistral:instruct")
)

llm = Ollama(base_url="http://localhost:11434",
             model="mistral:instruct",
             verbose=False,
            )

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            verbose=True,)

while True:
    query = input("Ask a question: ")
    response = qa_chain(query)
    print(response['result'])
```

Here is an example using this script. The first question uses the innate knowledge contained in the Mistral-7B model while the second question uses the text files in the directory **../data** as local documents. The test input file **economics.txt** has been edited to add the name of a fictional economist. I added this data to show that the second question is answered from the local document store.

```console
$ python rag_test.py 
> Entering new RetrievalQA chain...

> Finished chain.

11 + 2 = 13
Ask a question: Who says that economics is bullshit?


> Entering new RetrievalQA chain...

> Finished chain.

Pauli Blendergast, an economist who teaches at the University of Krampton Ohio, is known for saying that economics is bullshit.
```

## Wrap Up for Running Local LLMs Using Ollama

As I write this chapter in December 2023 most of my personal LLM experiments involve running models locally on my Mac mini (or sometimes in Google Colab) even though models available through OpenAI, Anthropic, etc. APIs are more capable. I find that the Ollama project is currently the easiest and most convenient way to run local models as REST services or embedded in Python scripts as in the two examples here.

