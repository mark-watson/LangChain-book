# More Useful Libraries for Working with Unstructured Text Data

Here we look at replacements for two libraries that were useful in earlier editions of this book (EmbedChain and Kor) and are no longer a good recommendation. Both illustrate a pattern worth knowing about generally: a small wrapper library earns its keep only as long as it stays ahead of the framework it wraps. Both of these were eventually overtaken by their underlying frameworks growing the exact convenience they provided.

## What Used to Be EmbedChain: "Query Your Own Data" with LlamaIndex

Taranjeet Singh's EmbedChain library ([https://github.com/embedchain/embedchain](https://github.com/embedchain/embedchain)) used to be a nice wrapper that simplified writing "query your own data" applications by choosing good defaults on top of LangChain. It has since been folded into a commercial memory-layer product, and the pattern it made convenient (point a loader at a directory, build an index, query it) is now just as easy directly in LlamaIndex, with the added benefit of running entirely on local models instead of requiring an OpenAI key.

I will show the same example I ran in earlier editions: searching the contents of some of the books and technical writing I have on my laptop. The code is `source-code/embedchain_test/`. Setup:

```console
$ cd source-code/embedchain_test
$ uv sync
$ ollama pull qwen3.5:4b
```

`process_pdfs.py` builds the index (the filename is kept for continuity with the earlier edition; the bundled sample data in `data/` is plain `.txt`, and `SimpleDirectoryReader` handles real PDFs the same way, no code change required):

```python
"""Process PDFs with LlamaIndex 0.14 (replaces embedchain).

Uses LlamaIndex's SimpleDirectoryReader to load PDF files and build
a vector index. Local models only - no external API keys required.
"""

import os

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

my_books_dir = "./data/"

if os.path.isdir(my_books_dir):
    documents = SimpleDirectoryReader(my_books_dir).load_data()
    print(f"Loaded {len(documents)} documents from {my_books_dir}")

    index = VectorStoreIndex.from_documents(documents)
    print("Index built successfully.")
else:
    print(f"Directory {my_books_dir} does not exist. Nothing to process.")
```

```console
$ uv run process_pdfs.py
Loaded 2 documents from ./data/
Index built successfully.
```

`app.py` queries that index with three questions:

```python
"""Embedchain replacement: simple RAG with LlamaIndex 0.14.

The embedchain library is unmaintained. This example replaces it with
LlamaIndex's SimpleDirectoryReader + VectorStoreIndex, which provides
the same add-and-query functionality using local models.
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=180.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Build index from PDF data directory
import os

data_dir = "./data/"
if os.path.isdir(data_dir):
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
else:
    # Fallback: use a simple in-memory document
    from llama_index.core import Document

    index = VectorStoreIndex.from_documents([Document(text="No data directory found.")])

# similarity_top_k=1: the default of 2 triggers a second, sequential
# "refine" LLM call per query. With whole-book-length source files (this
# directory's data/ is ~130K words) that second pass can take minutes on
# a small local model; one well-matched chunk is enough for these questions.
query_engine = index.as_query_engine(similarity_top_k=1)


def test(q):
    print(q)
    print(query_engine.query(q), "\n")


test("How can I iterate over a list in Haskell?")
test("How can I edit my Common Lisp files?")
test("How can I scrape a website using Common Lisp?")
```

The output looks like:

```console
$ uv run app.py
How can I iterate over a list in Haskell?
One way to iterate over a list in Haskell involves using list comprehensions where variables bind values directly from the lists provided within brackets. By utilizing syntax such as `x <- [0..3]`, you can process elements sequentially, allowing multiple variables to be iterated simultaneously for generating combinations or filtered results.

How can I edit my Common Lisp files?
You can use a text editor such as Emacs or Vi to edit Common Lisp files. For the Vi editor, you should enter vi followed by your filename (for example `vi nested.lisp`) and then type `:set sm` after running it; this configuration indicates matching opening parentheses whenever a closing parenthesis is typed.

For users choosing Emacs, configure their `.emacs` file or `_emacs` file in Windows to automatically recognize specific extensions for Lisp mode.

How can I scrape a website using Common Lisp?
Common Lisp libraries can be managed by cloning repositories into the `~/quicklisp/local-projects/` directory and running a Makefile target `make fetch`. Detailed methods for scraping websites specifically are not elaborated in this section regarding example distribution and installation procedures.
```

Two things worth noticing. First, this took a while: roughly a minute and a half per question, because `data/` here is two entire books (about 130,000 words combined), a much bigger corpus than the few-paragraph examples used elsewhere in this book, and a small local model reading a full retrieved chunk takes real time. Second, the third answer is honest about its limits rather than confidently wrong: it does not invent a scraping library or a `wget` command, it says the retrieved section does not cover that topic. A weaker or more eagerly fine-tuned model might have filled that gap with something plausible-sounding and false; watching a small local model decline to do that is, in its own way, reassuring.

## What Used to Be Kor: Structured Extraction with `.with_structured_output()`

The Kor library, written by Eugene Yurtsev, used to be a nice way to get an LLM to extract structured data from unstructured text; it generated the prompt boilerplate for you from a schema you defined. Kor itself is essentially unmaintained today, for a good reason: LangChain 1.0's `.with_structured_output()`, covered in the Extraction chapter, now does natively and more reliably what Kor used to paper over with prompt engineering. There is no need for a separate library or a separate schema language: a Pydantic model is the schema.

Here is the same date-extraction task Kor used to handle, in `source-code/kor/dates.py`:

```python
"""Extract dates from text using LangChain 1.0 structured output.

The old kor library is deprecated. LangChain 1.0's .with_structured_output()
replaces it with native Pydantic-based extraction.
"""

from pprint import pprint

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class DateExtraction(BaseModel):
    """Dates found in the text, formatted as 'January 12, 2023'."""

    month: str = Field(description="The month of the date found in the text")
    full_date: str | None = Field(
        description="The full date in 'Month Day, Year' format if available",
        default=None,
    )


llm = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False)
structured_llm = llm.with_structured_output(DateExtraction)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract dates from the text. Format dates as 'Month Day, Year'."),
    ("human", "Extract any dates from this text:\n\n{text}"),
])

chain = prompt | structured_llm

result = chain.invoke({"text": "I will go to California May 1, 2024"})
pprint(result)
```

Sample output:

```console
$ uv run dates.py
DateExtraction(month='May', full_date='May 1, 2024')
```

A validated Pydantic object, not a dict you have to trust and unpack by hand: `result.month` and `result.full_date` are typed attributes, and if the model's response does not fit the schema, this raises instead of silently handing you something malformed. That reliability, built into the framework itself, is exactly what a wrapper library like Kor used to add on top.
