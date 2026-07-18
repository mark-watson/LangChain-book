# Preface

I have been working in the field of artificial intelligence since 1982 and without a doubt Large Language Models (LLMs) like Genini 2.5 Pro and o3 represent the greatest advance in practical AI technology that I have experienced. Infrastructure projects like LangChain and LlamaIndex make it simpler to use LLMs and provide some level of abstraction to facilitate switching between LLMs. This book will use the [LangChain](https://github.com/langchain-ai/langchain) and [LlamaIndex](https://github.com/run-llama/llama_index) projects along with the OpenAI GPT-40 APIs, and local models run on your computer using Ollama to solve a series of interesting problems.

As I write this new edition in May 2025, I mostly run local LLMs but I still use OpenAI APIs, as well as well as APIs from Anthropic for the Claude 3.7 model and APIs from the French company Mistral.  Most of the newest examples are in the chapter on RAG and agents.

Harrison Chase started the LangChain project in October 2022 and as I wrote the first edition of this book in February 2023, the GitHub repository for LangChain 171 contributors and as I write this new edition LangChain has 2100+ contributors on GitHub. Jerry Liu started the GPT Index project (recently renamed to LlamaIndex) at the end of 2022 and the GitHub Repository for LlamaIndex [https://github.com/jerryjliu/gpt_index](https://github.com/run-llama/llama_index) currently has 54 contributors.

The GitHub repository for examples in this book is [https://github.com/mark-watson/langchain-book-examples.git](https://github.com/mark-watson/langchain-book-examples.git). Please note that I usually update the code in the examples repository fairly frequently for library version updates, etc.

While the documentation and examples online for LangChain and LlamaIndex are excellent, I am still motivated to write this book to solve interesting problems that I like to work on involving information retrieval, natural language processing (NLP), dialog agents, and the semantic web/linked data fields. I hope that you, dear reader, will be delighted with these examples and that at least some of them will inspire your future projects.

## Requests from the Author

This book will always be available to read free online at [https://leanpub.com/langchain/read](https://leanpub.com/langchain/read).

That said, I appreciate it when readers purchase my books because the income enables me to spend more time writing.

### Hire the Author as a Consultant

I am available for short consulting projects. Please see [https://markwatson.com](https://markwatson.com).

## Newer Commercial LangChain Integrations That Are Not Covered In This Book

LangChain has increasingly integrated proprietary services and developed features dependent on commercial APIs. Some of these services include:

- LangSmith: A commercial platform tied to LangChain that focuses on debugging, logging, and testing. It helps manage and improve prompt artifacts, and integrates with the LangChain Hub.
- LangChain Hub: This hub is designed as a repository where developers can share prompts, chains, agents, and more. It allows collaboration on LLM artifacts, making it easier for teams to build and improve applications. It supports private, organization-specific repositories alongside the public ones. Users can interact with prompts in a playground using their own API keys (for services like OpenAI or Anthropic) and can download or upload new artifacts using an SDK. LangChain is also extending this hub to include support for chains and agents 
	
I limit the examples in this book to the core LangChain library. Most of the examples have been updated to LangChain version 0.3.25

## Deprecated Book Examples

As I add new examples to this book I sometimes remove older examples that no longer seem relevant. The book text can be found in the GitHub repository for the code examples. Check out [https://github.com/mark-watson/langchain-book-examples/tree/main/CHAPTERS_and_CODE_no_longer_in-book](https://github.com/mark-watson/langchain-book-examples/tree/main/CHAPTERS_and_CODE_no_longer_in-book).

## LLM Hallucinations and RAG, Summarization, Structured Data Conversion, and Fact/Relationship Extraction LLM Applications

When we chat with LLMs and rely on the innate information they provide the results often contain so-called “hallucinations” that can occur when a model does not know an answer so it generates plausible sounding text. Many of the examples in this book are Retrieval Augmented Generation (RAG) style applications. RAG applications provide context text with chat prompts or queries so LLMs prioritize using the provided information to answer questions. The latest Google Gemini model has an input context size of one million tokens and even smaller models you can run on your own computer using tools like [Ollama](https://ollama.ai) often support up to 128K context size.

Beyond RAG, other effective LLM applications effective at minimizing hallucinations include text summarization, structured data conversion and extraction, and fact/relationship extraction. In text summarization, LLMs condense lengthy documents into concise summaries, focusing on core information. Structured data conversion and extraction involve transforming unstructured text into organized formats or pinpointing specific information.  Fact/relationship extraction allows LLMs to identify and understand key connections within text, making them less prone to misinterpretation and hallucinations.

## Comparing LangChain and LlamaIndex

Since I wrote the earlier editions of this book, LangChain has matured into a comprehensive platform—spanning open-source libraries and the managed agent orchestration service LangGraph—enabling modular pipelines, rich agent frameworks, and production-grade deployment tools. Meanwhile, LlamaIndex has evolved into an enterprise-focused suite (branded as LlamaCloud) offering GenAI-native parsing, schema-driven data extraction, knowledge management, and Agentic Document Workflows (ADW) for orchestrating complex, multi-step document processes. Choose LangChain when you need end-to-end flexibility in chaining, agents, and callback instrumentation with a clear path to production via LangGraph; choose LlamaIndex when your priority is high-performance indexing, advanced parsing, and agentic workflows over large enterprise datasets.

**LangChain**
- Composability & Chains: Exposes primitives for prompts, chains, agents, and callbacks, allowing developers to stitch together LLM calls, embeddings, and tools into custom pipelines.
- Agentic Orchestration (LangGraph): Introduced LangGraph in January 2024 to deploy and manage agents at scale, define tool-use policies, and monitor runtime performance.
- Extensive API Reference: The Python API catalogs 200+ classes (e.g., AgentExecutor, BaseSingleActionAgent) and offers deep support for multi-action agents, memory modules, and output parsers.
- Documentation Refresh: In May 2024, LangChain released v0.2 with versioned docs, a flatter structure, and consolidated tutorials, how-to guides, conceptual overviews, and API references.

**LlamaIndex**
- GenAI-Native Parsing: LlamaParse handles PDFs, tables, diagrams, and multimodal content, automatically chunking and formatting for optimal LLM ingestion.
- Schema-Driven Extraction: Provides a data-extraction engine to define output schemas, transforming unstructured documents into structured JSON or tabular records.
- Knowledge Management: Connectors to databases, file systems, and cloud stores feed into an indexed knowledge base accessible via vector embeddings or keyword search.
- Agentic Document Workflows (ADW): Introduced early 2025, ADW maintains state across steps, applies business rules, coordinates components, and executes actions based on document content.

 | Capability | LangChain | LlamaIndex |
|:-----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|
| **Prompting & Chains** | Highly customizable, chain-of-thought, branching, and callback hooks ([LangChain Concepts][1]) | Limited to indexed data retrieval plus templating |
| **Embedding & Vector Stores** | Built-in support for common stores (Chroma, Pinecone, Weaviate) ([LangChain How-To][2]) | Optimized local and cloud vector indices |
| **Agents & Tools** | Multi-action agents, toolkits, memory modules ([LangChain API Reference][3]) | Agent framework focused on knowledge assistants |
| **Data Ingestion** | Manual chunking or external utilities | Automated parsing + schema extraction |
| **Production Deployment** | LangGraph platform for scaling agents | LlamaCloud managed service |

[1]: https://python.langchain.com/docs/concepts/?utm_source=chatgpt.com
[2]: https://python.langchain.com/docs/how_to/?utm_source=chatgpt.com
[3]: https://python.langchain.com/api_reference/langchain/index.html?utm_source=chatgpt.com


## About the Author

I have written over 20 books, I have over 50 US patents, and I have worked at interesting companies like Google, Capital One, SAIC, Mind AI, and others. You can find links for reading most of my recent books free on my web site [https://markwatson.com](https://markwatson.com). If I had to summarize my career the short take would be that I have had a lot of fun and enjoyed my work. I hope that what you learn here will be both enjoyable and help you in your work.

If you would like to support my work please consider purchasing my books on [Leanpub](https://leanpub.com/u/markwatson) and star my git repositories that you find useful on [GitHub](https://github.com/mark-watson?tab=repositories&q=&type=public). You can also interact with me on social media on [Mastodon](https://mastodon.social/@mark_watson) and [Twitter](https://twitter.com/mark_l_watson). I am also available as a consultant: [https://markwatson.com](https://markwatson.com).

## Book Cover

I live in Sedona, Arizona. I took the book cover photo in January 2023 from the street that I live on.

## Acknowledgements

This picture shows me and my wife Carol who helps me with book production and editing.

{width: "50%"}
![Mark and Carol Watson](markcarol.jpg)

I would also like to thank the following readers who reported errors or typos in this book: Armando Flores, Peter Solimine, and David Rupp.


## Requirements for Running and Modifying Book Examples

I show full source code and a fair amount of example output for each book example so if you don't want to get access to some of the following APIs then you can still read along in the book.

To use OpenAI's GPT-3 and ChatGPT models you will need to sign up for an API key (free tier is OK) at [https://openai.com/api/](https://openai.com/api/) and set the environment variable **OPENAI_API_KEY** to your key value.

You will need to get an API key for examples using **Google's Knowledge Graph APIs**.

Reference: [Google Knowledge Graph APIs](https://cloud.google.com/enterprise-knowledge-graph/docs/search-api).

The example programs using Google's Knowledge Graph APIs assume that you have the file **~/.google_api_key** in your home directory that contains your key from [https://console.cloud.google.com/apis](https://console.cloud.google.com/apis).

You will need to install **SerpApi for examples integrating web search**. Please reference: [PyPi project page](https://pypi.org/project/google-search-results/).

You can sign up for a free non-commercial 100 searches/month account with an email address and phone number at [https://serpapi.com/users/welcome](https://serpapi.com/users/welcome).

You will also need [Zapier](https://zapier.com) account for the GMail and Google Calendar examples.

After reading though this book, you can review the website [LangChainHub](https://github.com/hwchase17/langchain-hub) which contains prompts, chains and agents that are useful for building LLM applications.

## Issues and Workarounds for Using the Material in this Book

The libraries that I use in this book are frequently updated and sometimes the documentation or code links change, invalidating links in this book. I will try to keep everything up to date. Please report broken links to me.

In some cases you will need to use specific versions or libraries for some of the code examples.


Because the Python code listings use colorized text you may find that copying code from this eBook may drop space characters. All of the code listings are in the GitHub repository for this book so you should clone the repository to experiment with the example code.
