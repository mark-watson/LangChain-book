# Preface

I have been working in the field of artificial intelligence since 1982, and Large Language Models (LLMs) remain the most exciting practical AI technology I have used in my career. What has changed since the last edition of this book is *how* I use them. In the summer of 2026, most of my personal LLM work runs on my own laptop, on hardware I already own, using open source libraries that I can read and modify. When I do reach for a hosted model, I pay per token, not per seat, and I own my data at both ends of the call.

This edition of the book is rewritten around that stance. It is for **solo developers and small teams (two to five people)** who want to use [LangChain](https://github.com/langchain-ai/langchain) and [LlamaIndex](https://github.com/run-llama/llama_index) as they were originally developed: as MIT-licensed open source libraries for stitching LLMs into real applications, without adopting any of the commercial platforms that have grown up alongside them. All library versions have been updated to the latest versions as-of July 2026.

## Who this book is for

You will get the most out of the book if:

- You are comfortable writing Python but you are not a deep learning researcher. You want working code you can extend, not a survey of every framework.
- You value control over convenience. You want to be able to run the whole stack on a laptop or a small VPS, understand every dependency, and never be a captive customer of any vendor.
- You are happy to pay per token for LLM inference (Gemini, Fireworks.ai, OpenAI, or similar) or to run models locally via [Ollama](https://ollama.com). You are *not* happy to pay a per-seat or per-trace SaaS bill for observability, deployment, or "agent platforms."
- You are skeptical of platform lock-in and want abstractions that let you swap the LLM, the vector store, or the framework itself.

You may find this book useful, even if you work at a large enterprise that has already committed to LangSmith, LangGraph Cloud, or LlamaCloud, but you are not the reader I wrote it for.

## What this book covers, and what it deliberately does not

The book is in two parts. **Part I** covers LangChain 1.0 and LangGraph 1.0 (both released October 2025, both MIT-licensed, both currently stable). **Part II** covers the open source `llama-index-core` package and the LlamaIndex Workflows API. Every example runs on your laptop. Every example either uses a local Ollama model as its default or shows a local Ollama alternative alongside any hosted model call.

The book **deliberately does not cover** any of the following, because they are commercial platforms rather than open source libraries:

- LangSmith, LangSmith Deployment, LangSmith Engine, Sandboxes, Fleet, or the LLM Gateway
- LangChain Hub, LangGraph Cloud, or managed LangServe
- LlamaCloud, LlamaParse, or the `llama-cloud-services` Python package
- Any paid third-party API where a free, self-hostable alternative exists (so SerpAPI, Zapier, and the Google Knowledge Graph API are out; DuckDuckGo, public SPARQL endpoints, and self-hosted vector stores are in)

If the topic of a chapter would require any of the above to work, that chapter is not in the book.

## What you need to install

Everything in this book runs with:

- **Python 3.12** or newer
- **[uv](https://docs.astral.sh/uv/)** for Python package management (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **[Ollama](https://ollama.com)** for running local models
- A text editor and a terminal

Nothing else is strictly required. If you want to use a hosted model for a specific example, you will need an API key for whichever provider (Gemini, Fireworks.ai, or OpenAI); the book will tell you when. You will never need a LangSmith or LlamaCloud account.

For hardware: an Apple Silicon Mac (16 GB RAM or more), a mid-range Linux box with an NVIDIA GPU, or a Windows machine with WSL2 and a decent GPU all work well. My own writing setup is a Mac mini with an M2 SoC and 32 GB of memory, which is enough to run models up to about 30 B parameters comfortably.

## Where the code lives

For this edition, the example code lives in the same repository as the manuscript, in the `source-code/` directory, one subdirectory per chapter. The old external examples repo (`github.com/mark-watson/langchain-book-examples`) is being deprecated but this GitHub repository contains a PDF of this entire book that was saved before this rewrite.

The source code and manuscript files for the July 2026 version of this book are both found in [https://github.com/mark-watson/LangChain-book](https://github.com/mark-watson/LangChain-book).

Every chapter in the book references its code by relative path.

Each chapter's code directory has its own `pyproject.toml` with pinned library versions, so an example that worked when the chapter was written will keep working even after the upstream libraries change underneath it.

## Requests from the Author

This book will always be available to read free online at [https://leanpub.com/langchain/read](https://leanpub.com/langchain/read).

That said, I appreciate it when readers purchase my books because the income enables me to spend more time writing.

### Hire the Author as a Consultant

I am available for short consulting projects. Please see [https://markwatson.com](https://markwatson.com).

## About the Author

I have written over 20 books, I have over 50 US patents, and I have worked at interesting companies like Google, Capital One, SAIC, Mind AI, and others. You can find links for reading most of my recent books free on my web site [https://markwatson.com](https://markwatson.com). If I had to summarize my career the short take would be that I have had a lot of fun and enjoyed my work. I hope that what you learn here will be both enjoyable and help you in your work.

If you would like to support my work please consider purchasing my books on [Leanpub](https://leanpub.com/u/markwatson) and star my git repositories that you find useful on [GitHub](https://github.com/mark-watson?tab=repositories&q=&type=public). You can also interact with me on social media on [Mastodon](https://mastodon.social/@mark_watson) and [Twitter](https://twitter.com/mark_l_watson). I am also available as a consultant: [https://markwatson.com](https://markwatson.com).

## Book Cover

I used to live in Sedona, Arizona. I took the book cover photo in January 2023 from the street that I lived on.

## Acknowledgements

This picture shows me and my wife Carol who helps me with book production and editing.

{width: "50%"}
![Mark and Carol Watson](markcarol.jpg)

I would also like to thank the following readers who reported errors or typos in earlier editions of this book: Armando Flores, Peter Solimine, and David Rupp.
