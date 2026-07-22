# Swapping Between Local and Commercial Models

In the overview chapter I made a promise: write your applications so you can move between a local model and a hosted one without rewriting anything but a single line. This short chapter delivers on that. The previous chapter already showed the trick once: the OpenAI version of `01_hello_model.py` differed from the Ollama version by exactly one constructor call. Here I generalize that to all four providers this book cares about (local **Ollama**, **OpenAI**, **Anthropic (Claude)**, and **Google (Gemini)**) in both LangChain and LlamaIndex.

The advice is the same in every framework and worth stating before any code:

- **Develop locally, escalate deliberately.** I prototype against `qwen3.5:4b` because iteration is instant and I pay for nothing. When I have concrete evidence a task needs more capability, I flip to a hosted model. Vibes are not evidence; point to a specific failure the local model produces.
- **Write the rest of your code against the abstract model type.** Accept a model object as a parameter; do not import `ChatOpenAI` (or any one provider) in your business logic. Everything downstream (`.invoke`, `.stream`, prompt pipes, tool binding, structured output) is identical no matter who serves the tokens.
- **Keep embeddings and rerankers local.** Swapping the *chat* model does not mean paying for embeddings. A local HuggingFace embedding model runs on your laptop in milliseconds; there is rarely a reason to send that work to an API.
- **Keys live in environment variables, never in source.** Each hosted provider reads one variable: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`. Ollama needs no key.

One caveat about the model names below: `gpt-4o-mini`, `claude-sonnet-5`, and `gemini-2.5-flash` are current as I write this in mid-2026 and *will* drift. Treat them as placeholders and check each provider's model list for the current identifiers.

## Switching providers in LangChain

Install the provider packages you want (the Ollama package is already a dependency of the book's examples):

```console
$ uv add langchain-openai langchain-anthropic langchain-google-genai
```

LangChain 1.0 ships a helper, `init_chat_model`, made for exactly this problem. Give it a `provider:model` string and it returns the right `BaseChatModel` subclass, importing the provider package lazily:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("ollama:qwen3.5:4b", temperature=0)   # local default
# model = init_chat_model("openai:gpt-4o-mini")
# model = init_chat_model("anthropic:claude-sonnet-5")
# model = init_chat_model("google_genai:gemini-2.5-flash")
```

The string is split on the first colon, so `qwen3.5:4b`, which itself contains a colon, is parsed correctly as the model name. Everything you learned in the previous chapter now works unchanged against `model`, whichever line is uncommented:

```python
model.invoke("What is the capital of Arizona?")
(prompt | model | StrOutputParser()).invoke({...})
model.bind_tools([add, multiply]).invoke("What is 137 * 24?")
model.with_structured_output(Country).invoke("Facts about Japan.")
```

Because `init_chat_model` takes a plain string, the whole provider choice can come from configuration. This one line lets you flip providers with an environment variable and zero code changes:

```python
import os

model = init_chat_model(os.getenv("CHAT_MODEL", "ollama:qwen3.5:4b"), temperature=0)
```

If you prefer explicit constructors (for editor autocompletion, or to pass provider-specific arguments), the classes are `ChatOllama` (`langchain_ollama`), `ChatOpenAI` (`langchain_openai`), `ChatAnthropic` (`langchain_anthropic`), and `ChatGoogleGenerativeAI` (`langchain_google_genai`). All four satisfy the same `BaseChatModel` protocol, so they are interchangeable wherever a model is expected.

## Switching providers in LlamaIndex

LlamaIndex keeps each integration in its own `llama-index-llms-*` package:

```console
$ uv add llama-index-llms-openai llama-index-llms-anthropic llama-index-llms-google-genai
```

There is no `init_chat_model` equivalent, but LlamaIndex centralizes the choice in a different way: set `Settings.llm` once at startup and every index, query engine, and agent reads from it. Uncomment whichever line you want:

```python
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)  # local default
# Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
# Settings.llm = Anthropic(model="claude-sonnet-5")
# Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
```

To make the provider configurable, wrap the choice in a small factory driven by an environment variable and assign its result to `Settings.llm`:

```python
import os

def make_llm():
    return {
        "ollama":    lambda: Ollama(model="qwen3.5:4b", request_timeout=120.0),
        "openai":    lambda: OpenAI(model="gpt-4o-mini", temperature=0),
        "anthropic": lambda: Anthropic(model="claude-sonnet-5"),
        "gemini":    lambda: GoogleGenAI(model="gemini-2.5-flash"),
    }[os.getenv("LLM_PROVIDER", "ollama")]()

Settings.llm = make_llm()
```

Note what does *not* change: `Settings.embed_model` stays on a local `HuggingFaceEmbedding` model regardless of which chat provider you pick, the same `BAAI/bge-small-en-v1.5` default used throughout Part II. Match the LLM to the task; keep embeddings local.

## Wrap up

Both frameworks reduce the “question of which LLM provider?" to a single, late-bound decision. In LangChain that decision is a `provider:model` string handed to `init_chat_model`; in LlamaIndex it is one assignment to `Settings.llm`. In both cases the rest of your application is written against an abstract model interface and never mentions a specific vendor.

That is not just tidy engineering; it is insurance. The commercial LLM landscape is turbulent, and any provider you depend on today may change its pricing, its models, its business  plans, or go out of business entirely. Keep the provider on the end of an environment variable, keep your embeddings local, and you can follow the best price-to-quality ratio from quarter to quarter without touching the code that actually does your work.
