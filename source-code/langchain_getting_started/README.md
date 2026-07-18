# Chapter 1 — LangChain 1.0 in one hour

Working examples for the first chapter of Part I. Every script runs end-to-end on your laptop.

## Setup

```console
$ uv sync
```

This creates a `.venv/` in this directory and installs the pinned versions from `pyproject.toml`.

## Prerequisites

- **Ollama running locally.** Start it with `ollama serve` in a separate terminal, and pull the tool-capable model the examples default to:

  ```console
  $ ollama pull qwen3.5:4b
  ```

- **(Optional) An OpenAI API key** for `02_hosted_model.py`. Set `OPENAI_API_KEY` in your shell if you want to run that one.

## Scripts

| Script | What it shows |
|---|---|
| `01_hello_model.py` | First `ChatOllama.invoke()` call. |
| `02_hosted_model.py` | Same call, but against `ChatOpenAI`. Illustrates provider-swap. |
| `03_stream_and_batch.py` | `.stream()` for token-by-token output, `.batch()` for parallel calls. |
| `04_prompt_template.py` | `ChatPromptTemplate` piped into a model with the LCEL `|` operator. |
| `05_output_parser.py` | Adding `StrOutputParser` and a Pydantic structured-output parser. |
| `06_tool_binding.py` | `.bind_tools()` to let the model call your Python functions. |

Run any one of them with:

```console
$ uv run 01_hello_model.py
```
