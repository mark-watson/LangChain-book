# Running local LLMs with Ollama

- `qwen3.5-4b.py` — two simple `ChatOllama.invoke()` calls, no chain.
- `rag_test.py` — RAG over `../data/*.txt` using `nomic-embed-text` for embeddings and `qwen3.5:4b` for generation, composed with LCEL.

```console
$ uv sync
$ ollama pull qwen3.5:4b
$ ollama pull nomic-embed-text
$ uv run qwen3.5-4b.py
$ uv run rag_test.py
```
