# Chapter 21 — Structured extraction

Two scripts showing structured-output extraction from unstructured text using a local Ollama model.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

## Scripts

| Script | What it shows |
|---|---|
| `01_structured_predict.py` | `llm.structured_predict(SomeModel, prompt_template, **vars)` — the one-shot extraction pattern. |
| `02_batch_extract.py` | Extract structured records from multiple input texts in a loop; collect them into a list of validated Pydantic objects. |
