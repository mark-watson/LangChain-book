# Extraction from text

Structured data extraction with LangChain 1.0's `.with_structured_output()`.

- `person_data.py` — extract one `PersonData` record from a paragraph of prose.
- `csv_to_json.py` — extract a list of `PersonRecord`s from a whole (messily formatted) CSV file in one call.

```console
$ uv sync
$ ollama pull qwen3.5:4b
$ uv run person_data.py
$ uv run csv_to_json.py
```
