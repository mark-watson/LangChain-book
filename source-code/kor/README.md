# Structured extraction (formerly using the Kor library)

Earlier editions of this example used [Kor](https://github.com/eyurtsev/kor), a library
for generating LLM extraction prompts from a schema. Kor is essentially unmaintained
now, and LangChain 1.0's `.with_structured_output()` does the same job natively against
a Pydantic model, so `dates.py` uses that instead.

```console
$ uv sync
$ ollama pull qwen3.5:4b
$ uv run dates.py
```
