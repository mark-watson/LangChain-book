# Using local Hugging Face models

- `simple_example.py` — `langchain_huggingface.HuggingFacePipeline` (google/flan-t5-base) via LCEL.
- `hf_transformer_local.py` — a raw `transformers` pipeline (facebook/opt-iml-1.3b) wrapped as a LlamaIndex `CustomLLM`.

```console
$ uv sync
$ uv run simple_example.py
$ uv run hf_transformer_local.py
```

No API keys or Hugging Face account needed — both scripts download their models once (cached under `~/.cache/huggingface/hub`) and run entirely locally.
