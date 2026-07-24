# Examples Using Hugging Face Open Source Models

Both examples in this chapter run entirely on your laptop: no Hugging Face account, no API key, no `HUGGINGFACEHUB_API_TOKEN`. Earlier editions of this chapter used LangChain's `HuggingFaceHub` wrapper, which calls Hugging Face's *hosted* inference endpoints and does need an account and a token. That wrapper is gone from LangChain 1.0; the current integration, `langchain_huggingface.HuggingFacePipeline`, downloads a model once and runs it locally with `transformers`, the same as every other local-model chapter in this book.

Set up tp run the two examples:

```console
$ cd source-code/hugging_face
$ uv sync
```

## Using LangChain as a Wrapper for a Local Hugging Face Pipeline

We will start with a short example using the prompt text support in LangChain. The following example is in the script **simple_example.py**:

```python
"""Use a HuggingFace model via LangChain 1.0 with a local pipeline.

Uses langchain_huggingface.HuggingFacePipeline (the 1.0 integration)
instead of the deprecated langchain.HuggingFaceHub / LLMChain.
"""

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

hf_llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    pipeline_kwargs={"temperature": 1e-6, "do_sample": True},
)

prompt = PromptTemplate.from_template("What year did {name} get elected as president?")

chain = prompt | hf_llm | StrOutputParser()

print(chain.invoke({"name": "George Bush"}))
```

`google/flan-t5-base` is a small (~250M parameter) instruction-tuned model, downloaded once on first run and cached under `~/.cache/huggingface/hub`. Two details worth getting right: `HuggingFacePipeline.from_model_id` takes separate `model_kwargs` and `pipeline_kwargs` dictionaries, and generation settings like `temperature` belong in the latter; passing `temperature` inside `model_kwargs` sends it to the model's constructor instead of the generation call and raises a `TypeError`. And `temperature` has no effect unless `do_sample=True` is also set; without it, generation is greedy and the temperature is silently ignored. The rest of the example is the LCEL shape you have seen throughout Part I: `prompt | hf_llm | StrOutputParser()`.

The output:

```console
$ uv run simple_example.py
1980
```

Wrong, for what it's worth (George W. Bush was elected in 2000), which is a fair reminder that a 250M-parameter model is not going to be a reliable source of facts. It is a demonstration of the plumbing, not a research assistant. By changing just the `model_id`, you can run this same pattern against any other local Hugging Face model that supports `text2text-generation` or `text-generation`.

## Creating a Custom LlamaIndex Hugging Face LLM Wrapper Class That Runs on Your Laptop

We will be downloading the Hugging Face model **facebook/opt-iml-1.3b**, a 2.6 gigabyte file. This model is downloaded the first time it is requested and is then cached in **~/.cache/huggingface/hub** for later reuse.

This example wraps a raw `transformers` pipeline as a LlamaIndex `CustomLLM`, the extension point LlamaIndex provides for exactly this: plugging in a model that has no first-party integration package.

```python
"""Local HuggingFace transformer as a custom LLM with LlamaIndex 0.14.

Uses a small text-generation model from HuggingFace, wrapped so
LlamaIndex can use it. Demonstrates how to plug a custom LLM into
the LlamaIndex Settings system.
"""

import time
from typing import ClassVar

import torch
from transformers import Pipeline, pipeline

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

NUM_OUTPUT = 64
MODEL_NAME = "facebook/opt-iml-1.3b"


class CustomLLM(CustomLLM):
    """Wrap a HuggingFace text-generation pipeline as a LlamaIndex LLM."""

    # ClassVar, not a Pydantic field: CustomLLM is a pydantic.BaseModel
    # under the hood, and pydantic 2 requires every plain class attribute
    # to be either an annotated field or explicitly marked as non-field.
    model_name: ClassVar[str] = MODEL_NAME
    pipeline_obj: ClassVar[Pipeline] = pipeline(
        "text-generation",
        model=MODEL_NAME,
        dtype=torch.bfloat16,
    )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=2048, num_output=NUM_OUTPUT,
                           model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        prompt_length = len(prompt)
        # min_new_tokens matters: opt-iml-1.3b is small and weakly
        # instruction-tuned, and on the longer RAG-style prompt LlamaIndex
        # builds ("Context information is below... answer the query"), it
        # will otherwise predict an immediate EOS and generate nothing.
        response = self.pipeline_obj(prompt, max_new_tokens=NUM_OUTPUT, min_new_tokens=8)
        text = response[0]["generated_text"][prompt_length:]
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        # Simple non-streaming implementation
        result = self.complete(prompt, **kwargs)
        yield CompletionResponse(text=result.text, delta=result.text)


time1 = time.time()

# Configure Settings with our custom LLM and local embeddings
Settings.llm = CustomLLM()
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# Load documents
documents = SimpleDirectoryReader("../data_small").load_data()
index = VectorStoreIndex.from_documents(documents)

time2 = time.time()
print(f"Time to load model and build index: {time2 - time1:.1f} seconds.")

query_engine = index.as_query_engine()

# Query and print response
response = query_engine.query("What is the definition of sport?")
print(response)

time3 = time.time()
print(f"Time for query/prediction: {time3 - time2:.1f} seconds.")
```

A `CustomLLM` in current LlamaIndex is a Pydantic model, which is why `model_name` and `pipeline_obj` need the `ClassVar` annotation; without it, Pydantic tries to treat the live `transformers.Pipeline` object as a validated field and raises at class-definition time. `complete()` must return a `CompletionResponse`, not a bare string; and `min_new_tokens=8` turns out to matter more than it looks; without it, this particular small, weakly instruction-tuned model will sometimes predict an immediate end-of-sequence token on LlamaIndex's longer "Context information is below... answer the query" prompt template and generate nothing at all.

When running on my Mac using Apple Silicon (the `mps` backend that PyTorch selects automatically), loading the model from cache and building the tiny index takes a couple of seconds, and the query itself well under a second:

```console
$ uv run hf_transformer_local.py
Time to load model and build index: 2.6 seconds.
Anything humans find amusing or entertaining. Answer Sport
Time for query/prediction: 0.8 seconds.
```

Exact wording will vary between runs and between machines; 1.3B-parameter models are small enough to be noticeably less polished than the models used elsewhere in this book, and this one is quoting fragments of the source document (`../data_small/sports.txt`) more than composing a fluent new sentence. That is a fair trade for a model this size, and it is still clearly grounded in the retrieved text rather than invented. If you want fluency as well as speed, `qwen3.5:4b` via Ollama (the model used everywhere else in this book) is both faster and much better at following instructions; this chapter uses a raw Hugging Face `transformers` pipeline specifically to show how to wrap a model that has no dedicated integration package.
