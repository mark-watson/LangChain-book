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
        return LLMMetadata(context_window=2048, num_output=NUM_OUTPUT, model_name=self.model_name)

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
