"""Local HuggingFace transformer as a custom LLM with LlamaIndex 0.14.

Uses a small text-generation model from HuggingFace, wrapped so
LlamaIndex can use it. Demonstrates how to plug a custom LLM into
the LlamaIndex Settings system.
"""

import time

import torch
from transformers import pipeline

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

NUM_OUTPUT = 64


class CustomLLM(CustomLLM):
    """Wrap a HuggingFace text-generation pipeline as a LlamaIndex LLM."""

    model_name = "facebook/opt-iml-1.3b"
    pipeline_obj = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=512, num_output=NUM_OUTPUT, model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> str:
        prompt_length = len(prompt)
        response = self.pipeline_obj(prompt, max_new_tokens=NUM_OUTPUT)
        return response[0]["generated_text"][prompt_length:]

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        # Simple non-streaming implementation
        text = self.complete(prompt, **kwargs)
        from llama_index.core.llms import CompletionResponse
        yield CompletionResponse(text=text, delta=text)


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
