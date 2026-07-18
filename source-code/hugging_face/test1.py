"""Local HuggingFace transformer + embeddings with LlamaIndex 0.14.

Same approach as hf_transformer_local.py but with a larger corpus
and explicit embedding-model configuration via Settings.
"""

import time

import torch
from transformers import pipeline

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


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
        return LLMMetadata(context_window=512, num_output=200, model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> str:
        prompt_length = len(prompt)
        response = self.pipeline_obj(prompt, max_new_tokens=200)
        return response[0]["generated_text"][prompt_length:]

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        text = self.complete(prompt, **kwargs)
        from llama_index.core.llms import CompletionResponse
        yield CompletionResponse(text=text, delta=text)


time1 = time.time()

# Configure Settings with our custom LLM and local embeddings
Settings.llm = CustomLLM()
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

print("Done configuring Settings")

# Build index
documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)
print("Done building index")

query_engine = index.as_query_engine()
print("Done creating query engine")


def query(query_string):
    response = query_engine.query(query_string)
    print(response)
    return response


query("what is the definition of Chemistry?")
query("what are the benefits of sports?")
query("why study economics?")
