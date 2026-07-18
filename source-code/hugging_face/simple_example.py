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
    model_kwargs={"temperature": 1e-6},
)

prompt = PromptTemplate.from_template("What year did {name} get elected as president?")

chain = prompt | hf_llm | StrOutputParser()

print(chain.invoke({"name": "George Bush"}))
