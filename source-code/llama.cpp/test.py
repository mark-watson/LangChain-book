"""Use a local GGUF model with LangChain 1.0 via llama-cpp-python.

The old langchain.llms.LlamaCpp is replaced by
langchain_community.llms.LlamaCpp (still the integration path in 1.0).
Make sure the model path is correct for your system.
"""

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/markw/llama.cpp/models/openassistant-llama2-13b-orca-8k-3319.Q5_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=True,
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "question": "If Mary is 30 years old and Bob is 25, who is older and by how much?"
})
print(result)
