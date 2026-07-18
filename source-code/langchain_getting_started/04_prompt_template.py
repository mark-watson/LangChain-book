"""Prompt templates and the LCEL pipe.

Instead of building message lists by hand, use ChatPromptTemplate. Instead of
threading the model call through Python glue code, use the `|` operator to
compose the prompt and the model into a single Runnable.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3.5:4b", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You answer in exactly one sentence."),
        ("human", "How do I {thing_to_do}?"),
    ]
)

chain = prompt | model

for task in ["get to the store", "hang a picture on the wall"]:
    result = chain.invoke({"thing_to_do": task})
    print(f"Q: How do I {task}?")
    print(f"A: {result.content.strip()}\n")
