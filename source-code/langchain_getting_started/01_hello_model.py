"""The smallest useful LangChain 1.0 program: one chat model, one .invoke() call."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3.5:4b", temperature=0)

messages = [
    SystemMessage(content="You answer concisely, in one sentence."),
    HumanMessage(content="What is the capital of Arizona?"),
]

response = model.invoke(messages)

print(type(response).__name__)
print(response.content)
