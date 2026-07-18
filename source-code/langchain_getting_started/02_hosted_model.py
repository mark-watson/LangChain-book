"""Same program as 01_hello_model.py, against OpenAI instead of Ollama.

The only line that changes is the model constructor. Every other line —
imports, message objects, .invoke() call — is identical. That interchangeability
is the point of the LangChain abstraction.
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY in your environment to run this example.")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage(content="You answer concisely, in one sentence."),
    HumanMessage(content="What is the capital of Arizona?"),
]

response = model.invoke(messages)
print(response.content)
