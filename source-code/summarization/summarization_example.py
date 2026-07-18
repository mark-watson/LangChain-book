"""Summarize a text file using a local Ollama model via LangChain 1.0.

Reads the prompt template from prompt.txt, fills in the input text,
and sends it to a local LLM.
"""

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# Read the prompt template
prompt_template = Path("prompt.txt").read_text()

# Read the input text
input_text = Path("../data/economics.txt").read_text()
prompt = prompt_template.replace("input_text", input_text)

# Use a local Ollama model
llm = ChatOllama(model="qwen3.5:4b", temperature=0)

response = llm.invoke([HumanMessage(content=prompt)])
print(response.content)
