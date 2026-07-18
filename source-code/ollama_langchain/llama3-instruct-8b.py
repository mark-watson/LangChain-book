"""Chat with a local Ollama model via LangChain 1.0.

Uses langchain_ollama.ChatOllama (the 1.0 integration package).
Requires 'ollama serve' to be running in another terminal.
"""

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3:instruct",
    temperature=0,
)

s = llm.invoke("If Sam is 27, Mary is 42, and Jerry is 33, what are their age differences? Be concise")
print(s.content)
