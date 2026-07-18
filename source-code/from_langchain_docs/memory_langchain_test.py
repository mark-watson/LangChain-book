"""Conversation memory in LangChain 1.0 using LCEL.

The old LLMChain + ConversationBufferMemory pattern is replaced by
a Runnable that appends messages to a list and passes the full
history to the model each turn.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3.5:4b", temperature=0)

system_msg = SystemMessage(content="You are a chatbot having a conversation with a human.")

history = ChatMessageHistory()

messages = [system_msg] + history.messages

for user_input in [
    "Hi there my friend. What is your name?",
    "My name is Mark. How are you?",
    "What do you have planned for today?",
]:
    print(f"Human: {user_input}")
    messages.append(HumanMessage(content=user_input))
    reply = model.invoke(messages)
    print(f"Chatbot: {reply.content}\n")
    messages.append(reply)
