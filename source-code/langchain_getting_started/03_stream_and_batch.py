"""Two other things every Runnable can do: stream and batch.

.stream() yields chunks as the model produces them, which is what you want
whenever a human is watching the output appear.

.batch() runs multiple inputs concurrently and returns the results in order,
which is what you want for offline processing where latency per item doesn't
matter but throughput does.
"""

from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3.5:4b", temperature=0)

print("=== .stream() ===")
for chunk in model.stream("Write a two-sentence description of Sedona, Arizona."):
    print(chunk.content, end="", flush=True)
print()

print("\n=== .batch() ===")
prompts = [
    "Name one bird native to Arizona.",
    "Name one bird native to Alaska.",
    "Name one bird native to Florida.",
]
responses = model.batch(prompts)
for prompt, response in zip(prompts, responses):
    print(f"{prompt!r} -> {response.content.strip()}")
