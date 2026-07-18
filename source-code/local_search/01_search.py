"""Run the search pipeline end-to-end and print the final synthesis."""

from _pipeline import build_pipeline

app = build_pipeline()

query = "What are the main challenges in running large language models on consumer laptops?"

print(f"USER: {query}\n")

initial: dict = {
    "query": query,
    "raw_results": [],
    "filtered_results": [],
    "pages": [],
    "summaries": [],
    "final_answer": "",
}

final = app.invoke(initial)

print("=== FINAL ANSWER ===")
print(final["final_answer"])
