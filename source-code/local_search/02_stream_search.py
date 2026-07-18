"""Stream the pipeline so each stage's output is visible as it happens."""

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

for step in app.stream(initial):
    for node_name, node_output in step.items():
        print(f"=== node: {node_name} ===")
        if node_name == "search":
            for r in node_output["raw_results"]:
                print(f"  - {r.get('title', '')}")
                print(f"    {r.get('href') or r.get('url')}")
        elif node_name == "filter":
            print(f"  kept {len(node_output['filtered_results'])} results")
        elif node_name == "fetch":
            for p in node_output["pages"]:
                print(f"  fetched {len(p['text'])} chars from {p['url']}")
        elif node_name == "summarize":
            for i, s in enumerate(node_output["summaries"]):
                snippet = s if len(s) < 200 else s[:200] + "..."
                print(f"  [{i}] {snippet}")
        elif node_name == "synthesize":
            print(node_output["final_answer"])
        print()
