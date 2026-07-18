"""Send a request to the running workflow server.

Requires that 01_serve_workflow.py is running in another terminal:

    uv run python 01_serve_workflow.py

Then run this client:

    uv run python 02_client.py
"""

import asyncio

import httpx


async def main():
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Health check
        resp = await client.get("http://127.0.0.1:8000/health")
        print(f"Health: {resp.json()}")

        # Ask a question
        resp = await client.post(
            "http://127.0.0.1:8000/ask",
            json={"question": "What is the capital of Arizona?"},
        )
        data = resp.json()
        print(f"AGENT: {data['answer']}")


asyncio.run(main())
