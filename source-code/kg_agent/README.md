# Chapter 12 — DBpedia and Wikidata as agent tools

Two ReAct agents, one per knowledge graph, that answer factual questions by querying public SPARQL endpoints.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

Requires an internet connection to reach `dbpedia.org` and `query.wikidata.org`. Neither endpoint requires an API key.

## Scripts

| Script | What it shows |
|---|---|
| `_dbpedia.py` | Two DBpedia tools (`find_entity`, `run_sparql`) and a ReAct agent that uses them. |
| `_wikidata.py` | Same shape for Wikidata. |
| `01_dbpedia_agent.py` | Runs the DBpedia agent on a demo question. |
| `02_wikidata_agent.py` | Runs the Wikidata agent on a demo question. |

Run either driver:

```console
$ uv run 01_dbpedia_agent.py
$ uv run 02_wikidata_agent.py
```

Add `.stream()` (as in Chapter 7's streaming example) if you want to watch each tool call as the agent works through the query.
