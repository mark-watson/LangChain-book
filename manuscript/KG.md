# DBpedia and Wikidata as agent tools

Two of the largest and most useful public knowledge graphs on the internet are [DBpedia](https://www.dbpedia.org) and [Wikidata](https://www.wikidata.org). Both are free to query, both use RDF as their data model, both speak SPARQL, and neither requires an API key. For a solo developer building anything that needs grounded factual data (people, places, organizations, dates, relationships), they are high-value data assets, and they compose beautifully with LangGraph agents. This chapter builds one small agent per KG and shows how to give the model SPARQL as a tool.

The example source code for this chapter is found in **source-code/kg_agent**.

When I worked on a knowledge graph project at Google in 2013 after writing two books on the semantic web, linked data, and knowledge graphs.

The previous edition of this chapter used the old `GPTSimpleVectorIndex` / `GPTTreeIndex` classes from LlamaIndex to wrap SPARQL results in an embedding index and query them as text. That was a workable pattern in 2023, but both classes have long since been removed from LlamaIndex, and the modern replacement, a ReAct agent that calls SPARQL directly, is simpler and answers a wider range of questions. It is also faster: no embedding step, no index build, no round trip through a vector store. So, dear reader, this example was a LlamaIndex example in the old edition of this book and is a LangChain example in this new 2026 edition.

I am not going to teach SPARQL from scratch here. If you have never used it before, the short version is:

- RDF stores data as **triples**: `<subject> <predicate> <object>`.
- SPARQL queries look like SQL with pattern matching over triples: `SELECT ?x WHERE { ?x <predicate> <object> }` finds every subject `?x` that has the given predicate/object pair.
- Every entity has a URI. `<http://dbpedia.org/resource/Germany>` is the DBpedia URI for Germany.
- Wikidata uses opaque QIDs instead of readable URIs: Q183 is Germany, Q80041 is Sedona, Arizona.
- Both endpoints have web query consoles: [dbpedia.org/sparql](https://dbpedia.org/sparql) and [query.wikidata.org](https://query.wikidata.org). Poking at them by hand is the fastest way to develop intuition.

For a deeper introduction, the "Linked Data, the Semantic Web, and Knowledge Graphs" chapter in my Hy Language book is free to read online at [leanpub.com/hy-lisp-python/read](https://leanpub.com/hy-lisp-python/read). I have also written full books dedicated to SPARQL if the topic pulls you in.

## The example

We define two agents, one per KG. Both live in `source-code/kg_agent/`. Each has two tools:

- **`find_entity(name)`**: search the KG by label and return candidate URIs (DBpedia) or QIDs (Wikidata).
- **`run_sparql(query)`**: execute a SPARQL query and return the raw bindings as JSON.

The agent's job is to first look up the entities its question mentions to get their identifiers, then write a SPARQL query using those identifiers, then execute it. Setup:

```console
$ cd source-code/kg_agent
$ uv sync
$ ollama pull qwen3.5:4b
```

Both endpoints are public and require an internet connection but no credentials.

## The DBpedia agent

We start with defining the DBPedia agent `_dbpedia.py`:

```python
import json

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from SPARQLWrapper import JSON, SPARQLWrapper

ENDPOINT = "https://dbpedia.org/sparql"


def _query(sparql: str) -> list[dict]:
    wrapper = SPARQLWrapper(ENDPOINT, agent="LangChain-book-example/1.0")
    wrapper.setQuery(sparql)
    wrapper.setReturnFormat(JSON)
    return wrapper.query().convert()["results"]["bindings"]


@tool
def find_entity(name: str) -> str:
    """Look up DBpedia entities whose English label matches `name`."""
    sparql = f"""
    SELECT DISTINCT ?s ?label ?comment WHERE {{
      ?s rdfs:label {json.dumps(name)}@en .
      OPTIONAL {{ ?s rdfs:comment ?comment . FILTER (lang(?comment) = 'en') }}
      BIND ({json.dumps(name)} AS ?label)
    }} LIMIT 5
    """
    try:
        rows = _query(sparql)
    except Exception as exc:
        return f"Lookup failed: {exc}"

    results = []
    for row in rows:
        results.append(
            {
                "uri": row["s"]["value"],
                "label": row["label"]["value"],
                "comment": row.get("comment", {}).get("value", ""),
            }
        )
    return json.dumps(results, ensure_ascii=False)


@tool
def run_sparql(query: str) -> str:
    """Run a SPARQL query against DBpedia."""
    try:
        rows = _query(query)
    except Exception as exc:
        return f"Query failed: {exc}"
    return json.dumps(rows[:20], ensure_ascii=False)


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an assistant that answers questions using DBpedia. "
        "You have two tools:\n"
        "- find_entity(name): search DBpedia for a URI matching an English label.\n"
        "- run_sparql(query): execute a SPARQL query against DBpedia.\n\n"
        "Procedure:\n"
        "1. Identify the entities the question mentions.\n"
        "2. Use find_entity on each entity to get its DBpedia URI.\n"
        "3. Write a SPARQL query that uses those URIs to answer the question. "
        "Wrap URIs in angle brackets.\n"
        "4. Call run_sparql to execute the query.\n"
        "5. Return a concise English answer using the query results.\n\n"
        "Useful DBpedia prefixes are already known to the endpoint (dbo:, dbp:, dbr:, "
        "rdfs:, foaf:). You do not need to declare them, but you can if you prefer."
    )
)


def build_dbpedia_agent():
    model = ChatOllama(model="qwen3.5:4b", temperature=0)
    return create_react_agent(model, [find_entity, run_sparql], prompt=SYSTEM_PROMPT)
```

Three things to notice.

**`SPARQLWrapper` handles the transport.** It POSTs the query to the endpoint, requests JSON back, and parses the response. Setting the `agent` string is worth doing: the DBpedia endpoint occasionally rate-limits requests that use its default user-agent, and a custom one is polite besides.

**`find_entity` uses `json.dumps(name)` to escape the search string.** SPARQL string literals use quotes and are vulnerable to injection the same way SQL strings are. `json.dumps` produces a properly-escaped, double-quoted string, and SPARQL accepts JSON-style string literals.

**`run_sparql` truncates results to 20 rows.** The DBpedia endpoint can return thousands of rows for a broad query, which would blow past the model's context window. Twenty is a compromise: enough to answer most questions, small enough that the model can inspect the results directly.

The driver script, `01_dbpedia_agent.py` is short:

```python
from langchain_core.messages import HumanMessage
from _dbpedia import build_dbpedia_agent

agent = build_dbpedia_agent()

question = "What countries border Germany?"

print(f"USER: {question}")

result = agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config={"recursion_limit": 30},
)

print(f"AGENT: {result['messages'][-1].content.strip()}")
```

Representative output:

```console
$ uv run 01_dbpedia_agent.py
USER: What countries border Germany?
AGENT: Germany borders Denmark, Poland, the Czech Republic, Austria, Switzerland,
       France, Luxembourg, Belgium, and the Netherlands.
```

Under the hood the agent has called `find_entity("Germany")` to get `<http://dbpedia.org/resource/Germany>`, then written a query along the lines of:

```sparql
SELECT DISTINCT ?country ?label WHERE {
  <http://dbpedia.org/resource/Germany> dbo:borders ?country .
  ?country rdfs:label ?label .
  FILTER (lang(?label) = 'en')
}
```

then executed it and summarized the results. Add `.stream()` from the "Building a ReAct agent with LangGraph + Ollama" chapter's streaming example if you want to watch the intermediate tool calls.

## The Wikidata agent

Wikidata uses opaque QIDs instead of readable URIs, and its query patterns look slightly different. `_wikidata.py` is the same shape, with two differences worth calling out.

**Entity search uses the `wikibase:mwapi` SERVICE.** Wikidata's SPARQL endpoint has direct access to the MediaWiki API through a special SERVICE block. This is much better than raw `rdfs:label` matching because it handles fuzzy search, redirects, and multiple languages transparently. The relevant SPARQL:

```sparql
SELECT ?item ?itemLabel ?itemDescription WHERE {
  SERVICE wikibase:mwapi {
    bd:serviceParam wikibase:api "EntitySearch" .
    bd:serviceParam wikibase:endpoint "www.wikidata.org" .
    bd:serviceParam mwapi:search "Bill Clinton" .
    bd:serviceParam mwapi:language "en" .
    ?item wikibase:apiOutputItem mwapi:item .
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} LIMIT 5
```

The `wikibase:label` service at the end automatically populates `?itemLabel` and `?itemDescription` in English.

**The system prompt calls out common properties.** Wikidata's properties are numbered (P31, P569, ...) and there is no way for a model to guess them without looking them up. The prompt seeds the agent with the ones it is likely to need:

- `wdt:P31`: instance of
- `wdt:P39`: position held
- `wdt:P580` / `wdt:P582`: start / end time
- `wdt:P17`: country
- `wdt:P569`: date of birth
- `wdt:P106`: occupation

You could instead give the agent a third tool for looking up property IDs by name (the `EntitySearch` API returns properties as well as entities), and I have tried versions of this agent both ways. Baking common properties into the prompt keeps this example short; a real project would probably want the extra tool.

Here is the complete source code for `_wikidata.py`:

```python
"""A Wikidata SPARQL tool set and a ReAct agent built around it.

Same shape as `_dbpedia.py`, different endpoint and URI style.

Wikidata uses opaque QIDs (Q80041 for Sedona, Arizona) rather than
human-readable URIs, and its property URIs live under wdt: (e.g. wdt:P31
for "instance of"). The entity-search tool uses Wikidata's own search API
(SERVICE wikibase:mwapi) which handles fuzzy matching much better than
raw rdfs:label matching.
"""

import json

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from SPARQLWrapper import JSON, SPARQLWrapper

ENDPOINT = "https://query.wikidata.org/sparql"


def _query(sparql: str) -> list[dict]:
    wrapper = SPARQLWrapper(ENDPOINT, agent="LangChain-book-example/1.0")
    wrapper.setQuery(sparql)
    wrapper.setReturnFormat(JSON)
    return wrapper.query().convert()["results"]["bindings"]


@tool
def find_entity(name: str) -> str:
    """Look up Wikidata entities whose label matches `name` (fuzzy search).

    Returns up to five candidate results as a JSON list of
    {qid, label, description} objects. The `qid` (e.g. "Q80041") is the
    piece to use in SPARQL — write it as wd:Q80041.
    """
    escaped = json.dumps(name)
    sparql = f"""
    SELECT ?item ?itemLabel ?itemDescription WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch" .
        bd:serviceParam wikibase:endpoint "www.wikidata.org" .
        bd:serviceParam mwapi:search {escaped} .
        bd:serviceParam mwapi:language "en" .
        ?item wikibase:apiOutputItem mwapi:item .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT 5
    """
    try:
        rows = _query(sparql)
    except Exception as exc:
        return f"Lookup failed: {exc}"

    results = []
    for row in rows:
        uri = row["item"]["value"]
        qid = uri.rsplit("/", 1)[-1]
        results.append(
            {
                "qid": qid,
                "label": row.get("itemLabel", {}).get("value", ""),
                "description": row.get("itemDescription", {}).get("value", ""),
            }
        )
    return json.dumps(results, ensure_ascii=False)


@tool
def run_sparql(query: str) -> str:
    """Run a SPARQL query against Wikidata. Returns the raw bindings as JSON."""
    try:
        rows = _query(query)
    except Exception as exc:
        return f"Query failed: {exc}"
    return json.dumps(rows[:20], ensure_ascii=False)


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an assistant that answers questions using Wikidata. "
        "You have two tools:\n"
        "- find_entity(name): fuzzy-search Wikidata for a QID matching an English label.\n"
        "- run_sparql(query): execute a SPARQL query against Wikidata.\n\n"
        "Procedure:\n"
        "1. Identify the entities the question mentions.\n"
        "2. Use find_entity on each entity to get its QID (like Q80041).\n"
        "3. Write a SPARQL query that references entities as wd:QID and "
        "properties as wdt:PID.\n"
        "4. Call run_sparql to execute the query.\n"
        "5. Return a concise English answer using the query results.\n\n"
        "Include this label helper at the end of any SELECT so labels come back "
        "in English:\n"
        "  SERVICE wikibase:label { bd:serviceParam wikibase:language 'en'. }\n"
        "Common properties: wdt:P31 (instance of), wdt:P39 (position held), "
        "wdt:P580 (start time), wdt:P582 (end time), wdt:P17 (country), "
        "wdt:P569 (date of birth), wdt:P106 (occupation)."
    )
)


def build_wikidata_agent():
    model = ChatOllama(model="qwen3.5:4b", temperature=0)
    return create_react_agent(model, [find_entity, run_sparql], prompt=SYSTEM_PROMPT)

```

The second example using the WikiData agent is found in file `02_wikidata_agent.py`:

```python
from langchain_core.messages import HumanMessage
from _wikidata import build_wikidata_agent

agent = build_wikidata_agent()

question = "When was Bill Clinton president of the United States?"

print(f"USER: {question}")

result = agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config={"recursion_limit": 30},
)

print(f"AGENT: {result['messages'][-1].content.strip()}")
```

A representative run:

```console
$ uv run 02_wikidata_agent.py
USER: When was Bill Clinton president of the United States?
AGENT: Bill Clinton was president of the United States from January 20, 1993
       to January 20, 2001.
```

The agent has looked up Bill Clinton's QID (Q1124), queried his positions held with `wdt:P39`, filtered to the presidency, and pulled the start and end dates via `wdt:P580` and `wdt:P582`.

## DBpedia versus Wikidata

Which one to reach for depends on the question.

- **DBpedia** is easier to explore because URIs are readable (`dbr:Germany` instead of `wd:Q183`) and its property names are English words (`dbo:borders` instead of `wdt:P47`). Great for prototyping and for questions that mostly involve English-language Western topics.
- **Wikidata** has broader coverage, more languages, and more reliable up-to-date data because its edits are curated. Its query patterns are more verbose but its data is generally cleaner.

In practice I use DBpedia when I am writing a query interactively (its readable URIs are nicer to reason about) and Wikidata when I need coverage or freshness. An agent can be given tools for both, if a query needs it; you would just add both tool sets to a single `create_react_agent` call.

## Where to take this next

Everything you learned in Chapters "Building a ReAct agent with LangGraph + Ollama" through "Multi-agent supervisor pattern" composes with these KG agents:

- **Add a checkpointer** (Chapter "Durable, restart-safe agents") for follow-up questions across turns. "Which of those countries has the largest population?" makes sense as a second turn only if the agent remembers the first answer.
- **Add an approval interrupt** (Chapter "Human-in-the-loop patterns") if the KG could ever return sensitive information you want a human to see before it goes to the user.
- **Add these tools to a supervisor graph** (Chapter "Multi-agent supervisor pattern") as a "facts specialist" alongside your other specialists. This is the pattern I use most often: the KG agent is one of several specialists a supervisor can call on for grounded factual data.

## What we covered

- DBpedia and Wikidata are large public knowledge graphs with SPARQL endpoints and no API keys.
- Giving a LangGraph agent two tools (entity lookup and SPARQL execution) turns SPARQL into an accessible skill for an LLM that does not know SPARQL by heart.
- DBpedia is friendlier to prototype against; Wikidata has better coverage and cleaner data.
- The pattern is identical for both KGs and composes with checkpointers, HITL, and supervisor graphs from earlier chapters.

The next chapter "A Perplexity-style local search agent" wraps up Part I, tying together web search, RAG, and multi-step reasoning.
