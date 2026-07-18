"""A DBpedia SPARQL tool set and a ReAct agent built around it.

Two tools:

- `find_entity(name)` — searches DBpedia by English rdfs:label and
  returns up to five candidate URIs plus their short descriptions.
- `run_sparql(query)` — runs an arbitrary SPARQL query against the
  DBpedia endpoint and returns the raw bindings as JSON.

The agent's job is to first look up entity URIs it needs, then write a
SPARQL query using those URIs, then execute the query.
"""

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
    """Look up DBpedia entities whose English label matches `name`.

    Returns up to five candidate results as a JSON list of
    {uri, label, comment} objects.
    """
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
    """Run a SPARQL query against DBpedia. Returns the raw bindings as JSON."""
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
