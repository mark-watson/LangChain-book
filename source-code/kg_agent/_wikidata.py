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
