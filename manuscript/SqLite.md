# Natural-language SQLite

The previous edition of this chapter used `SQLDatabaseChain` from `langchain_experimental.sql`. That class has been deprecated for over a year and the replacement is a proper LangGraph ReAct agent equipped with the SQL toolkit. The new version is more code but strictly better along every axis: the agent inspects the schema before writing queries, checks its own SQL for mistakes, retries on errors, and stays entirely within the primitives you have already learned in Chapters 6 through 10.

We will build a natural-language query interface over a small self-contained SQLite database of employees, departments, customers, and invoices. Everything runs locally — Ollama for the LLM, SQLite for the database, `langchain-community` for the toolkit that gives the agent its SQL tools.

## The sample database

The `_make_db.py` script builds `company.db` with four tables:

- **departments** (id, name)
- **employees** (id, first_name, last_name, department_id, hire_date, salary)
- **customers** (id, first_name, last_name, city, country)
- **invoices** (id, customer_id, employee_id, invoice_date, total)

Five employees across three departments, six customers across five countries, nine invoices. Small enough that you can eyeball whether the agent's answers are correct, big enough that the queries are non-trivial (aggregations, joins, per-group sums).

Setup:

```console
$ cd source-code/sql_agent
$ uv sync
$ ollama pull qwen3:8b
```

The first agent invocation triggers `_make_db.py` automatically if `company.db` is missing.

## `SQLDatabase` and `SQLDatabaseToolkit`

The two pieces from `langchain-community` that do the real work:

- **`SQLDatabase`** — a thin wrapper around a SQLAlchemy engine. Given a URI (`sqlite:///company.db` in our case), it can list tables, describe schemas, and execute queries.
- **`SQLDatabaseToolkit`** — packages up four LangChain `Tool` objects that operate on a `SQLDatabase`. This is the object we hand to `create_react_agent`.

The four tools the toolkit exposes:

| Tool | Purpose |
|---|---|
| `sql_db_list_tables` | List every table in the database. |
| `sql_db_schema` | Return the CREATE statement plus three sample rows for one or more tables. |
| `sql_db_query` | Execute a SQL query and return the resulting rows (or an error string). |
| `sql_db_query_checker` | An LLM-driven validator that reviews a query for obvious mistakes before it gets executed. |

That set is deliberately narrow. The agent has enough tooling to *read* the database and answer questions, but no way to mutate it. Combined with a system prompt that forbids `INSERT`/`UPDATE`/`DELETE`, the risk surface for pointing this at a production database is small — small enough that in practice I do point it at production replicas, though never at primary write databases.

## Building the agent

`source-code/sql_agent/_agent.py`:

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from _make_db import DB_PATH, make_db

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an assistant that answers questions about a SQLite database. "
        "Use the available tools to explore and query the database. "
        "Always follow this procedure:\n"
        "1. Call sql_db_list_tables to see what tables exist.\n"
        "2. Call sql_db_schema on any tables that look relevant to the question.\n"
        "3. Write a SQL query that answers the question.\n"
        "4. Call sql_db_query_checker on your query.\n"
        "5. Call sql_db_query to execute the (possibly corrected) query.\n"
        "6. Return a concise, plain-English answer to the user's question.\n\n"
        "Never write INSERT, UPDATE, DELETE, DROP, CREATE, or ALTER statements. "
        "Read-only queries only."
    )
)


def build_sql_agent():
    if not DB_PATH.exists():
        make_db()

    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    model = ChatOllama(model="qwen3:8b", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    return create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
```

Three configuration decisions worth spelling out.

**The system prompt.** The six-step procedure is not optional. Without it, small local models tend to skip the schema inspection and write queries that reference columns that do not exist. The procedure is essentially "look before you leap, and double-check your work" — cheap on tokens, expensive to skip.

**The prohibition on DML.** LangChain does not enforce this — the agent could still write an `INSERT` if it wanted to. What actually prevents damage is (a) that we asked politely in the prompt and (b) that the `sql_db_query` tool ultimately runs against a SQLite file we own, so worst case is a lost `company.db` we regenerate. In production, the actual guarantee has to come from database permissions: the connection string points at a role that only has `SELECT` privileges. The prompt is a belt-and-suspenders layer, not the primary defense.

**Passing the model to the toolkit.** `SQLDatabaseToolkit(db=db, llm=model)` uses the model for the `sql_db_query_checker` tool. Not the same as the outer model in `create_react_agent`, though we use the same one for simplicity. In some setups it makes sense to use a small fast model here and a stronger one for the outer agent.

## Running it

`01_sql_agent.py`:

```python
from langchain_core.messages import HumanMessage
from _agent import build_sql_agent

agent = build_sql_agent()

QUESTIONS = [
    "How many employees are there?",
    "Which employee has the highest salary?",
    "Which customer has generated the most total revenue?",
    "What is the total revenue per department?",
]

for q in QUESTIONS:
    print(f"USER: {q}")
    result = agent.invoke(
        {"messages": [HumanMessage(content=q)]},
        config={"recursion_limit": 30},
    )
    final = result["messages"][-1]
    print(f"AGENT: {final.content.strip()}\n")
```

Representative output:

```console
$ uv run 01_sql_agent.py
USER: How many employees are there?
AGENT: There are 5 employees.

USER: Which employee has the highest salary?
AGENT: Bob Brown, in the Engineering department, has the highest salary at $110,000.

USER: Which customer has generated the most total revenue?
AGENT: John Smith has generated the most total revenue, with three invoices totaling $3,050.

USER: What is the total revenue per department?
AGENT: Support brought in $4,175 across four invoices, and Sales brought in $8,700 across five invoices. Engineering has no invoices.
```

The last question requires a two-table join (`invoices` to `employees` to `departments`), a GROUP BY, and a SUM. On the sample data set the agent's answer matches what you get from running the query by hand — but you should not take the agent's word for it. That is what streaming is for.

## Watching the SQL get written

For debugging any agent that writes SQL, `.stream()` is essential. `02_stream_sql_agent.py`:

```python
for step in agent.stream(
    {"messages": [HumanMessage(content=question)]},
    config={"recursion_limit": 30},
):
    for node_name, node_output in step.items():
        print(f"=== node: {node_name} ===")
        for m in node_output.get("messages", []):
            if getattr(m, "tool_calls", None):
                for call in m.tool_calls:
                    args = str(call["args"])
                    if len(args) > 200:
                        args = args[:200] + "..."
                    print(f"  tool_call: {call['name']}({args})")
            if m.content:
                snippet = m.content if len(m.content) < 300 else m.content[:300] + "..."
                print(f"  {type(m).__name__}: {snippet}")
        print()
```

A representative trace for "What is the total revenue per department?":

```console
$ uv run 02_stream_sql_agent.py
USER: What is the total revenue per department?

=== node: agent ===
  tool_call: sql_db_list_tables({})

=== node: tools ===
  ToolMessage: customers, departments, employees, invoices

=== node: agent ===
  tool_call: sql_db_schema({'table_names': 'departments, employees, invoices'})

=== node: tools ===
  ToolMessage: CREATE TABLE departments (...); CREATE TABLE employees (...); CREATE TABLE invoices (...);
    /*
    3 rows from departments table:
    id  name
    1   Engineering
    2   Sales
    3   Support
    ...

=== node: agent ===
  tool_call: sql_db_query_checker({'query': 'SELECT d.name, SUM(i.total) AS revenue FROM invoices i JOIN employees e ON i.employee_id = e.id JOIN departments d ON e.department_id = d.id GROUP BY d.name'})

=== node: tools ===
  ToolMessage: SELECT d.name, SUM(i.total) AS revenue FROM invoices i JOIN employees e ON i.employee_id = e.id JOIN departments d ON e.department_id = d.id GROUP BY d.name

=== node: agent ===
  tool_call: sql_db_query({'query': 'SELECT d.name, SUM(i.total) AS revenue FROM invoices i JOIN employees e ON i.employee_id = e.id JOIN departments d ON e.department_id = d.id GROUP BY d.name'})

=== node: tools ===
  ToolMessage: [('Sales', 8700.0), ('Support', 4175.0)]

=== node: agent ===
  AIMessage: Sales brought in $8,700 across five invoices, and Support brought in $4,175 across four. Engineering has no invoices.
```

Five tool calls — list tables, get schema, check the query, run the query, produce the answer — each one visible. When an agent's answer is wrong, one of these steps is where it goes wrong. Streaming turns "why did the agent say that?" from an hour of head-scratching into a five-minute inspection.

## Where to take this next

Adapting this to a real project is a small number of changes:

- **Point at a different database.** Change the URI in `SQLDatabase.from_uri(...)`. SQLAlchemy speaks PostgreSQL, MySQL, SQL Server, Snowflake, BigQuery, and everything else the reader is likely to have.
- **Restrict which tables the agent can see.** `SQLDatabase(engine, include_tables=[...])` filters the tables the toolkit exposes. Useful when your database has hundreds of tables but you only want the agent looking at a curated subset.
- **Add an approval interrupt on `sql_db_query`.** Wrap the tool node with the pattern from Chapter 9 to require human approval on any query that touches specific tables, or on any query whose EXPLAIN plan is expensive.
- **Add a checkpointer.** Chapter 8's `SqliteSaver` gives the agent per-thread conversation memory — useful when users ask a series of related questions ("...and what about last quarter?").

## What we covered

- `SQLDatabaseChain` from the previous edition is deprecated; the modern replacement is a ReAct agent using `SQLDatabaseToolkit`.
- The toolkit exposes four tools: list tables, get schema, run a query, check a query.
- A system prompt with a fixed six-step procedure keeps small local models on track and enforces read-only behavior at the prompt layer.
- `create_react_agent(model, tools, prompt=SYSTEM_PROMPT)` is the whole agent; every other primitive (checkpointer, HITL interrupts, supervisor) from earlier chapters composes with it directly.
- `.stream()` is the primary debugging tool — every SQL query the agent writes is visible in the trace.

Chapter 12 leaves relational data behind for the semantic web: DBpedia and Wikidata as agent tools, with SPARQL as the query language.
