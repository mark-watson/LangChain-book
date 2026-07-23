# Natural-language SQLite

The previous edition of this chapter used `SQLDatabaseChain` from `langchain_experimental.sql`. That class has been deprecated for over a year, and the replacement is a proper LangGraph ReAct agent equipped with SQL tools. The new version is more code but strictly better along every axis: the agent inspects the schema before writing queries, checks its own SQL for mistakes, retries on errors, and stays entirely within the primitives you have already learned in Chapters "LangGraph 1.0 fundamentals" through "Multi-agent supervisor pattern".

We will build a natural-language query interface over a small self-contained SQLite database of employees, departments, customers, and invoices. Everything runs locally: Ollama for the LLM, SQLite for the database, and raw SQLAlchemy (not `langchain-community`; more on why below) for the tools that give the agent its SQL access.

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
$ ollama pull qwen3.5:4b
```

The first agent invocation triggers `_make_db.py` automatically if `company.db` is missing.

## The four SQL tools

Older LangChain editions gave you this agent's tools for free: `langchain_community.agent_toolkits.SQLDatabaseToolkit` wrapped a `SQLDatabase` object and handed back four ready-made `Tool` instances. That toolkit has no standalone LangChain 1.0 replacement, and pulling in the whole `langchain-community` package for four tools it does not otherwise need is not worth it for this book's stack; `langchain-community` is not a dependency anywhere else in this book. So `_agent.py` reimplements the same four tools directly on top of SQLAlchemy, which is already a dependency for the database work:

| Tool | Purpose |
|---|---|
| `sql_db_list_tables` | List every table in the database. |
| `sql_db_schema` | Return the CREATE statement plus three sample rows for one or more tables. |
| `sql_db_query` | Execute a SQL query and return the resulting rows (or an error string). |
| `sql_db_query_checker` | An LLM-driven validator that reviews a query for obvious mistakes before it gets executed. |

Same tool names, same behavior as far as the agent is concerned, same system-prompt-driven procedure; only the implementation moved. `sql_db_list_tables` and `sql_db_schema` read from a `sqlalchemy.MetaData` object reflected once at startup; `sql_db_query` runs raw SQL through the SQLAlchemy engine and returns the rows as a string; `sql_db_query_checker` sends the query to the model with the same review prompt LangChain's own checker tool used, so the model gets equally specific guidance about `NOT IN` with `NULL`, `UNION` versus `UNION ALL`, and the rest of the usual SQL mistakes.

That set is deliberately narrow. The agent has enough tooling to *read* the database and answer questions, but no way to mutate it. Combined with a system prompt that forbids `INSERT`/`UPDATE`/`DELETE`, the risk surface for pointing this at a production database is small, small enough that in practice I do point it at production replicas, though never at primary write databases.

## Building the agent

`source-code/sql_agent/_agent.py`:

```python
import sqlalchemy as sa
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from sqlalchemy.exc import SQLAlchemyError

from _make_db import DB_PATH, make_db

QUERY_CHECKER_PROMPT = """
{query}
Double check the {dialect} query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """

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


def _sample_rows(engine: sa.Engine, table: sa.Table, n: int = 3) -> str:
    columns = "\t".join(col.name for col in table.columns)
    with engine.connect() as conn:
        rows = conn.execute(sa.select(table).limit(n)).fetchall()
    rows_str = "\n".join("\t".join(str(v)[:100] for v in row) for row in rows)
    return f"{n} rows from {table.name} table:\n{columns}\n{rows_str}"


def make_sql_tools(engine: sa.Engine, model: ChatOllama) -> list:
    metadata = sa.MetaData()
    metadata.reflect(bind=engine)

    @tool
    def sql_db_list_tables() -> str:
        """Input is an empty string, output is a comma-separated list of tables in the database."""
        return ", ".join(sorted(metadata.tables))

    @tool
    def sql_db_schema(table_names: str) -> str:
        """Get the CREATE statement and sample rows for a comma-separated list of tables."""
        chunks = []
        for name in (t.strip() for t in table_names.split(",")):
            table = metadata.tables.get(name)
            if table is None:
                return f"Error: table '{name}' not found"
            create_stmt = str(sa.schema.CreateTable(table).compile(engine)).strip()
            chunks.append(f"{create_stmt}\n\n/*\n{_sample_rows(engine, table)}\n*/")
        return "\n\n".join(chunks)

    @tool
    def sql_db_query(query: str) -> str:
        """Execute a SQL query against the database and get back the result.
        If the query is not correct, an error message will be returned.
        If an error is returned, rewrite the query, check the query, and try again."""
        try:
            with engine.begin() as conn:
                result = conn.execute(sa.text(query))
                if not result.returns_rows:
                    return ""
                rows = [tuple(row) for row in result.fetchall()]
            return str(rows) if rows else ""
        except SQLAlchemyError as e:
            return f"Error: {e}"

    @tool
    def sql_db_query_checker(query: str) -> str:
        """Use this tool to double check if your query is correct before executing it.
        Always use this tool before executing a query with sql_db_query!"""
        prompt = QUERY_CHECKER_PROMPT.format(query=query, dialect=engine.dialect.name)
        return model.invoke(prompt).content

    return [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]


def build_sql_agent():
    if not DB_PATH.exists():
        make_db()

    engine = sa.create_engine(f"sqlite:///{DB_PATH}")
    model = ChatOllama(
        model="qwen3.5:4b",
        temperature=0,
        num_ctx=16336,
        reasoning=False,
    )
    tools = make_sql_tools(engine, model)

    return create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
```

Four configuration decisions worth spelling out.

**The system prompt.** The six-step procedure is not optional. Without it, small local models tend to skip the schema inspection and write queries that reference columns that do not exist. The procedure is essentially "look before you leap, and double-check your work": cheap on tokens, expensive to skip.

**The prohibition on DML.** LangChain does not enforce this; the agent could still write an `INSERT` if it wanted to. What actually prevents damage is (a) that we asked politely in the prompt and (b) that the `sql_db_query` tool ultimately runs against a SQLite file we own, so worst case is a lost `company.db` we regenerate. In production, the actual guarantee has to come from database permissions: the connection string points at a role that only has `SELECT` privileges. The prompt is a belt-and-suspenders layer, not the primary defense.

**Passing the model into `make_sql_tools`.** The `sql_db_query_checker` tool needs a model to review the query before it runs, so `model` is threaded into `make_sql_tools(engine, model)` and closed over by that one tool. It does not have to be the same model object as the outer agent (in some setups it makes sense to use a small, fast model here and a stronger one for the outer agent), but we use the same one for simplicity.

**`num_ctx=16336` and `reasoning=False`.** These two settings are not optional. The ReAct loop produces a long transcript (the schema tool returns full `CREATE` statements plus sample rows, the query checker echoes the query back, and each tool call adds two messages), so the default 2k-token context window fills up within the first couple of steps and the model loses track of what it has already done; `num_ctx=16336` leaves headroom for a full run. `qwen3.5:4b` is also a "thinking" model that by default wraps its output in `<think>...</think>` tags; with thinking on, the agent mis-serializes tool calls and can loop or return an empty answer because the real response never left the stripped think block. `reasoning=False` turns Ollama's native `think` flag off so the model emits plain tool calls and a real final answer.

## Running it

`01_sql_agent.py`:

```python
from langchain_core.messages import HumanMessage
from _agent import build_sql_agent

agent = build_sql_agent()

QUESTIONS = [
#    "How many employees are there?",
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

The first question from earlier drafts of this script ("How many employees are there?") is commented out above. It's the simplest of the four (a bare `COUNT(*)`, no ordering or join required), so the three that remain are the ones that actually put the agent's schema-reading and query-checking loop to work. Uncomment it if you want a warm-up question.

Representative output, captured by actually running the script against the shipped `company.db`:

```console
$ uv run 01_sql_agent.py
USER: Which employee has the highest salary?
AGENT: The employee with the highest salary is **Bob Brown**, who earns $110,000 per year.

USER: Which customer has generated the most total revenue?
AGENT: **Marie Dupont** has generated the most total revenue, with **$3,700**. This comes from two invoices: $2,100 on February 10 and another invoice (not shown in the sample data) that contributed to her total.

USER: What is the total revenue per department?
AGENT: The total revenue per department is:

| Department | Total Revenue |
|------------|---------------|
| Sales      | $12,875.00    |

Based on the data in your database, only one invoice was generated by an employee from the Sales department (employee Carol Chen), which totaled $1,287.50 across two invoices ($1,250 + $850). There were no invoices associated with employees from Engineering or Support departments in this dataset.
```

Two things worth noticing in that transcript. First, the numbers that matter are real: Bob Brown genuinely has the highest salary ($110,000), Marie Dupont genuinely generated the most revenue ($3,700, across exactly two invoices), and $12,875.00 attributed entirely to Sales is exactly what running the aggregate query by hand returns; every invoice in this sample database happens to have been written by one of two Sales employees, so Engineering and Support really do show zero. Second, the prose wrapped around those numbers is not entirely trustworthy: the department answer says "only one invoice," names a single employee, and offers a dollar breakdown ($1,250 + $850) that does not even sum to the total it claims. The tabulated answer is correct; the model's own explanation of how it got there is partly confabulated. That gap (a right answer with an unreliable explanation bolted onto it) is exactly why you check an agent's work by watching its tool calls, not by reading its summary. That is what streaming is for.

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

The real trace for "What is the total revenue per department?", the same question behind the confabulated summary above:

```console
$ uv run 02_stream_sql_agent.py
USER: What is the total revenue per department?

=== node: agent ===
  tool_call: sql_db_list_tables({})

=== node: tools ===
  ToolMessage: customers, departments, employees, invoices

=== node: agent ===
  tool_call: sql_db_schema({'table_names': 'departments,employees,invoices,customers'})

=== node: tools ===
  ToolMessage: CREATE TABLE departments (
	id INTEGER, 
	name TEXT NOT NULL, 
	PRIMARY KEY (id)
)

/*
3 rows from departments table:
id	name
1	Engineering
2	Sales
3	Support
*/

CREATE TABLE employees (
	id INTEGER, 
	first_name TEXT NOT NULL, 
	last_name TEXT NOT NULL, 
	department_id INTEGER NOT NULL, 
	hire_date...

=== node: agent ===
  tool_call: sql_db_query_checker({'query': 'SELECT d.name AS department, SUM(i.total) AS total_revenue FROM invoices i JOIN employees e ON i.employee_id = e.id JOIN departments d ON e.department_id = d.id GROUP BY d.id ORDER BY d.id'...)

=== node: tools ===
  ToolMessage: SELECT d.name AS department, SUM(i.total) AS total_revenue FROM invoices i JOIN employees e ON i.employee_id = e.id JOIN departments d ON e.department_id = d.id GROUP BY d.id ORDER BY d.id;

=== node: agent ===
  tool_call: sql_db_query({'query': 'SELECT d.name AS department, SUM(i.total) AS total_revenue FROM invoices i JOIN employees e ON i.employee_id = e.id JOIN departments d ON e.department_id = d.id GROUP BY d.id ORDER BY d.id'...)

=== node: tools ===
  ToolMessage: [('Sales', 12875.0)]

=== node: agent ===
  AIMessage: The total revenue per department is:

| Department | Total Revenue |
|------------|---------------|
| Sales      | $12,875.00    |

Based on the data in your database, only one invoice was generated by an employee from the Sales department (employee Carol Chen), which totaled $1,287.50 across two in...
```

Four tool calls (list tables, get schema, check the query, run the query), then the final answer. Five steps, each one visible, and the trace explains the confabulation above. The `sql_db_query` result is a single bare row, `[('Sales', 12875.0)]`: no employee name attached anywhere. The only employee names that appear *anywhere* in the whole transcript come from the `sql_db_schema` step's three-row sample of the `employees` table, which, because `LIMIT 3` returns rows in primary-key order, shows Alice Anderson, Bob Brown, and Carol Chen, but not Dan Davis. Carol and Dan are both in Sales and both wrote invoices; the model only ever *saw* Carol by name. When it had to narrate an answer it did not have the receipts for, it reached for the one Sales employee it had actually observed. The lesson generalizes: an agent's summary can smuggle in details from anywhere in its context window, not just from the tool call that produced the number you asked about, which is one more reason to check the trace instead of trusting the prose.

## Where to take this next

Adapting this to a real project is a small number of changes:

- **Point at a different database.** Change the URI passed to `sa.create_engine(...)`. SQLAlchemy speaks PostgreSQL, MySQL, SQL Server, Snowflake, BigQuery, and everything else the reader is likely to have; nothing else in `make_sql_tools` or `build_sql_agent` needs to change.
- **Restrict which tables the agent can see.** `metadata.reflect(bind=engine, only=[...])` limits reflection to a named subset of tables, so `sql_db_list_tables` and `sql_db_schema` never mention the rest. Useful when your database has hundreds of tables but you only want the agent looking at a curated subset.
- **Add an approval interrupt on `sql_db_query`.** Wrap the tool node with the pattern from Chapter "Human-in-the-loop patterns" to require human approval on any query that touches specific tables, or on any query whose EXPLAIN plan is expensive.
- **Add a checkpointer.** The "Durable, restart-safe agents" chapter's `SqliteSaver` gives the agent per-thread conversation memory, useful when users ask a series of related questions ("...and what about last quarter?").

## What we covered

- `SQLDatabaseChain` from the previous edition is deprecated, and its would-be LangChain 1.0 successor `SQLDatabaseToolkit` has no standalone replacement either; the modern approach is four small tools hand-rolled directly on SQLAlchemy.
- Those four tools (list tables, get schema, run a query, check a query) give a ReAct agent everything it needs to answer read-only questions, and nothing it needs to change data.
- A system prompt with a fixed six-step procedure keeps small local models on track and enforces read-only behavior at the prompt layer.
- `create_react_agent(model, tools, prompt=SYSTEM_PROMPT)` is the whole agent; every other primitive (checkpointer, HITL interrupts, supervisor) from earlier chapters composes with it directly.
- `.stream()` is the primary debugging tool. A tabulated answer can be correct even when the agent's prose explanation of it is not; the trace is how you tell the difference, and how you catch an agent quietly borrowing details from the wrong part of its own context window.

Chapter "DBpedia and Wikidata as agent tools" leaves relational data behind for the semantic web, with SPARQL as the query language.
