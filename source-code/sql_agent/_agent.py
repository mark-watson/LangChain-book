"""Build a ReAct agent equipped with hand-rolled SQL tools.

langchain_community.agent_toolkits.SQLDatabaseToolkit has no standalone
1.0 replacement, so these four tools are reimplemented directly on top of
SQLAlchemy (already a dependency) instead of pulling in the retired
langchain-community package:

- `sql_db_list_tables` — list the tables in the database.
- `sql_db_schema`      — get the CREATE statement and a few sample rows
                         for one or more tables.
- `sql_db_query`       — execute a SQL query and return the result rows.
- `sql_db_query_checker` — an LLM-powered validator that reviews a query
                           for obvious mistakes before you run it.

The system prompt instructs the model to work in the standard order:
list tables, get schema for the ones it cares about, write a query, check
it, run it. It also forbids anything that mutates the database.
"""

import sqlalchemy as sa
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from sqlalchemy.exc import SQLAlchemyError

from _make_db import DB_PATH, make_db

# Same wording LangChain's own SQL query checker tool used, so the model
# gets equally specific guidance without pulling in langchain_community.
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
    """Reimplements the four SQLDatabaseToolkit tools directly on SQLAlchemy."""
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
    # Two model settings are essential for this agent to work reliably:
    #
    # num_ctx=16336 — The ReAct loop produces a long transcript (the schema
    #   tool returns full CREATE statements plus sample rows, the query
    #   checker echoes the query back, and each tool call adds two messages).
    #   The default 2k-context window fills up within the first couple of
    #   steps, truncating the system prompt or earlier tool results and
    #   causing the model to lose track of what it has already done.
    #   16k leaves comfortable headroom for a 6-step run.
    #
    # reasoning=False — qwen3.5 is a "thinking" model: by default it wraps its
    #   output in <think>...</think> tags that langchain_ollama strips from the
    #   visible content. With thinking on, the agent mis-serializes tool calls
    #   (ollama raises "XML syntax error ... element <function> closed by
    #   </parameter>"), loops on sql_db_schema({}) because it forgets to fill
    #   the required table_names arg, and finally returns an empty answer
    #   because the whole response lived inside the stripped think block.
    #   reasoning=False flips ollama's native `think` flag off so the model
    #   emits plain tool calls and a real final answer instead.
    model = ChatOllama(
        model="qwen3.5:4b",
        temperature=0,
        num_ctx=16336,
        reasoning=False,
    )
    tools = make_sql_tools(engine, model)

    return create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
