"""Build a ReAct agent equipped with the SQLDatabaseToolkit.

The toolkit gives the agent four tools:

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
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    return create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
