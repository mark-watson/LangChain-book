# Structured extraction

Sometimes the LLM's job is not to answer a question or hold a conversation but to turn an unstructured input into a structured record. Extracting names, addresses, and emails from a customer message. Turning meeting notes into calendar events. Converting a paragraph of product description into fields for a database. This class of task is where a good structured-output primitive earns its weight.

LlamaIndex's primitive is **`llm.structured_predict(SchemaClass, prompt_template, **vars)`**. Hand it a Pydantic model and a prompt template; get back a validated instance of the model. The framework handles the JSON-schema plumbing, the "here is the schema, please output JSON" prompt engineering, and the retry loop when validation fails.

Everything lives in `source-code/llama_index_extract/`. Setup:

```console
$ cd source-code/llama_index_extract
$ uv sync
$ ollama pull qwen3.5:4b
```

## One-shot extraction

`01_structured_predict.py`:

```python
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field


class Person(BaseModel):
    """A person mentioned in text."""

    name: str = Field(description="The person's full name.")
    address: str | None = Field(default=None, description="Street address if given.")
    email: str | None = Field(default=None, description="Email address if given.")


llm = Ollama(model="qwen3.5:4b", temperature=0, request_timeout=120.0, thinking=False)

prompt = PromptTemplate(
    "Extract structured information about the person mentioned in the following text. "
    "If a field is not mentioned, leave it null.\n\n"
    "Text: {input_text}"
)

text = (
    "Mark Johnson enjoys living in Berkeley, California at 102 Dunston Street "
    "and can be reached at mjess@foobar.com."
)

person: Person = llm.structured_predict(Person, prompt, input_text=text)

print(f"name    = {person.name}")
print(f"address = {person.address}")
print(f"email   = {person.email}")
```

Two design points worth being explicit about.

**The docstrings on `Field(description=...)` are part of the prompt.** The model reads them along with the JSON schema, and the quality of your field descriptions directly affects the quality of the extraction. Vague descriptions produce sloppy extraction; precise descriptions produce reliable extraction.

**The `Optional` (`str | None`) with `default=None`** is how you tell the model a field may not be present. Without it, the model tends to hallucinate values for missing fields.

Expected output:

```console
$ uv run 01_structured_predict.py
name    = Mark Johnson
address = 102 Dunston Street, Berkeley, California
email   = mjess@foobar.com
```

## Batch extraction

The same primitive in a loop. `02_batch_extract.py`:

```python
class Event(BaseModel):
    """A calendar-style event described in text."""

    title: str = Field(description="Short title of the event.")
    date: str = Field(description="Date in YYYY-MM-DD format.")
    location: str | None = Field(default=None, description="Location if mentioned.")


llm = Ollama(model="qwen3.5:4b", temperature=0, request_timeout=120.0, thinking=False)

prompt = PromptTemplate(
    "Extract calendar event information from the text below. "
    "If the location is not mentioned, leave it null.\n\n"
    "Text: {input_text}"
)

notes = [
    "Meeting with Carol on March 15, 2026 at the Sedona office.",
    "Team lunch on April 3, 2026.",
    "Book launch party April 20, 2026 at the downtown bookstore in Flagstaff.",
]

events: list[Event] = []
for note in notes:
    event = llm.structured_predict(Event, prompt, input_text=note)
    events.append(event)

for e in events:
    print(f"{e.date}  {e.title!r}  location={e.location!r}")
```

This is the shape of nearly every "migrate a folder of unstructured notes into a database" workflow. If throughput matters, replace the sequential loop with `llm.astructured_predict` and an `asyncio.gather`; for local models with limited concurrency, keeping it sequential is often faster than trying to parallelize.

## When to reach for this vs a chat model with tools

`structured_predict` and tool-calling look similar — both make the model output structured data. The difference is what you do with the result.

- **`structured_predict`** returns a Pydantic object directly. Use it when the LLM's job is to *produce* structured data as the final answer.
- **Tool calling** (Chapters 4 and 16) returns tool call requests that you then execute. Use it when the LLM's job is to *decide* to do something that produces data.

There is overlap. A ReAct agent whose final answer is a Pydantic object is a valid pattern. But if all you need is "text in, structured record out," `structured_predict` is dramatically simpler.

## What we covered

- `llm.structured_predict(SchemaClass, prompt, **vars)` extracts a validated Pydantic object from unstructured text in one call.
- Field descriptions on the Pydantic model are part of the prompt — treat them carefully.
- Optional fields with `default=None` prevent the model from hallucinating missing values.
- Batch extraction is just the same primitive in a loop; use `astructured_predict` for concurrency when needed.

Chapter 19 wraps up Part II by deploying a workflow as a service with plain FastAPI — one process, no cloud.
