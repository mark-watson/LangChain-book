"""Batch extraction: many input texts, many structured records out.

The pattern is the same as the single-shot version; you just wrap it in
a loop. In real projects, this is the shape you use to migrate a folder
of unstructured notes into a searchable database.
"""

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field


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
