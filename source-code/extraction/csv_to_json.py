"""Convert CSV rows to structured JSON using LangChain 1.0 structured output.

Extends the single-object pattern in person_data.py: here the model
extracts a *list* of Pydantic objects from a whole CSV blob in one call,
instead of one object from one paragraph of prose.
"""

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


class PersonRecord(BaseModel):
    """One row of the extracted CSV data."""

    last_name: str = Field(description="The person's last name")
    first_name: str = Field(description="The person's first name")
    email: str | None = Field(description="The person's email address", default=None)


class PersonRecords(BaseModel):
    """All rows extracted from the CSV text."""

    people: list[PersonRecord]


llm = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False)
structured_llm = llm.with_structured_output(PersonRecords)

with open("test.csv") as f:
    input_csv = f.read()

prompt = (
    "Convert the following CSV data to structured records. "
    "The file is not consistent about quoting or spacing:\n\n"
    f"{input_csv}"
)

result = structured_llm.invoke(prompt)
for person in result.people:
    print(person)
