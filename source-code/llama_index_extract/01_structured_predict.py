"""One-shot structured extraction with `structured_predict`.

Give the LLM a Pydantic schema and a prompt; get back a validated
instance of the schema. The framework handles the JSON-schema plumbing
and the retry-on-validation-error loop.
"""

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
