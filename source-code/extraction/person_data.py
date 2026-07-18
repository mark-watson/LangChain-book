"""Extract structured data from text using LangChain 1.0 structured output.

Uses .with_structured_output() with a Pydantic model — the LangChain 1.0
way to get reliable JSON from an LLM. Works with a local Ollama model.
"""

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class PersonData(BaseModel):
    """Extracted person information."""

    name: str = Field(description="The person's full name")
    address: str | None = Field(description="The person's street address", default=None)
    email: str | None = Field(description="The person's email address", default=None)


llm = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False)
structured_llm = llm.with_structured_output(PersonData)

# Read the prompt template
with open("prompt.txt") as f:
    prompt_template = f.read()

input_text = (
    "Mark Johnson enjoys living in Berkeley California at 102 Dunston Street "
    "and use mjess@foobar.com for contacting him."
)
prompt = prompt_template.replace("input_text", input_text)

result = structured_llm.invoke(prompt)
print(result)
