"""Two flavors of output parsing.

StrOutputParser is the everyday one: it pulls the raw text out of an
AIMessage so the rest of your code doesn't have to unwrap `.content`.

.with_structured_output(Pydantic) is the one that changes how you build
apps. Give the model a Pydantic schema and it hands back a validated
object, which is the fastest way to get reliable JSON out of an LLM in 2026.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

model = ChatOllama(model="qwen3.5:4b", temperature=0)

print("=== StrOutputParser ===")
prompt = ChatPromptTemplate.from_messages(
    [("human", "In one word, what color is the sky at noon?")]
)
chain = prompt | model | StrOutputParser()
print(chain.invoke({}).strip())


class Country(BaseModel):
    """Facts about a country."""

    name: str = Field(description="The country's common English name.")
    capital: str = Field(description="The country's capital city.")
    population_millions: float = Field(description="Population in millions, approximate.")


print("\n=== .with_structured_output() ===")
structured_model = model.with_structured_output(Country)

for country in ["Canada", "Germany", "Japan"]:
    result = structured_model.invoke(f"Give me facts about {country}.")
    print(result)
