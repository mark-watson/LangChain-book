"""Extract dates from text using LangChain 1.0 structured output.

The old kor library is deprecated. LangChain 1.0's .with_structured_output()
replaces it with native Pydantic-based extraction.
"""

from pprint import pprint

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class DateExtraction(BaseModel):
    """Dates found in the text, formatted as 'January 12, 2023'."""

    month: str = Field(description="The month of the date found in the text")
    full_date: str | None = Field(
        description="The full date in 'Month Day, Year' format if available",
        default=None,
    )


llm = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False)
structured_llm = llm.with_structured_output(DateExtraction)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract dates from the text. Format dates as 'Month Day, Year'."),
    ("human", "Extract any dates from this text:\n\n{text}"),
])

chain = prompt | structured_llm

result = chain.invoke({"text": "I will go to California May 1, 2024"})
pprint(result)
