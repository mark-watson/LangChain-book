"""Web search via DuckDuckGo (free, no API key needed).

The old GoogleSerperAPIWrapper required a paid SerpAPI key.
DuckDuckGo is the book's default free search tool.
"""

from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()


def search_web(query):
    return search.invoke(query)


print(search_web("What is the capital of Arizona?"))
