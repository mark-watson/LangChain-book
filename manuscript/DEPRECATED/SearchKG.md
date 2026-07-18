# Using Google's Knowledge Graph APIs With LangChain

Google's Knowledge Graph (KG) is a knowledge base that Google uses to serve relevant information in an info-box beside its search results. It allows the user to see the answer in a glance, as an instant answer. The data is generated automatically from a variety of sources, covering places, people, businesses, and more. I worked at Google in 2013 on a project that used their KG for an internal project.


Google's public Knowledge Graph Search API lets you find entities in the Google Knowledge Graph. The API uses standard schema.org types and is compliant with the JSON-LD specification. It supports entity search and lookup. 

You can use the Knowledge Graph Search API to build applications that make use of Google's Knowledge Graph. For example, you can use the API to build a search engine that returns results based on the entities in the Knowledge Graph.

In the next chapter we also use the public KGs DBPedia and Wikidata. One limitation of Google's KG APIs is that it is designed for entity (people, places, organizations, etc.) lookup. When using DBPedia and Wikidata it is possible to find a wider range of information using the SPARQL query language, such as relationships between entities. You can use the Google KG APIs to find some entity relationships, e.g., all the movies directed by a particular director, or all the books written by a particular author. You can also use the API to find information like all the people who have worked on a particular movie, or all the actors who have appeared in a particular TV show.

## Setting Up To Access Google Knowledge Graph APIs

To get an API key for Google's Knowledge Graph Search API, you need to go to the Google API Console, enable the Google Knowledge Graph Search API, and create an API key to use in your project. You can then use this API key to make requests to the Knowledge Graph Search API. 

To create your application's API key, follow these steps:

- Go to the API Console.
- From the projects list, select a project or create a new one.
- If the APIs & services page isn't already open, open the left side menu and select APIs & services.
- On the left, choose Credentials.
- Click Create credentials and then select API key.

You can then use this API key to make requests to the Knowledge Graph Search APIs.

When I use Google's APIs I set the access key in **~/.google_api_key** and read in the key using:

```python
api_key=open(str(Path.home())+"/.google_api_key").read()
```

You can also use environment variables to store access keys. Here is a code snippet for making an API call to get information about me:

```python
import json
from urllib.parse import urlencode
from urllib.request import urlopen
from pathlib import Path
from pprint import pprint

api_key =
    open(str(Path.home()) + "/.google_api_key").read()
query = "Mark Louis Watson"
service_url =
    "https://kgsearch.googleapis.com/v1/entities:search"
params = {
    "query": query,
    "limit": 10,
    "indent": True,
    "key": api_key,
}
url = service_url + "?" + urlencode(params)
response = json.loads(urlopen(url).read())
pprint(response)
```

The JSON-LD output would look like:

```console
{'@context': {'@vocab': 'http://schema.org/',
              'EntitySearchResult':
              'goog:EntitySearchResult',
              'detailedDescription':
              'goog:detailedDescription',
              'goog': 'http://schema.googleapis.com/',
              'kg': 'http://g.co/kg',
              'resultScore': 'goog:resultScore'},
 '@type': 'ItemList',
 'itemListElement': [{'@type': 'EntitySearchResult',
                      'result': {'@id': 'kg:/m/0b6_g82',
                                 '@type': ['Thing',
                                           'Person'],
                                 'description': 'Author',
                                 'name':
                                 'Mark Louis Watson',
                                 'url':
                                 'http://markwatson.com'},
                      'resultScore': 43}]}
```

In order to not repeat the code for getting entity information from the Google KG, I wrote a utility **Google_KG_helper.py** that encapsulates the previous code and generalizes it into a mini-library.

```python
"""Client for calling Knowledge Graph Search API."""

import json
from urllib.parse import urlencode
from urllib.request import urlopen
from pathlib import Path
from pprint import pprint

api_key =
    open(str(Path.home()) + "/.google_api_key").read()

# use Google search API to get information
# about a named entity:

def get_entity_info(entity_name):
    service_url =
      "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": entity_name,
        "limit": 1,
        "indent": True,
        "key": api_key,
    }
    url = service_url + "?" + urlencode(params)
    response = json.loads(urlopen(url).read())
    return response

def tree_traverse(a_dict):
    ret = []
    def recur(dict_2, a_list):
        if isinstance(dict_2, dict):
            for key, value in dict_2.items():
                if key in ['name', 'description',
                           'articleBody']:
                    a_list += [value]
                recur(value, a_list)
        if isinstance(dict_2, list):
            for x in dict_2:
                recur(x, a_list)
    recur(a_dict, ret)
    return ret


def get_context_text(entity_name):
    json_data = get_entity_info(entity_name)
    return ' '.join(tree_traverse(json_data))

if __name__ == "__main__":
    get_context_text("Bill Clinton")
```

The main test script is in the file **Google_Knowledge_Graph_Search.py**:

```python
"""Example of Python client calling the
   Knowledge Graph Search API."""

from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex
import Google_KG_helper

def kg_search(entity_name, *questions):
    ret = ""
    context_text = Google_KG_helper.get_context_text(entity_name)
    print(f"Context text: {context_text}")
    doc = Document(text=context_text)
    index = VectorStoreIndex.from_documents([doc])
    for question in questions:
        response = index.as_query_engine().query(question)
        ret += f"QUESTION:  {question}\nRESPONSE: {response}\n"
    return ret

if __name__ == "__main__":
    s = kg_search("Bill Clinton",
                  "When was Bill president?")
    print(s)
```

The example output is:

```console
$ python Google_Knowledge_Graph_Search.py
Context text: William Jefferson Clinton is an American politician who served as the 42nd president of the United States from 1993 to 2001. A member of the Democratic Party, he previously served as Governor of Arkansas from 1979 to 1981 and again from 1983 to 1992.  42nd U.S. President Bill Clinton
QUESTION:  When was Bill president?
RESPONSE: Bill Clinton was president from 1993 to 2001.
```

Accessing Knowledge Graphs from Google, DBPedia, and Wikidata allows you to integrate real world facts and knowledge with your applications. While I mostly work in the field of deep learning I frequently also use Knowledge Graphs in my work and in my personal research. I think that you, dear reader, might find accessing highly structured data in KGs to be more reliable and in many cases simpler than using web scraping.
