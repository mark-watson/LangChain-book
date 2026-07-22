# Extraction of Facts and Relationships from Text Data

Traditional methods for extracting email addresses, names, addresses, etc. from text included the use of hand-crafted regular expressions and custom software. LLMs are text processing engines with knowledge of grammar, sentence structure, and some real world embedded knowledge. Using LLMs can reduce the development time of information extraction systems.

## Key Capabilities of LLMs for Fact and Relationship Extraction

- Named Entity Recognition (NER): LLMs excel at identifying and classifying named entities within text. This includes pinpointing people, organizations, locations, dates, quantities, etc. NER forms the basis of any fact extraction process, as entities are the core elements around which facts are organized.
- Relationship Extraction (RE): LLMs are adept at understanding the grammatical structure of sentences and the contextual meaning of words. This enables them to identify relationships between the entities they've identified, such as: Employment relationships ("Jane Smith works for Microsoft") Ownership ("Apple acquired Beats Electronics") and Location-based relationships ("The Louvre Museum is located in Paris")
- Semantic Understanding: LLMs possess a deep understanding of language semantics. This allows them to go beyond simple keyword matching and grasp the nuances and implicit meanings within text, leading to more accurate fact extraction.
- Knowledge Base Augmentation: Pre-trained LLMs draw on their vast knowledge bases (from being trained on massive text datasets) to fill in gaps when text is incomplete and support the disambiguation of entities or relationships.

## Techniques and Approaches

- Fine-tuned Question Answering: LLMs can be fine-tuned to directly answer factual questions posed based on a text. For example, given a news article and the question, "When did the event occur?", the LLM can pin down the relevant date within the text.
- Knowledge Graph Construction: LLMs play a crucial role in automatically constructing knowledge graphs. These graphs are structured representations of facts and relationships extracted from text. LLMs identify the entities, relationships, and help enrich the graphs with relevant attributes.
- Zero-shot or Few-shot Learning: Advanced LLMs can extract certain facts and relationships with minimal or no additional training examples. This is especially valuable in scenarios where manually labelled data is scarce or time-consuming to create.

## Benefits

- Accuracy: LLMs often surpass traditional rule-based systems in accuracy, particularly when working with complex or varied text formats.
- Scalability: LLMs can process vast amounts of text data to efficiently extract facts and relationships, enabling the analysis of large-scale datasets.
- Time-saving: The ability of LLMs to adapt and learn reduces the need for extensive manual rule creation or feature engineering, leading to faster development of fact extraction systems.

## Applications

- Financial Analysis: Identifying key facts and relationships within financial reports and news articles to support investment decisions.
- Legal Research: Extracting relevant clauses, case law, and legal relationships from complex legal documents.
- Scientific Literature Analysis: Building databases of scientific findings and discoveries by extracting relationships and networks from research papers.
- Customer Support: Analyzing customer feedback and queries to understand product issues, sentiment, and commonly reported problems.

## Example Prompts for Getting Information About a Person from Text and Generating JSON

Before using LLMs directly in application code I like to experiment with prompts. Here we will use a two-shot approach of providing as context two examples of text and the extracted JSON data, followed by text we want to process:

```text
Given the two examples below, extract the names, addresses, and email addresses of individuals mentioned later as Process Text. Format the extracted information in JSON, with keys for "name", "address", and "email". If any information is missing, use "null" for that field. Be concise in your output by providing only the output JSON.

Example 1:
Text: "John Doe lives at 1234 Maple Street, Springfield. His email is johndoe@example.com."
Output: 
{
  "name": "John Doe",
  "address": "1234 Maple Street, Springfield",
  "email": "johndoe@example.com"
}

Example 2:
Text: "Jane Smith has recently moved to 5678 Oak Avenue, Anytown. She hasn't updated her email yet."
Output: 
{
  "name": "Jane Smith",
  "address": "5678 Oak Avenue, Anytown",
  "email": null
}

Process Text: "Mark Johnson enjoys living in Berkeley California at 102 Dunston Street and use mjess@foobar.com for contacting him."
Output:
```

This prompt is in the file **prompt_examples/two-shot-2.txt**, and it already includes one instruction worth noticing: "Be concise in your output by providing only the output JSON." Run it against a local model:

```console
$ cd source-code/prompt_examples
$ ollama run qwen3.5:4b --think=false < two-shot-2.txt
{
  "name": "Mark Johnson",
  "address": "102 Dunston Street, Berkeley California",
  "email": "mjess@foobar.com"
}
```

Clean, bare JSON, with nothing to strip before you can hand it to `json.loads()`. That is not the default behavior. Drop the "be concise" sentence (the file **two-shot-2-verbose.txt** is the same prompt without it) and run it again:

```console
$ ollama run qwen3.5:4b --think=false < two-shot-2-verbose.txt
```json
{
  "name": "Mark Johnson",
  "address": "Berkeley California, 102 Dunston Street",
  "email": "mjess@foobar.com"
}
```
```

Same facts, but now wrapped in a Markdown code fence: harmless to a human reading a terminal, and a `JSONDecodeError` waiting to happen in code that expects bare JSON. One sentence in the prompt is the whole difference.

`--think=false` disables `qwen3.5`'s reasoning trace at the CLI level. Without it, `ollama run` prints several paragraphs of visible "thinking" before either answer; try it once to see what it looks like, then leave the flag on for anything you actually want to read. `person_data.py` below sets the same behavior through the Python API with `thinking=False`.

## Example Code

To use this example we make the **Process Text** a variable that is replaced before processing by an LLM. Copy **two-shot-2.txt** to **two-shot-2-var.txt** and change the second-to-last line in the file:

```text
Process Text: "{input_text}"
```

Now let's wrap these ideas up in a short Python example. `source-code/extraction/person_data.py` uses LangChain 1.0's structured output instead of hand-parsing JSON out of a text completion: you give it a Pydantic model, and `.with_structured_output()` handles getting the model to fill it in correctly:

```python
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


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
```

`prompt.txt` here is the same two-shot template as `two-shot-2-var.txt`, just kept alongside the script instead of in `prompt_examples/`. `.with_structured_output(PersonData)` wraps the model so that instead of returning an `AIMessage` you get back a validated `PersonData` instance directly: no JSON parsing, no code fence to strip, no missing-field bugs, because Pydantic raises immediately if the model's output does not fit the schema. This is strictly more reliable than the prompt-only approach above; the two-shot examples in the prompt still help, but the schema is now enforced in code rather than requested in English.

The output looks like:

```console
$ uv run person_data.py
name='Mark Johnson' address='102 Dunston Street, Berkeley California' email='mjess@foobar.com'
```

That is Pydantic's default `repr()` for the `PersonData` object; `result.name`, `result.address`, and `result.email` are ordinary typed attributes, ready to use without any parsing step.

## From One Record to Many: CSV to JSON

The same technique extends past single paragraphs of prose. `source-code/extraction/csv_to_json.py` takes a whole CSV file (or several rows at once) and extracts a *list* of structured records in a single call, by wrapping the per-row model in a container model:

```python
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
```

`test.csv` is deliberately messy (inconsistent quoting, inconsistent spacing), the kind of file you actually get from someone's ad hoc export rather than a clean data pipeline:

```csv
last_name,first_name,email
"Jackson",Michael,mj@musicgod.net
Jordan,Michael,"mike@retired.com"
Smith, John, john@acme41.com
```

A hand-written CSV parser would need explicit rules for the quoting inconsistencies and the stray leading space before `John`. The LLM does not care: it reads the file as text and fills in the schema:

```console
$ uv run csv_to_json.py
last_name='Jackson' first_name='Michael' email='mj@musicgod.net'
last_name='Jordan' first_name='Michael' email='mike@retired.com'
last_name='Smith' first_name='John' email='john@acme41.com'
```

`PersonRecords.people` is a plain Python list of `PersonRecord` objects, with no per-row LLM calls, no manual JSON assembly, and the same reliability guarantee as the single-record example: if the model's response does not fit the schema, Pydantic raises rather than handing you a malformed record. For a handful of rows like this, one call is plenty; for a much larger file you would chunk it and call `.with_structured_output()` once per chunk rather than trying to fit thousands of rows in a single prompt.

## What we covered

- A two-shot prompt is a fast way to prototype an extraction task before writing any code, and small instructions inside it (like "be concise") change the shape of the output, not just its tone.
- `.with_structured_output(PydanticModel)` moves schema enforcement from the prompt into code: the model still does the extraction, but a malformed response raises instead of silently producing bad JSON.
- The same pattern extracts one object from a paragraph or a list of objects from a whole file: wrap the per-item model in a container model and let the LLM read the input as unstructured text, regardless of how messy its formatting is.
