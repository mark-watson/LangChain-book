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

Before using LLMs directly in application code I like to experiment with prompts. Here we will use a two-shot approach of providing as context two examples of text and the extracted JSON data, followed by text we want to process. Consider the following that I ran on my old M1 8G MacBook:

```text
Given the two examples below, extract the names, addresses, and email addresses of individuals mentioned later as Process Text. Format the extracted information in JSON, with keys for "name", "address", and "email". If any information is missing, use "null" for that field.

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

This prompt is in the file **prompt_examples/two-shot-2.txt**.

The output can be overly verbose:
 
 ```console
 $ ollama run llama3:instruct < two-shot-2.txt
Here is the extracted information in JSON format:

{
"name": "Mark Johnson",
"address": "102 Dunston Street, Berkeley California",
"email": "mjess@foobar.com"
}

Note that I used the address format from Example 1, which combines the street 
address with the city and state. If you want to separate these fields into different
keys (e.g., "street", "city", "state"), let me know!
 ```
 
 While the comments the llama3-8b-instruct model makes are interesting, let’s modify the prompt to ask for concise output that only includes the generated JSON:
 
 ```text
 Given the two examples below, extract the names, addresses, and email addresses of individuals mentioned later as Process Text. Format the extracted information in JSON, with keys for "name", "address", and "email". If any information is missing, use "null" for that field. Be concise in your output by providing only the output JSON.
```

The rest of the prompt is unchanged, now the output is:

```console
$ ollama run llama3:instruct < two-shot-2.txt
{
  "name": "Mark Johnson",
  "address": "102 Dunston Street, Berkeley California",
  "email": "mjess@foobar.com"
}
```


## Example Code

To use this example we would use the same prompt except we would  make the **Process Text** a variable that is replaced before processing by an LLM. We copy the file **two-shot-2.txt** to **two-shot-2-var.txt** and change the second to the last line in the file:

```text
Process Text: "{input_text}"
```

Now let’s wrap these ideas up in a short Python example in the file **extraction/person_data.py**:

```python
import openai
from openai import OpenAI
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Read the prompt from a text file
with open('prompt.txt', 'r') as file:
    prompt_template = file.read()

# Substitute a string variable into the prompt
input_text = "Mark Johnson enjoys living in Berkeley California at 102 Dunston Street and use mjess@foobar.com for contacting him."
prompt = prompt_template.replace("input_text", input_text)

# Use the OpenAI completion API to generate a response with GPT-4
completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": prompt,
        },
    ],
)

print(completion.choices[0].message.content)
```

The output looks like:

```console
$ python person_data.py
{
  "name": "Mark Johnson",
  "address": "102 Dunston Street, Berkeley California",
  "email": "mjess@foobar.com"
}
```

For reference, the complete **completion** object looks like this:

```text
ChatCompletion(id='chatcmpl-9LBZao4hFMmw7VrYcRbQIR2EGzvCj', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='{\n  "name": "Mark Johnson",\n  "address": "102 Dunston Street, Berkeley California",\n  "email": "mjess@foobar.com"\n}', role='assistant', function_call=None, tool_calls=None))], created=1714836402, model='gpt-4-0613', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=34, prompt_tokens=223, total_tokens=257))
```
