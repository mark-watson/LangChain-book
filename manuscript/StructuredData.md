# LLM Techniques for Structured Data Conversion

Here we look at a simple example of converting CSV spreadsheet files to JSON but the idea of data conversion using LLMs is general purpose.

Using LLMs helps handle ambiguity. Traditional Symbolic AI methods often struggle with the nuance of human language. LLMs, with their understanding of context, can resolve ambiguity and provide more accurate extraction.

LLMs are also effective at handling complex or previously unseen formats (one shot). LLMs are trained on vast amounts of diverse text data, making them more adaptable to unexpected variations in data formats than rule-based approaches.

Using LLMs for application development can reduce manual effort by automating many parts of the conversion process that traditionally required significant human intervention and the creation of detailed extraction rules.

## Example Prompt for Converting CSV Files to JSON

In the prompt we supply a few examples for converting between these two formats:

```text
Given the example below, convert a CSV spreadsheet text file to a JSON text file:

Example:
CSV:
name,address, email
John Doe, 1234 Maple Street, Springfield,johndoe@example.com
"Jane Smith", "5678 Oak Avenue, Anytown", jane@smith764323.com
Output: 
{
  "name": "John Doe",
  "address": "1234 Maple Street, Springfield",
  "email": "johndoe@example.com"
}
{
  "name": "Jane Smith",
  "address": "5678 Oak Avenue, Anytown",
  "email": null
}

Process Text: "{input_csv}"
Output:


```

## Example Code for Converting CSV Files to JSON

The example in file **structured_data_conversion/person_data.py** reads the prompt template file and substitutes the CSV data from the test file **test.csv**. The modified prompt is passed to the OpenAI completion API:

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
with open('test.csv', 'r') as file:
    input_csv = file.read()
prompt = prompt_template.replace("input_csv", input_csv)

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

Here is the test CSV input file:

```csv
last_name,first_name,email
"Jackson",Michael,mj@musicgod.net
Jordan,Michael,"mike@retired.com"
Smith, John, john@acme41.com
```

Notice that this file is not consistent in quoting strings, hopefully making this a more general example of data you might see in the *wild*. The generated JSON looks like:

```json
{
  "last_name": "Jackson",
  "first_name": "Michael",
  "email": "mj@musicgod.net"
}
{
  "last_name": "Jordan",
  "first_name": "Michael",
  "email": "mike@retired.com"
}
{
  "last_name": "Smith",
  "first_name": "John",
  "email": "john@acme41.com"
}
```