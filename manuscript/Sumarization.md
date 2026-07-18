# Using LLMs to Summarize Text

LLMs bring a new level of ability to text summarization tasks. With their ability to process massive amounts of information and "understand" natural language, they're able to capture the essence of lengthy documents and distill them into concise summaries. Two main types of summarization dominate with LLMs: extractive and abstractive. Extractive summarization pinpoints the most important sentences within the original text, while abstractive summarization  requires the LLM to paraphrase or generate new text to represent the core ideas. If you are interested in extractive summarization there is a chapter on this topic in my [Common Lisp AI book](https://leanpub.com/lovinglisp/read) (link to read online).

LLMs excel in text summarization for several reasons.  Their deep understanding of language semantics allows them to identify key themes, even when wording varies across a document. Additionally, they have an ability for maintaining logical consistency within summaries, ensuring that the condensed version makes sense as a cohesive unit. Modern LLMs are also trained on massive datasets encompassing diverse writing styles, helping them adapt to different sources and generate summaries tailored to specific audiences.
 
The applications of LLM-powered text summarization are vast.  They can help researchers digest lengthy scientific reports quickly, allow businesses to analyze customer feedback efficiently, or provide concise news briefs for busy individuals. LLM-based summarization also has the potential to improve accessibility, creating summaries for those with reading difficulties or summarizing complex information into simpler language. As LLM technology continues to advance, we can expect even more innovative and accurate summarization tools in the future.

## Example Prompt

In this example, the prompt is simply:

```text
Summarize the following text: "{input_text}"
Output:
```

## Code Example

The example in file **summarization/summarization_example.py** reads a prompt file and substitutes the text from the test file **../data/economics.txt**:

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
with open('../data/economics.txt', 'r') as file:
    input_text = file.read()
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

The length of the output summary is about 20% of the length of the original text.
