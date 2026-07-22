# Using LLMs to Summarize Text

LLMs bring a new level of ability to text summarization tasks. With their ability to process massive amounts of information and "understand" natural language, they're able to capture the essence of lengthy documents and distill them into concise summaries. Two main types of summarization dominate with LLMs: extractive and abstractive. Extractive summarization pinpoints the most important sentences within the original text, while abstractive summarization requires the LLM to paraphrase or generate new text to represent the core ideas. If you are interested in extractive summarization there is a chapter on this topic in my [Common Lisp AI book](https://leanpub.com/lovinglisp/read) (link to read online).

LLMs excel in text summarization for several reasons. Their deep understanding of language semantics allows them to identify key themes, even when wording varies across a document. Additionally, they have an ability for maintaining logical consistency within summaries, ensuring that the condensed version makes sense as a cohesive unit. Modern LLMs are also trained on massive datasets encompassing diverse writing styles, helping them adapt to different sources and generate summaries tailored to specific audiences.

The applications of LLM-powered text summarization are vast. They can help researchers digest lengthy scientific reports quickly, allow businesses to analyze customer feedback efficiently, or provide concise news briefs for busy individuals. LLM-based summarization also has the potential to improve accessibility, creating summaries for those with reading difficulties or summarizing complex information into simpler language.

## Example Prompt

In this example, the prompt is simply:

```text
Summarize the following text: "{input_text}"
Output:
```

## Code Example

Everything lives in `source-code/summarization/`. Setup:

```console
$ cd source-code/summarization
$ uv sync
$ ollama pull qwen3.5:4b
```

The example in file **summarization/summarization_example.py** reads the prompt file above and substitutes the text from **../data/economics.txt**, the same chemistry/economics/health/sports corpus used throughout the book:

```python
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# Read the prompt template
prompt_template = Path("prompt.txt").read_text()

# Read the input text
input_text = Path("../data/economics.txt").read_text()
prompt = prompt_template.replace("input_text", input_text)

# Use a local Ollama model
llm = ChatOllama(model="qwen3.5:4b", temperature=0)

response = llm.invoke([HumanMessage(content=prompt)])
print(response.content)
```

No API key, no hosted service: `ChatOllama` and a single `HumanMessage` are the whole call. `Path.read_text()` replaces the `open(...)`/`file.read()` pattern you may be used to from other languages; for two small file reads it is the more idiomatic choice in current Python.

The output:

```console
$ uv run summarization_example.py
The text defines economics as a social science that analyzes the production, distribution, and consumption of goods and services by studying how economic agents manage scarce resources to achieve desired ends. It distinguishes between microeconomics (individuals and markets) and macroeconomics (the overall economy). A significant portion focuses on the Austrian School, which emphasizes subjective human choices, price mechanisms, laissez-faire policies, and minimal government intervention, founded by Carl Menger, Eugen von Böhm-Bawerk, and Ludwig von Mises. The text also notes that economics has been professionalized since around 1900 through graduate programs in universities.
```

The source file is 550 words; this summary is 92, roughly a sixth of the original, and it correctly pulls out the document's actual structure (the micro/macro distinction, the Austrian School's specific claims and founders, the professionalization note) rather than just trimming sentences from the top. That is the abstractive/extractive distinction from the introduction in practice: nothing here is a verbatim sentence lifted from the source, it is a paraphrase built from the model's understanding of the whole document.
