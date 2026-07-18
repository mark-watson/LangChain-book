# More Useful Libraries for Working with Unstructured Text Data

Here we look at examples using two libraries that I find useful for my work: EmbedChain and Kor.

## EmbedChain Wrapper for LangChain Simplifies Application Development

Taranjeet Singh developed a very nice wrapper library EmbedChain [https://github.com/embedchain/embedchain](https://github.com/embedchain/embedchain) that simplifies writing "query your own data" applications by choosing good defaults for the LangChain library.

I will show one simple example that I run on my laptop to search the contents of all of the books I have written as well as a large number of research papers. You can find my example in the GitHub repository for this book in the directory **langchain-book-examples/embedchain_test**. As usual, you will need an OpenAI API account and set the environment variable **OPENAI_API_KEY** to the value of your key.

I have copied PDF files for all of this content to the directory **~/data** on my laptop. It takes a short while to build a local vector embedding data store so I use two Python scripts. The first script **process_pdfs.py** that is shown here:

```python
# reference: https://github.com/embedchain/embedchain

from embedchain import App
import os

test_chat = App()

my_books_dir = "/Users/mark/data/"

for filename in os.listdir(my_books_dir):
    if filename.endswith('.pdf'):
        print("processing filename:", filename)
        test_chat.add("pdf_file",
                      os.path.join(my_books_dir,
                      filename))
```

Here is a demo Python script **app.py** that makes three queries:

```python
from embedchain import App

test_chat = App()

def test(q):
    print(q)
    print(test_chat.query(q), "\n")

test("How can I iterate over a list in Haskell?")
test("How can I edit my Common Lisp files?")
test("How can I scrape a website using Common Lisp?")
```

The output looks like:

```console
$ python app.py
How can I iterate over a list in Haskell?
To iterate over a list in Haskell, you can use recursion or higher-order functions like `map` or `foldl`. 

How can I edit my Common Lisp files?
To edit Common Lisp files, you can use Emacs with the Lisp editing mode. By setting the default auto-mode-alist in Emacs, whenever you open a file with the extensions ".lisp", ".lsp", or ".cl", Emacs will automatically use the Lisp editing mode. You can search for an "Emacs tutorial" online to learn how to use the basic Emacs editing commands. 

How can I scrape a website using Common Lisp?
One way to scrape a website using Common Lisp is to use the Drakma library. Paul Nathan has written a library using Drakma called web-trotter.lisp, which is available under the AGPL license at articulate-lisp.com/src/web-trotter.lisp. This library can be a good starting point for your scraping project. Additionally, you can use the wget utility to make local copies of a website. The command "wget -m -w 2 http:/knowledgebooks.com/" can be used to mirror a site with a two-second delay between HTTP requests for resources. The option "-m" indicates to recursively follow all links on the website, and the option "-w 2" adds a two-second delay between requests. Another option, "wget -mk -w 2 http:/knowledgebooks.com/", converts URI references to local file references on your local mirror. Concatenating all web pages into one file can also be a useful trick. 
```


## Kor Library

The Kor library was written by Eugene Yurtsev. Kor is useful for using LLMs to extract structured data from unstructured text. Kor works by generating appropriate prompt text to explain to GPT-3.5 what information to extract and adding in the text to be processed.

The [GitHub repository for Kor](https://github.com/eyurtsev/kor) is under active development so please check the project for updates. Here is the [documentation](https://eyurtsev.github.io/kor/).

For the following example, I modified an example in the Kor documentation for extracting dates in text.

```python
" From documentation: https://eyurtsev.github.io/kor/"

from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain.chat_models import ChatOpenAI
from pprint import pprint
import warnings ; warnings.filterwarnings('ignore')

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2000,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1.0,
)

schema = Object(
    id="date",
    description=(
        "Any dates found in the text. Should be output in the format:"
        " January 12, 2023"
    ),
    attributes = [
        Text(id = "month",
             description = "The month of the date",
             examples=[("Someone met me on December 21, 1995",
                        "Let's meet up on January 12, 2023 and discuss our yearly budget")])
    ],
)

chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')


pred = chain.predict_and_parse(text="I will go to California May 1, 2024")['data']
print("* month mentioned in text=", pred)
```

Sample output:

```console
$ python dates.py
* month mentioned in text= {'date': {'month': 'May'}}
```

Kor is a library focused on extracting data from text. You can get the same effects by writing for own prompts manually for GPT style LLMs but using Tor can save development time.
