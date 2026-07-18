# Examples Using Hugging Face Open Source Models

To start with you will need to create a free account on the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/index) and get an API key and install:

    pip install --upgrade huggingface_hub

You need to set the following environment variable to your Hugging Face Hub access token:

    HUGGINGFACEHUB_API_TOKEN

So far in this book we have been using the OpenAI LLM wrapper:

```python
from langchain.llms import OpenAI
```

Here we will use the alternative Hugging Face wrapper class:

```python
from langchain import HuggingFaceHub
```

The LangChain library hides most of the details of using both APIs. This is a really good thing. I have had a few discussions on social tech media with people who object to the non open source nature of OpenAI. While I like the convenience of using OpenAI's APIs, I always like to have alternatives for proprietary technology I use.

The Hugging Face Hub endpoint in LangChain connects to the Hugging Face Hub and runs the models via their free inference endpoints. We need a Hugging Face account and API key to use these endpoints3. There exists two Hugging Face LLM wrappers, one for a local pipeline and one for a model hosted on Hugging Face Hub. Note that these wrappers only work for models that support the text2text-generation and text-generation tasks. Text2text-generation refers to the task of generating a text sequence from another text sequence. For example, generating a summary of a long article. Text-generation refers to the task of generating a text sequence from scratch.


## Using LangChain as a Wrapper for Hugging Face Prediction Model APIs

We will start with a simple example using the prompt text support in LangChain. The following example is in the script **simple_example.py**:

```python
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-6}
)

prompt = PromptTemplate(
    input_variables=["name"],
    template="What year did {name} get elected as president?",
)

llm_chain = LLMChain(prompt=prompt, llm=hub_llm)

print(llm_chain.run("George Bush"))
```

By changing just a few lines of code, you can run many of the examples in this book using the Hugging Face APIs in place of the OpenAI APIs.

The LangChain documentation lists the source code for a wrapper to use local Hugging Face embeddings [here](https://langchain.readthedocs.io/en/latest/_modules/langchain/embeddings/self_hosted_hugging_face.html).

## Creating a Custom LlamaIndex Hugging Face LLM Wrapper Class That Runs on Your Laptop

We will be downloading the Hugging Face model **facebook/opt-iml-1.3b** that is a 2.6 gigabyte file. This model is downloaded the first time it is requested and is then cached in **~/.cache/huggingface/hub** for later reuse.

This example is modified from an example for custom LLMs in the [LlamaIndex documentation](https://github.com/jerryjliu/llama_index/blob/main/docs/how_to/customization/custom_llms.md). Note that I have used a much smaller model in this example and reduced the prompt and output text size.

```python
# Derived from example:
#   https://gpt-index.readthedocs.io/en/latest/how_to/custom_llms.html

import time
import torch
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding
from llama_index import GPTListIndex, PromptHelper
from llama_index import LLMPredictor
from transformers import pipeline

max_input_size = 512
num_output = 64
max_chunk_overlap = 10
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

class CustomLLM(LLM):
    model_name = "facebook/opt-iml-1.3b"
    # I am not using a GPU, but you can add device="cuda:0"
    # to the pipeline call if you have a local GPU or
    # are running this on Google Colab:
    pipeline = pipeline("text-generation", model=model_name,
                        model_kwargs={"torch_dtype":torch.bfloat16})

    def _call(self, prompt, stop = None):
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=num_output)
        first_response = response[0]["generated_text"]
        # only return newly generated tokens
        returned_text = first_response[prompt_length:]
        return returned_text

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self):
        return "custom"

time1 = time.time()

# define our LLM
llm_predictor = LLMPredictor(llm=CustomLLM())

# Load the your data
documents = SimpleDirectoryReader('../data_small').load_data()
index = GPTListIndex(documents, llm_predictor=llm_predictor,
                     prompt_helper=prompt_helper)
index = index.as_query_engine(llm_predictor=llm_predictor)

time2 = time.time()
print(f"Time to load model from disk: {time2 - time1} seconds.")

# Query and print response
response = index.query("What is the definition of sport?")
print(response)

time3 = time.time()
print(f"Time for query/prediction: {time3 - time2} seconds.")
```

When running on my M1 MacBook Pro using only the CPU (no GPU or Neural Engine configuration) we can read the model from disk quickly but it takes a while to process queries:

```console
$ python hf_transformer_local.py
INFO:llama_index.token_counter.token_counter:> [build_index_from_documents] Total LLM token usage: 0 tokens
INFO:llama_index.token_counter.token_counter:> [build_index_from_documents] Total embedding token usage: 0 tokens
Time to load model from disk: 1.5303528308868408 seconds.
INFO:llama_index.token_counter.token_counter:> [query] Total LLM token usage: 182 tokens
INFO:llama_index.token_counter.token_counter:> [query] Total embedding token usage: 0 tokens

"Sport" comes from the Old French desport meaning "leisure", with the oldest definition in English from around 1300 being "anything humans find amusing or entertaining".[4]
Time for query/prediction: 228.8184850215912 seconds.
```

Even though my M1 MacBook does fairly well when I configure TensorFlow and PyTorch to use the Apple Silicon GPUs and Neural Engines, I usually do my model development using Google Colab.

Let's rerun the last example on Colab:

![](local.png)

Using a standard Colab GPU, the query/prediction time is much faster. Here is a [link to my Colab notebook](https://colab.research.google.com/drive/1Ecg-0iid3AD05zM4HgPXTVHcgkGxyi3q?usp=sharing) if you would prefer to run this example on Colab instead of on your laptop.
