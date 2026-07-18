# Running Local LLMs Using Llama.cpp and LangChain

We saw an example at the end of the last chapter running a local LLM. Here we use the [Llama.cpp](https://github.com/ggerganov/llama.cpp) project to run a local model with LangChain. I write this in October 2023 about six months after I wrote the previous chapter. While the examples in the last chapter work very well if you have an NVIDIA GPU, I now prefer using Llama.cpp because it also works very well with Apple Silicon. My Mac has a M2 SOC with 32G of internal memory which is suitable for running fairly large LLMs efficiently.

## Installing Llama.cpp with a Llama2-13b-orca Model

Now we look an an approach to run LLMs locally on your own computers.

Among the many open and public models, I chose Hugging Face's Llama2-13b-orca model because of its support for natural language processing tasks. The combination of Llama2-13b-orca with the llama.cpp library is well supported by LangChain and will meet our requirements for local deployment and ease of installation and use.

Start by cloning the **llama.cpp** project and building it:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
make
mkdir models
```

Then get a model file from [https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-8K-3319-GGUF](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-8K-3319-GGUF) and copy to **./models** directory:


```bash
$ ls -lh models
8.6G openassistant-llama2-13b-orca-8k-3319.Q5_K_M.gguf
```

It is not strictly required for you to clone Llama.cpp from GitHub because the LangChain library includes full support for encapsulating Llama.cpp via the llama-cpp-python library. That said, you can also run Llama.cpp from the command line and it includes a REST server option and I find it useful beyond the requirements for the example in this chapter.

Note that there are many different variations of this model that trade off quality for memory use. I am using one of the larger models. If you only have 8G of memory try a smaller model.

## Python Example

The following script is in the file **langchain-book-examples/llama.cpp/test.py** and is derived from the LangChain documentation: [https://python.langchain.com/docs/integrations/llms/llamacpp](https://python.langchain.com/docs/integrations/llms/llamacpp).

We start by importing the following modules and classes from the **langchain** library: **LlamaCpp**, **PromptTemplate**, **LLMChain**, and callback-related entities. An instance of **PromptTemplate** is then created with a specified template that structures the input question and answer format. A **CallbackManager** instance is established with **StreamingStdOutCallbackHandler** as its argument to facilitate token-wise streaming during the model's inference, which is useful for seeing text as it is generated.

We then create an instance of the **LlamaCpp** class with specified parameters including the model path, temperature, maximum tokens, and others, along with the earlier created **CallbackManager** instance. The **verbose** parameter is set to True, implying that detailed logs or outputs would be provided during the model's operation, and these are passed to the **CallbackManager**. The script then defines a new prompt regarding age comparison and invokes the **LlamaCpp** instance with this prompt to generate and output a response.

```python
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/markw/llama.cpp/models/openassistant-llama2-13b-orca-8k-3319.Q5_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager, 
    verbose=True, # Verbose for callback manager
)

prompt = """
Question: If Mary is 30 years old and Bob is 25, who is older and by how much?
"""
print(llm(prompt))
```
 
Here is example output (with output shortened for brevity):

```
 $ p test.py
llama_model_loader: loaded meta data with 20 key-value pairs and 363 tensors from /Users/markw/llama.cpp/models/openassistant-llama2-13b-orca-8k-3319.Q5_K_M.gguf (version GGUF V2 (latest))

My Answer: Mary is older by 5 years.
A more complete answer should be: "To determine whether Mary or Bob is older, first find the difference in their ages. This can be done by subtracting the smaller number from the larger number. 
For example, let's say Mary is 30 years old and Bob is 25 years old. To find out who is older, we need to subtract Bob's age (25) from Mary's age (30). The answer is 5. Therefore, Mary is 5 years older than Bob."
```

While using APIs from OpenAI, Anthropic, and other providers is simple and frees developers from the requirements for running LLMs, new tools like Llama.cpp make it easier and less expensive   to run and deploy LLMs yourself. My preference, dear reader, is to have as much control as possible over software and systems that I depend on and experiment with.

