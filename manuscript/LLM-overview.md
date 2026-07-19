# Large Language Model Overview

[Large language models](https://blogs.nvidia.com/blog/2022/10/10/llms-ai-horizon/) are a subset of artificial intelligence that use deep learning and neural networks to process natural language. [Transformers](https://www.linkedin.com/pulse/chatgpt-tip-iceberg-paul-golding) are a type of neural network architecture that can learn context in sequential data using self-attention mechanisms. They were introduced in 2017 by a team at Google Brain and have become popular for LLM research. Some older examples of [transformer-based](https://factored.ai/transformer-based-language-models/) LLMs are [BERT, GPT-3, T5 and Megatron-LM](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)).

The main points we will discuss in this book are:

- LLMs are deep learning algorithms that can understand and generate natural language based on massive datasets.
- LLMs use techniques such as self-attention, masking, and fine-tuning to learn complex patterns and relationships in language. LLMs can understand and generate natural language because they use transformer models, which are a type of neural network that can process sequential data such as text using attention mechanisms. Attention mechanisms allow the model to focus on relevant parts of the input and output sequences while ignoring irrelevant ones.
- LLMs can perform various natural language processing (NLP) and natural language generation (NLG) tasks, such as summarization, translation, prediction, classification, and question answering.
- Even though LLMs were initially developed for NLP applications, LLMs have also shown potential in other domains such as computer vision and computational biology by leveraging their generalizable knowledge and transfer learning abilities.


[BERT models](https://en.wikipedia.org/wiki/BERT_(Language_model)) are one of the first types of transformer models that were widely used. BERT was developed by Google AI Language in 2018. BERT models are a family of masked language models that use transformer architecture to learn bidirectional representations of natural language. BERT models can understand the meaning of ambiguous words by using the surrounding text as context. The "magic trick" here is that training data comes almost free because in masking models, you programmatically chose random words, replace them with a missing word token, and the model is trained to predict the missing words. This process is repeated with massive amounts of training data from the web, books, etc.

Here are some "papers with code" links for BERT (links are for code, paper links in the code repositories):

- [https://github.com/allenai/scibert](https://github.com/allenai/scibert)
- [https://github.com/google-research/bert](https://github.com/google-research/bert)

## Technological Change is Increasing at an Exponential Rate

When I wrote the first edition of this book it was difficult to run LLMs locally on my own computers. Now in May 2025, I can use Ollama to run very useful models on the old M1 8G MacBook I am writing this on:

```console
 $ ollama list
NAME                       ID              SIZE      MODIFIED    
phi4-reasoning:plus        f0ad3edce8e4    11 GB     4 days ago     
qwen3.5:4b                 e4b5fd7f8af0    5.2 GB    6 days ago     
qwen3:30b                  2ee832bc15b5    18 GB     6 days ago     
gemma3:12b-it-qat          5d4fa005e7bb    8.9 GB    2 weeks ago    
gemma3:4b-it-qat           d01ad0579247    4.0 GB    2 weeks ago    
gemma3:27b-it-qat          29eb0b9aeda3    18 GB     2 weeks ago    
openthinker:latest         4e61774f7d1c    4.7 GB    3 weeks ago    
deepcoder:latest           12bdda054d23    9.0 GB    3 weeks ago    
granite3-dense:latest      5c2e6f3112f4    1.6 GB    3 weeks ago    
nomic-embed-text:latest    0a109f422b47    274 MB    3 weeks ago    
qwen2.5:14b                7cdf5a0187d5    9.0 GB    3 weeks ago    
qwq:latest                 38ee5094e51e    19 GB     3 weeks ago    
cogito:32b                 0b4aab772f57    19 GB     3 weeks ago    
deepseek-r1:32b            38056bbcbb2d    19 GB     3 weeks ago    
mistral-small:latest       8039dd90c113    14 GB     3 weeks ago    
phi4-mini:latest           78fad5d182a7    2.5 GB    3 weeks ago    
reader-lm:latest           33da2b9e0afe    934 MB    3 weeks ago    
deepseek-r1:8b             28f8fd6cdc67    4.9 GB    3 weeks ago    
llama3.2:1b                baf6a787fdff    1.3 GB    3 weeks ago    
smollm2:latest             cef4a1e09247    1.8 GB    3 weeks ago    
unsloth_AZ_1B:latest       0b3006d8395a    807 MB    3 weeks ago    
llama3.2:latest            a80c4f17acd5    2.0 GB    3 weeks ago    
llava:7b                   8dd30f6b0cb1    4.7 GB    3 weeks ago    
```

The good news is that techniques you learn now for incorporating LLMs into your own applications and you increased knowledge of and ease of writing effective prompts for LLMs will be useful even as models become more powerful.

## What LLMs Are and What They Are Not

Large Language Models are text predictors. Given a prompt, or context text and a prompt or question, an LLM predicts a highly likely text completion. As human beings we have a tendency to ascribe deep intelligence and world knowledge to LLMs. I try to avoid this misconception. A year ago I asked ChatGPT to write a poem about my pet parrot escaping out the window in the style of poet Elizabeth Bishop. When an friend asked that ChatGPT rewrite the poem in the style of more modern poet Billy Collins we both were surprised how closely it mimicked the styles of both poets. Surely this must be some deep form of intelligence, right? No, this phenomenon is text prediction on a model trained on most books and most web content. 

LLMs compress knowledge of language and some knowledge of the world into a compact representation. Clever software developers can certainly build useful and interesting systems using LLMs and this is the main topic of this book. My hope is that by experimenting with writing prompts, learning the differences between available models, and practicing applying LLMs to transform textual data that you will develop your own good ideas and build your own applications that you and other people find useful.

## Big Tech Businesses vs. Small Startups Using Large Language Models

Both Microsoft and Google play both sides of this business game: they want to sell cloud LLM services to developers and small startup companies and they would also like to achieve lock-in for their consumer services like Office 365, Google Docs and Sheets, etc.

Microsoft has been integrating AI technology into workplace emails, slideshows, and spreadsheets as part of its ongoing partnership with OpenAI, the company behind ChatGPT. Microsoft's Azure OpenAI service offers a powerful tool to enable these outcomes when leveraged with their data lake of more than two billion metadata and transactional elements.

Google has opened access to their Gemini Model based AI/chat search service. I have used various Google APIs for years in code I write. I have no favorites in the battle between tech giants, rather I am mostly interested in what they build that I can use in my own projects.

Hugging Face, creates LLMs and also hosts those developed by other companies, is working on open-source rivals to ChatGPT and will [use AWS](https://iblnews.org/aws-partners-with-hugging-face-an-ai-startup-rival-to-chatgpt-working-on-open-source-models/) for that as well. Cohere AI, Anthropic, Hugging Face, and Stability AI are some of the startups that are competing with OpenAI and Hugging Face APIs. Hugging Face is a great source of specialized models, that is, standard models that have been fine tuned for specific applications. I love that Hugging Face models can be run via their APIs and also self-hosted on our own servers and sometimes even on our laptops. Hugging Face is a fantastic resource and even though I use their models much less frequently in this book than OpenAI APIs, you should embrace the hosting and open source flexibility of Hugging Face.
Starting in late 2023 I also stated heavily using the [Ollama](https://ollama.com) platform for downloading and running models on my laptop. There is a chapter in this book on using Ollama. In this book I most freequently use OpenAI APIs because they are so widely used.

Dear reader, I didn't write this book for developers working at established AI companies (although I hope such people find the material here useful). I wrote this book for small developers who want to scratch their own itch by writing tools that save them time. I also wrote this book hoping that it would help developers build capabilities into the programs they design and write that rival what the big tech companies are doing.
