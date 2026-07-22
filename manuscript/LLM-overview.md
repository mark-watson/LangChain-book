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

When I wrote the first edition of this book it was difficult to run LLMs locally on my own computers. Now, in 2026, I can use Ollama to run very useful models on the old M1 8G MacBook, a new 16G MacBook Air, and an old 32G Mac Mini. I am writing this on my Mac Mini:

```console
 $ ollama list
NAME                         ID              SIZE      MODIFIED     
nomic-embed-text:latest      0a109f422b47    274 MB    25 hours ago    
gemma4:e2b-it-qat            07ea59a47401    4.3 GB    2 days ago      
qwen3.5:4b                   2a654d98e6fb    3.4 GB    2 days ago      
gemma4:26b-mlx               c8656f50f0a6    17 GB     3 days ago      
laguna-xs-2.1:latest         a8562dfd0cad    20 GB     2 weeks ago     
gemma4:12b-mlx               117d0d84cf2a    7.7 GB    2 weeks ago     
gemma4:12b-it-qat            38044be4f923    7.2 GB    2 weeks ago     
qwen3.6:35b-a3b-nvfp4-48k    8c4e86c1307e    21 GB     3 weeks ago     
qwen3.5:9b                   6488c96fa5fa    6.6 GB    4 months ago  
```

The good news is that techniques you learn now for incorporating LLMs into your own applications and you increased knowledge of and ease of writing effective prompts for LLMs will be even more useful as models become more powerful. Dear readers, my personal view is that we want the flexibility of running local models and models from multiple inference providers. We lean into running local models on llama fairly heavily in this book but in practice it is not difficult switching between local and cloud providers.

## What LLMs Are and What They Are Not

Large Language Models were originally text predictors. Now LLMs are often multi-modal operating on text, audio, photos, and even video.

Given a prompt, or context text and a prompt or question, an LLM predicts a highly likely text completion. As human beings we have a tendency to ascribe deep intelligence and world knowledge to LLMs. I try to avoid this misconception. A year ago I asked ChatGPT to write a poem about my pet parrot escaping out the window in the style of poet Elizabeth Bishop. When an friend asked that ChatGPT rewrite the poem in the style of more modern poet Billy Collins we both were surprised how closely it mimicked the styles of both poets. Surely this must be some deep form of intelligence, right? No, this phenomenon is text prediction on a model trained on most books and most web content. 

LLMs compress knowledge of language and some knowledge of the world into a compact representation. Clever software developers can certainly build useful and interesting systems using LLMs and this is the main topic of this book. My hope is that by experimenting with writing prompts, learning the differences between available models, and practicing applying LLMs to transform textual data that you will develop your own good ideas and build your own applications that you and other people find useful.

## Big Tech Businesses vs. Small Startups Using Large Language Models

Both Microsoft and Google play both sides of this business game: they want to sell cloud LLM services to developers and small startup companies and they would also like to achieve lock-in for their consumer services like Office 365, Google Docs and Sheets, etc.

Microsoft has been integrating AI technology into workplace emails, slideshows, and spreadsheets as part of its ongoing partnership with OpenAI, the company behind ChatGPT. Microsoft's Azure OpenAI service offers a powerful tool to enable these outcomes when leveraged with their data lake of more than two billion metadata and transactional elements.

Google has opened access to their Gemini Model based AI/chat search service. I have used various Google APIs for years in code I write. I have no favorites in the battle between tech giants, rather I am mostly interested in what they build that I can use in my own projects.

As I write this updated book in July or 2026 I question the long term business viability of AI companies like OpenAI and Anthropic. Both companies have very good technology but it appears that they may not be commercially viable in the long run. Dear reader, I suggest that you be flexible and write your software to easily switch models and providers. Fortunately LangChain and Lamma-Index make this flexibility easier.

Hugging Face, creates LLMs and also hosts those developed by other companies, is working on open-source rivals to ChatGPT and will [use AWS](https://iblnews.org/aws-partners-with-hugging-face-an-ai-startup-rival-to-chatgpt-working-on-open-source-models/) for that as well. Cohere AI, Anthropic, Hugging Face, FireWorks.ai, and Stability AI are some of the startups that are competing with OpenAI and Hugging Face APIs. Hugging Face is a great source of specialized models, that is, standard models that have been fine tuned for specific applications. I love that Hugging Face models can be run via their APIs and also self-hosted on our own servers and sometimes even on our laptops. Hugging Face is a fantastic resource, and you should embrace the hosting and open source flexibility it offers; this book uses Hugging Face models directly in a couple of chapters, and Ollama (itself built on top of the same open model ecosystem) in most other examples.

Dear reader, I didn't write this book for developers working at established AI companies (although I hope such people find the material here useful). I wrote this book for small developers who want to scratch their own itch by writing tools that save them time. I also wrote this book hoping that it would help developers build capabilities into the programs they design and write that rival what the big tech companies are doing.
