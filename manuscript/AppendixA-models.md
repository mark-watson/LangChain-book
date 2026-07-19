# Appendix A. Model sizes and what they're good for

This is deliberately not a "best model of the month" appendix. Any concrete recommendation I make here will be dated by the time you read it. Instead, this is a stable-over-time discussion of what different *sizes* of models tend to be good and bad at, when a hosted model is worth the token cost, and how to think about the different *roles* a model can play in an application.

## Local model sizes

I group local models into three tiers based on parameter count, which is the closest thing to a size-independent quality proxy that exists.

**Small (roughly 1-4B parameters).** In 2026 the strongest representatives are `llama3.2:3b`, `gemma3:4b-it-qat`, `phi4-mini`, and `smollm2`. RAM footprint: 2-4 GB. Inference speed on Apple Silicon or a modern NVIDIA card: often over 40 tokens per second.

These are good at: short-form generation, focused classification, on/off-topic filtering (the classic "reply Y or N" gate from the local-search pipeline in Chapter 13), simple prompt-following, generating small structured outputs.

These are bad at: multi-hop reasoning, sustained tool use, keeping their place in long conversations, handling ambiguity gracefully. A 3B model that gets a tool-calling scenario wrong will often silently produce prose describing what it *would* do instead of actually calling the tool. This failure mode is one of the most common gotchas in the whole book; if you see it, size up.

Use small models for the parts of a pipeline where speed matters more than quality: relevance filters, one-shot classifiers, routing decisions with only a few options.

**Medium (roughly 7-14B parameters).** In 2026 the strongest representatives are `qwen3:8b`, `gemma3:12b-it-qat`, `mistral-small`, and `deepseek-r1:8b`. RAM footprint: 5-9 GB. Inference speed: usually 20-40 tokens per second on the same hardware.

These are the workhorses. Every LangGraph and LlamaIndex agent in this book was tested against `qwen3.5:4b` because it hits a sweet spot: reliably supports tool calling, follows multi-step system prompts, handles a few thousand tokens of context without losing its place, produces reasonable structured output. Not as smart as the largest local models or hosted models, but the drop is usually the difference between "gets it right" and "gets it right and explains it well."

Use medium models for anything with tools, anything with structured output requirements, anything with a system prompt longer than a paragraph, anything where you would be embarrassed if the model got the answer wrong.

**Large (roughly 27B+ parameters).** In 2026 the strongest representatives are `qwen3:30b`, `gemma3:27b-it-qat`, `deepseek-r1:32b`, `mistral-small:24b`. RAM footprint: 15-20 GB. Inference speed: 8-15 tokens per second on a laptop. On a workstation with 32-40 GB of GPU memory, faster.

These are close to hosted-model quality on many tasks. Multi-hop reasoning holds up. Tool calling is robust. Long-context handling is genuine, not simulated. The tradeoff is that they are slow enough that streaming becomes important — waiting 30 seconds for a first token to arrive is not viable for interactive use.

Use large models when quality is your bottleneck: research assistants, complex agent orchestration, high-stakes structured extraction. Or when you specifically need to keep everything local for compliance or privacy.

## When to pay for a hosted model

Local models cover more ground than most developers expect. Hosted models still earn their token cost in specific situations:

- **You need the very best available quality** and you have concrete evidence a local model is not good enough. Vibes are not evidence; make sure you can point to a specific failure case that hosted models handle and local ones do not.
- **You have unpredictable, spiky workloads** that would need a big GPU idle most of the time. Pay-per-token beats renting a GPU 24/7.
- **You need a specific capability only some hosted models have.** Vision, extremely long context (millions of tokens), specialized fine-tunes, or an unusually strong specific benchmark.
- **You need multiple people to share access** to the same "AI" without setting up infrastructure.

Among hosted providers the book mentions three: Gemini, Fireworks.ai, and OpenAI. All three offer OpenAI-compatible chat APIs; all three are competitive on quality and pricing in 2026. The choice between them mostly comes down to which prices happen to be lowest this quarter for your specific token mix. None of them is a lock-in decision — LangChain and LlamaIndex both wrap them uniformly.

## Roles a model can play

The word "model" hides several distinct roles in an application. Different sizes and different specific models are appropriate for different roles.

**Chat / instruction-following.** The role the whole book focuses on. Any medium or larger model works. Look for `-instruct` or `-chat` suffixes on Ollama model names.

**Tool calling.** A specialization of chat. Not every chat-capable model supports tool calling well. As of mid-2026 the ones I use are qwen3, llama3.x, gemma3, and mistral-small. If your model of choice does not appear in tool-calling benchmarks, assume it does not support tools and pick a different one.

**Embedding.** Turning text into vectors. Not related to the chat model in your pipeline; picked separately. Good defaults for local use: `BAAI/bge-small-en-v1.5` (fast, small), `BAAI/bge-base-en-v1.5` (slower, better). Chapter 15 discusses the tradeoffs.

**Reranking.** Cross-encoder that scores query-document pairs for relevance. Chapter 17 covers this. Default: `BAAI/bge-reranker-base`. Nothing to do with your chat model; runs on CPU or GPU independently.

**LLM judge.** A specialization of chat: an LLM asked to evaluate another LLM's output. Appendix B covers this. Use the strongest model you have available for the judge, because the judgment quality directly bounds the evaluation quality.

## Rules of thumb

- **Default to medium local** unless you have a specific reason not to. Fast enough for interactive use, capable enough for most real applications, free.
- **Size up when tools misbehave.** Silent tool-not-called failures are the most common reason to move from small to medium or medium to large.
- **Keep embeddings and rerankers local.** Neither the quality nor the cost of hosted alternatives justifies the dependency.
- **Match model to role, not to project.** A pipeline can use a small model for filtering, a medium model for tool calling, and a large model for final synthesis. That is often cheaper *and* higher-quality than using one big model everywhere.
- **When in doubt, prototype with `qwen3.5:4b`.** It is not the best model at any specific thing but it is the least likely to embarrass you across the range of things the book covers.
