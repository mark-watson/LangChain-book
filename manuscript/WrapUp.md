# Book Wrap Up

This book has been fun to write. I remain as excited about LLMs and the applied libraries around them as I was when I started the first edition in early 2023, and more excited about running the whole stack on my own hardware than I was three editions ago. In 2023 running a useful LLM on a laptop was a research project. In 2026 it is a `brew install ollama` away.

What I hope you take away from the book, in one paragraph: the two most-used LLM application frameworks are both fully usable as open source libraries without touching any of the commercial platforms grown up around them. LangChain 1.0 + LangGraph 1.0 give you every primitive you need to build stateful, durable, human-in-the-loop, multi-agent systems. LlamaIndex 0.14 + Workflows give you the same coverage from a slightly different design center. Both run on your laptop, connect to any LLM (local via Ollama, hosted via any provider), and cost exactly zero in library fees.

The only cost that scales with your project is inference — the LLM calls themselves. And in 2026 even that cost is optional; a Mac mini with 32 GB of RAM can run models good enough for real work, at a fixed hardware cost, forever.

## Where the book falls short

I want to be honest about the things I skipped.

- **Advanced retrieval patterns** beyond reranking and hybrid search. Query rewriting, HyDE, dense-plus-lexical-plus-graph fusion, retrieval-time filtering by rich metadata — every one of these is a chapter I could write and did not.
- **Fine-tuning** of local models. In 2026 this has become straightforward with `unsloth` and similar tools; a follow-up book (which I may write) is where this belongs.
- **Multi-modal work.** Vision, audio, and video are increasingly first-class in both frameworks. The book stayed text-only to keep the scope tractable.
- **A serious eval / observability chapter.** Appendices B and C touch on the topic; a full treatment deserves more than an appendix.

If any of these matter enough to you that you would buy a follow-up book, let me know via the contact info on [markwatson.com](https://markwatson.com). I write my books based on what readers ask for.

## Thank you

To the readers who have followed this book across four editions and who send corrections and suggestions: thank you. Every edition has been better because of you.

To my wife Carol who handles the editing, formatting, and cover work: thank you. The book exists because you make the parts I am bad at get done.

To Harrison Chase, Jerry Liu, and the two teams building LangChain, LangGraph, and LlamaIndex: thank you for keeping the core libraries free and open source even as the commercial pressures around them grow. It is not lost on me that this book is entirely possible only because you have made that choice, and I hope enough of your users continue to reward the OSS-first stance that it stays viable.

Best regards,

Mark Watson  
Sedona, Arizona
