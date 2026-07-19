# Creating an index

    uv run process_pdfs.py

Despite the script name (kept for continuity with earlier editions), `data/` holds plain
`.txt` files, not PDFs — `SimpleDirectoryReader` handles both the same way, so pointing
it at real PDFs instead needs no code change.

# Querying the indexed files

```console
$ uv run app.py
How can I iterate over a list in Haskell?
One way to iterate over a list in Haskell involves using list comprehensions where variables bind values directly from the lists provided within brackets...

How can I edit my Common Lisp files?
You can use a text editor such as Emacs or Vi to edit Common Lisp files...

How can I scrape a website using Common Lisp?
Common Lisp libraries can be managed by cloning repositories into the `~/quicklisp/local-projects/` directory...
```

Each question takes roughly a minute and a half on a local `qwen3.5:4b` — `data/` is
whole-book-length text (~130K words combined), much larger than the few-paragraph
corpus used elsewhere in this book, so retrieval and generation both have more text
to work with. `app.py` sets `similarity_top_k=1` specifically to avoid a second,
sequential refine call on top of that.
