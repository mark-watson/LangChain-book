# Using LLMs To Organize Information in Our Google Drives

My digital life consists of writing, working as an AI practitioner, and learning activities that I justify with my self-image of a "gentleman scientist." Cloud storage like GitHub, Google Drive, Microsoft OneDrive, and iCloud are central to my activities.

About ten years ago I spent two months of my time writing a system in Clojure that was planned to be my own custom and personal DropBox, augmented with various NLP tools and a FireFox plugin to send web clippings directly to my personal system. To be honest, I stopped using my own project after a few months because the time it took to organize my information was a greater opportunity cost than the value I received.

In this chapter I walk through a small system for pulling text files out of Google Drive and making them queryable with a local LLM. It is deliberately minimal — a fetch script and an index-and-query script — but it is the same shape as anything larger you would build on top of your own Drive.

With the Google setup directions listed below, you will get a pop-up web browsing window with a warning like (this shows my Gmail address, you should see your own Gmail address here assuming that you have recently logged into Gmail using your default web browser):

![](gwarning.png)

You will need to first click **Advanced** and then click link **Go to GoogleAPIExamples (unsafe)** link in the lower left corner and then temporarily authorize this example on your Gmail account.

## Setting Up Requirements

You need to create a credential at [https://console.cloud.google.com/cloud-resource-manager](https://console.cloud.google.com/cloud-resource-manager) (changing application type to "Desktop"):

- Search for 'Google Drive API', select the entry, and click 'Enable'.
- Select 'Credentials' from the left menu, click 'Create Credentials', select 'OAuth client ID'.
- Now, the product name and consent screen need to be set -> click 'Configure consent screen' and follow the instructions. Once finished:
- Select 'Application type' to be Desktop application.
- Enter an appropriate name.
- Input `http://localhost:8080` for 'Authorized JavaScript origins'.
- Input `http://localhost:8080/` for 'Authorized redirect URIs'.
- Click 'Save'.
- Click 'Download JSON' on the right side of Client ID. Google names the download `client_secret_<really long ID>.json` — **rename it to exactly `client_secrets.json`** and copy it into the `source-code/google_drive_llm/` directory. The script below looks for that exact filename and prints setup instructions instead of a stack trace if it is missing.

## Write Utility To Fetch All Text Files From Top Level Google Drive Folder

For this example we will just authenticate our test script with Google, and copy all top level text files with names ending with ".txt" to the local file system in subdirectory **data**. The library doing the Google Drive work is `pydrive2` — the maintained fork of the original `pydrive`, which stopped receiving updates some years ago. The code is in the directory **google_drive_llm** in file **fetch_txt_files.py**:

```python
"""Fetch .txt files from Google Drive using PyDrive2.

PyDrive2 is the maintained successor to PyDrive. Requires Google OAuth
credentials set up per the PyDrive2 documentation.
"""

import sys
from pathlib import Path

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

_SETUP_INSTRUCTIONS = """
Google OAuth credentials not found (client_secrets.json is missing).

To set up access to Google Drive:

  1. Go to the Google Cloud Console and create a project:
     https://console.cloud.google.com/

  2. Enable the Google Drive API for your project:
     https://console.cloud.google.com/apis/library/drive.googleapis.com

  3. Create OAuth 2.0 credentials (Desktop app type) and download
     the JSON file, saving it as 'client_secrets.json' in this directory.

  4. Full quickstart guide:
     https://developers.google.com/drive/api/quickstart/python

Then re-run this script — a browser window will open to complete sign-in.
"""


def get_txt_files(drive, dir_id="root"):
    """Get all plain text files with .txt extension in a Google Drive directory."""

    file_list = drive.ListFile(
        {"q": f"'{dir_id}' in parents and trashed=false"}
    ).GetList()
    for file1 in file_list:
        print("title: %s, id: %s" % (file1["title"], file1["id"]))
    return [
        [file1["title"], file1["id"], file1.GetContentString()]
        for file1 in file_list
        if file1["title"].endswith(".txt")
    ]


def test():
    if not Path("client_secrets.json").exists():
        print(_SETUP_INSTRUCTIONS)
        sys.exit(0)

    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    fl = get_txt_files(drive)
    for f in fl:
        print(f)
        with open("data/" + f[0], "w") as fh:
            fh.write(f[2])


if __name__ == "__main__":
    test()
```

Two differences from an OAuth-less script worth noting. `get_txt_files` now takes `drive` as a parameter instead of closing over a module-level global — easier to test, and it makes the dependency on a successful auth step explicit. And `test()` checks for `client_secrets.json` before touching the network at all, so a reader who has not done the Google Cloud Console setup yet gets a short list of what to do next instead of an OAuth stack trace.

For testing I have one text file, `sports.txt`, that I want copied from my Google Drive. Running the script opens a browser for the OAuth consent flow, then lists and downloads every top-level `.txt` file:

```console
$ uv run fetch_txt_files.py
Your browser has been opened to visit:

    https://accounts.google.com/o/oauth2/auth?client_id=...&redirect_uri=http%3A%2F%2Flocalhost%3A8080%2F&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive&access_type=offline&response_type=code

Authentication successful.

title: testdata, id: 1TZ9bnL5XYQvKACJw8VoKWdVJ8jeCszJ
title: sports.txt, id: 18RN4ojvURWt5yoKNtDdAJbh4fvmRpzwb
 ...
['sports.txt', '18RN4ojvURWt5yoKNtDdAJbh4fvmRpzwb', 'Sport is generally recognised as activities based in physical athleticism or physical dexterity...']
```

That is a representative run against my own Drive, file IDs abbreviated — yours will list whatever `.txt` files sit at your Drive's top level. I have not re-run the OAuth flow itself for this edition, since it needs a real Google account and browser-based consent — exactly the kind of per-reader setup this book otherwise avoids — but `LocalWebserverAuth()`'s console output is standard PyDrive2 behavior and has been stable across versions.

## Generate Vector Indices for Files in Specific Google Drive Directories

The script in the last section should have created copies of your Drive's `.txt` files in `data/`. `index_and_QA.py` indexes that directory with LlamaIndex and answers a question against it — entirely locally, no OpenAI key required:

```python
"""Index Google Drive text files and answer questions with LlamaIndex 0.14.

Uses local HuggingFace embeddings and a local Ollama LLM.
Run fetch_txt_files.py first to download the .txt files into ./data/.
"""

import sys
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

data_dir = Path("data")
if not data_dir.exists() or not any(data_dir.iterdir()):
    print(
        "No files found in ./data/\n\n"
        "Run fetch_txt_files.py first to download your Google Drive .txt files.\n"
        "That script requires Google OAuth credentials (client_secrets.json).\n"
        "Setup guide: https://developers.google.com/drive/api/quickstart/python"
    )
    sys.exit(0)

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

print(query_engine.query("What is the definition of sport?"))
```

Two LlamaIndex-era details worth calling out, both covered in more depth in Part II of this book: `GPTSimpleVectorIndex` and `save_to_disk`/`load_from_disk` — the API this chapter used in earlier editions — no longer exist. `VectorStoreIndex.from_documents(...)` is the current equivalent, and `Settings.llm` / `Settings.embed_model` configure the LLM and embedding model globally instead of threading a `ServiceContext` through every call. And there is no `OPENAI_API_KEY` anywhere: the embedding model is a small local `HuggingFaceEmbedding`, and the LLM is a local Ollama model, so this whole example runs offline once the files are on disk.

To try this without setting up Google OAuth at all, this repository ships a `data/sports.txt` sample file — the same file used as the worked example below — so you can run `index_and_QA.py` immediately and only deal with `fetch_txt_files.py` once you actually want to pull your own Drive contents.

Output, run against that sample file:

```console
$ uv run index_and_QA.py
Sport is generally recognized as activities based in physical athleticism or dexterity, typically governed by rules to ensure fair competition and consistent adjudication of the winner. The term originates from Old French *desport* meaning "leisure," with an English definition dating back around 1300 being anything humans find amusing or entertaining. Additionally, some bodies advocate widening this definition to include all physical activity and exercise, including those completed just for fun.
```

It is interesting to see how the query result is rewritten in a nice form, compared to the raw text in `data/sports.txt`:

```console
$ cat data/sports.txt 
Sport is generally recognised as activities based in physical athleticism or physical dexterity.[3] Sports are usually governed by rules to ensure fair competition and consistent adjudication of the winner.

"Sport" comes from the Old French desport meaning "leisure", with the oldest definition in English from around 1300 being "anything humans find amusing or entertaining".[4]

Other bodies advocate widening the definition of sport to include all physical activity and exercise. For instance, the Council of Europe include all forms of physical exercise, including those completed just for fun.
```

## Google Drive Example Wrap Up

If you already use Google Drive to store your working notes and other documents, then you might want to expand the simple example in this chapter to build your own query system for your documents. In addition to Google Drive, I also use Microsoft Office 365 and OneDrive in my work and personal projects.

I haven't written my own connectors yet for OneDrive but this is on my personal to-do list using the Microsoft library [https://github.com/OneDrive/onedrive-sdk-python](https://github.com/OneDrive/onedrive-sdk-python).
