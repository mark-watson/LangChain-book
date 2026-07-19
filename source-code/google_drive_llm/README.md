# Setup

Follow the directions in the book (GoogleDrive.md) to set up Google OAuth credentials
as `client_secrets.json` in this directory.

```console
$ uv sync
$ ollama pull qwen3.5:4b
$ uv run fetch_txt_files.py
$ uv run index_and_QA.py
```

Note: this directory ships a `data/sports.txt` sample file, so you can run
`index_and_QA.py` immediately without setting up Google auth first. Only
`fetch_txt_files.py` needs real Google Drive credentials.
