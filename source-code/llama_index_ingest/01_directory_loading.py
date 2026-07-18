"""SimpleDirectoryReader with metadata enrichment.

Three things to notice:

- `file_metadata` is a callback that runs for each file and lets you add
  fields to the Document's metadata dict. Useful for anything that isn't
  derivable from the file path.
- `recursive=True` walks subdirectories.
- `exclude=[...]` skips files matching a pattern — the standard escape
  hatch for corpora that mix source code, images, and text.

Every Document that comes out of SimpleDirectoryReader already has
file_name, file_path, file_type, file_size, creation_date, and
last_modified_date populated automatically.
"""

from pathlib import Path

from llama_index.core import SimpleDirectoryReader


def add_custom_metadata(file_path: str) -> dict:
    """Runs once per file. Return whatever extra metadata you want."""
    p = Path(file_path)
    return {
        "topic": p.stem,  # "sports.txt" -> topic="sports"
        "collection": "book_data",
    }


reader = SimpleDirectoryReader(
    input_dir="../data",
    recursive=True,
    exclude=["*.png", "*.jpg", ".DS_Store"],
    file_metadata=add_custom_metadata,
)

documents = reader.load_data()

print(f"Loaded {len(documents)} documents\n")
for d in documents:
    meta = d.metadata
    print(f"file_name : {meta.get('file_name')}")
    print(f"topic     : {meta.get('topic')}")
    print(f"file_size : {meta.get('file_size')} bytes")
    print(f"first 80c : {d.text[:80]!r}")
    print()
