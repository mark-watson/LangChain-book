"""Build one VectorStoreIndex per topic file in ../data/.

Returns a dict of {topic_name: query_engine} that both scripts share.
"""

from pathlib import Path

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=180.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

DATA_DIR = Path(__file__).parent.parent / "data"


def build_per_topic_engines() -> dict[str, object]:
    engines = {}
    for path in sorted(DATA_DIR.glob("*.txt")):
        topic = path.stem
        text = path.read_text(encoding="utf-8").strip()
        doc = Document(text=text, metadata={"topic": topic})
        index = VectorStoreIndex.from_documents([doc])
        engines[topic] = index.as_query_engine()
    return engines
