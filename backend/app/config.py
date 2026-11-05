import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", PROJECT_ROOT / "backend" / "data"))
DATABASE_PATH = DATA_DIR / "rag.sqlite3"
UPLOADS_DIR = DATA_DIR / "uploads"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
GENERATION_MODEL = os.getenv("RAG_GENERATION_MODEL", "qwen2.5:7b")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
