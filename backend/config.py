import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "pdf-rag-app"))
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "")
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "storage" / "chroma"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "storage" / "uploads"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "storage" / "results"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_rag_collection")

os.environ.setdefault("LANGSMITH_TRACING", LANGSMITH_TRACING)
os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGSMITH_TRACING)
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)

if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)

if LANGSMITH_ENDPOINT:
    os.environ.setdefault("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)

for path in [CHROMA_DIR, UPLOAD_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
