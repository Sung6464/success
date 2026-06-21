"""Central configuration for proxy RAG. Mapped to main Config class."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Import Config from root config.py
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import Config

# Models
GEN_MODEL = Config.AZURE_OPENAI_LLM_MODEL or "gpt-4.1"
EMBED_MODEL = Config.AZURE_OPENAI_EMBEDDING_MODEL or "text-embedding-3-large"
EMBED_DIM = int(os.getenv("PP_EMBED_DIM", "3072"))

# Keys / Endpoints
OPENAI_API_KEY           = Config.OPENAI_API_KEY or ""
AZURE_OPENAI_API_KEY     = Config.AZURE_OPENAI_LLM_API_KEY or ""
AZURE_OPENAI_ENDPOINT    = Config.AZURE_OPENAI_LLM_API_BASE or ""
AZURE_OPENAI_API_VERSION = Config.AZURE_OPENAI_LLM_API_VERSION or "2024-12-01-preview"

AZURE_OPENAI_EMBEDDING_API_KEY     = Config.AZURE_OPENAI_EMBEDDING_API_KEY or ""
AZURE_OPENAI_EMBEDDING_ENDPOINT    = Config.AZURE_OPENAI_EMBEDDING_API_BASE or ""
AZURE_OPENAI_EMBEDDING_API_VERSION = Config.AZURE_OPENAI_EMBEDDING_API_VERSION or "2024-02-01"
AZURE_EMBED_DEPLOYMENT             = Config.AZURE_OPENAI_EMBEDDING_MODEL or ""

# Azure AI Document Intelligence
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or ""
DOCUMENT_INTELLIGENCE_KEY      = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY") or ""

def has_document_intelligence() -> bool:
    return bool(DOCUMENT_INTELLIGENCE_ENDPOINT and DOCUMENT_INTELLIGENCE_KEY)

# Retrieval parameters
RECALL_K = int(os.getenv("PP_RECALL_K", "200"))
CANDIDATE_K = int(os.getenv("PP_CANDIDATE_K", "50"))
FINALIST_K = int(os.getenv("PP_FINALIST_K", "5"))
SNIPPET_CHARS = int(os.getenv("PP_SNIPPET_CHARS", "150"))
MAX_IMAGES_PER_ANSWER = int(os.getenv("PP_MAX_IMAGES", "6"))

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # success-stories-retrival-system/
DATA_DIR = Path(os.getenv("PP_DATA_DIR", BASE_DIR / "data"))

PAPERS_DIR = DATA_DIR / "extracted_papers"
TREES_DIR = DATA_DIR / "trees"
INDEX_DIR = DATA_DIR / "index"
PROXIES_DIR = DATA_DIR / "proxies"

PROXY_FILE = PROXIES_DIR / "proxy_pointers.jsonl"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"

for _d in (PAPERS_DIR, TREES_DIR, INDEX_DIR, PROXIES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

def require_api_key() -> str:
    if not (OPENAI_API_KEY or AZURE_OPENAI_API_KEY):
        raise RuntimeError("No OpenAI or Azure API key found.")
    return OPENAI_API_KEY or AZURE_OPENAI_API_KEY
