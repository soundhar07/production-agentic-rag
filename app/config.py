"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.
"""

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

# Load .env into os.environ so LangChain/LangSmith SDK can read tracing config
load_dotenv()

# The LangSmith SDK reads the LANGSMITH_* env vars; this repo's .env uses the older
# LANGCHAIN_* names. Bridge them so traces emit regardless of which the SDK expects.
for _new, _old in (
    ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
    ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
    ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
):
    if not os.getenv(_new) and os.getenv(_old):
        os.environ[_new] = os.environ[_old]

class Settings(BaseSettings):
    
    # LLM Configuration
    google_api_key: str
    primary_model: str = "gemini-3.1-flash-lite"
    fallback_model: str = "gemma4:12b"

    # === RAG / Retrieval ===
    # Embedding model — served locally by Ollama via OllamaEmbeddings (2560 dims,
    # no API key, no quota). The corpus is static, so it is embedded exactly once.
    embedding_model: str = "qwen3-embedding:4b"
    pdf_path: str = "docs/constitution.pdf"
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "constitution"
    chunks_path: str = "./chroma_db/chunks.jsonl"
    # Page index (0-based) where the article body begins — pages before this are
    # the cover, preface and table of contents (whose one-line titles mimic
    # article headings and would pollute retrieval). The Preamble lives here.
    body_start_page: int = 31
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    # One rewrite attempt before giving up (grade is binary; no score threshold).
    max_rag_retries: int = 1

    # LangSmith
    langchain_tracing_v2: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "production-api"
    
    
    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 3000
    max_retries: int = 3
    
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - loaded once, reused everywhere."""
    return Settings()