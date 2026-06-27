"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

# Load .env into os.environ so LangChain/LangSmith SDK can read tracing config
load_dotenv()

class Settings(BaseSettings):
    
    # LLM Configuration
    google_api_key: str
    primary_model: str = "gemini-3.1-flash-lite"
    fallback_model: str = "gemma4:12b"
    
    # LangSmith
    langchain_tracing_v2: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "production-api"
    
    
    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3
    
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - loaded once, reused everywhere."""
    return Settings()