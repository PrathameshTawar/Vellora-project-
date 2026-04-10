"""Environment variable loading for SwarmIQ v2."""

import os

from pydantic_settings import BaseSettings
from typing import Optional

RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "25"))
RATE_LIMIT_TPM: int = int(os.getenv("RATE_LIMIT_TPM", "10000"))
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_RESEARCH_ITERATIONS: int = int(os.getenv("MAX_RESEARCH_ITERATIONS", "3"))
MAX_CRITIQUE_REVISIONS: int = int(os.getenv("MAX_CRITIQUE_REVISIONS", "2"))
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

class Settings(BaseSettings):
    """SwarmIQ v2 configuration."""

    # OpenAI
    openai_api_key: Optional[str] = None

    # Pinecone
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "swarmiq-v2"

    # Search APIs
    tavily_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None

    # LLM
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2

    # Logging
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


config = Settings()
