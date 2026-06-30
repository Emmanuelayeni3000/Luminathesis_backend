from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Keys
    groq_api_key: str = ""
    llama_cloud_api_key: str = ""
    semantic_scholar_api_key: str = ""
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # Database
    database_url: str = ""

    # CORS
    frontend_url: str = "https://luminathesis-ai.vercel.app"

    # Model identifiers
    extractor_model: str = "qwen-3.6-27b"
    architect_model: str = "gpt-oss-120b"
    editor_model: str = "qwen-3.6-27b"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Semantic Scholar
    scholar_target_count: int = 50
    scholar_batch_size: int = 100

    # Vector search
    vector_top_k: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()
