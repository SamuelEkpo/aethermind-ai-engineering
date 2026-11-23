"""
Centralized application settings.

Uses pydantic-settings so every config value is typed, validated at startup,
and overridable via environment variables or a .env file. Never read
os.environ directly elsewhere in the codebase — import `settings` from here.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "AetherMind"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- Security ---
    secret_key: str = "change-me-before-any-real-deployment"
    access_token_expire_minutes: int = 60

    # --- Database ---
    database_url: str = "postgresql+asyncpg://aethermind:change-me@localhost:5432/aethermind"

    # --- Cache ---
    redis_url: str = "redis://localhost:6379/0"

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"

    # --- Model ---
    model_checkpoint_dir: str = "training/checkpoints"
    tokenizer_vocab_size: int = 32000
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — safe to call repeatedly (e.g. as a FastAPI dependency)."""
    return Settings()


settings = get_settings()
