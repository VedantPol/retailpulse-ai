from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "RetailPulse AI API"
    api_version: str = "1.0.0"
    artifacts_dir: Path = Path("backend/artifacts")
    data_seed: int = 42
    forecast_horizon_default: int = 30
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def data_dir(self) -> Path:
        return self.artifacts_dir / "data"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()

