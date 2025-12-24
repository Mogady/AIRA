"""
Configuration management for A.I.R.A.

Configuration sources:
- config.yaml: Application settings (non-secrets)
- Environment variables: Secrets and infrastructure (API keys, DB hosts, passwords)

ENV vars override YAML values for infrastructure settings.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel


class AppSettings(BaseModel):
    """Application-level settings."""
    name: str = "A.I.R.A."
    version: str = "1.0.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"


class APISettings(BaseModel):
    """API server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = ["*"]


class DatabaseSettings(BaseModel):
    """PostgreSQL database settings."""
    host: str = "localhost"
    port: int = 5432
    name: str = "aira"
    user: str = "aira"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10

    @property
    def url(self) -> str:
        """Generate async database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        """Generate sync database URL (for migrations)."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseModel):
    """Redis settings."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""

    @property
    def url(self) -> str:
        """Generate Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class LLMSettings(BaseModel):
    """LLM provider settings."""
    provider: Literal["mock", "anthropic"] = "mock"
    model: str = "claude-sonnet-4-5-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120
    api_key: str = ""  # From ENV: ANTHROPIC_API_KEY


class EmbeddingsSettings(BaseModel):
    """Embeddings provider settings."""
    provider: Literal["mock", "openai"] = "mock"
    model: str = "text-embedding-ada-002"
    dimensions: int = 1536
    api_key: str = ""  # From ENV: OPENAI_API_KEY


class NewsSettings(BaseModel):
    """News API settings."""
    provider: Literal["mock", "newsapi"] = "mock"
    base_url: str = "https://newsapi.org/v2"
    articles_per_request: int = 5
    max_age_days: int = 30
    api_key: str = ""  # From ENV: NEWS_API_KEY


class FinancialSettings(BaseModel):
    """Financial data provider settings."""
    provider: Literal["mock", "yfinance"] = "mock"
    cache_ttl: int = 3600


class AgentSettings(BaseModel):
    """Agent behavior settings."""
    max_iterations: int = 10
    max_reflection_cycles: int = 2
    reflection_enabled: bool = True


class MonitoringSettings(BaseModel):
    """Scheduled monitoring settings."""
    enabled: bool = True
    check_interval_hours: int = 24
    min_new_articles: int = 5


class StorageSettings(BaseModel):
    """Storage provider settings."""
    provider: Literal["memory", "postgres"] = "postgres"  # Default to postgres for production


class Settings(BaseModel):
    """Main settings container."""
    app: AppSettings = AppSettings()
    api: APISettings = APISettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    news: NewsSettings = NewsSettings()
    financial: FinancialSettings = FinancialSettings()
    agent: AgentSettings = AgentSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    storage: StorageSettings = StorageSettings()


def _load_yaml_config(yaml_path: Path) -> dict:
    """Load YAML config file if it exists."""
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _apply_env_overrides(settings: Settings) -> Settings:
    """
    Apply environment variable overrides for infrastructure and secrets.

    ENV vars override YAML for:
    - Infrastructure: hosts, ports (for Docker flexibility)
    - Secrets: API keys, passwords
    """
    # Database (infrastructure + secrets)
    if os.environ.get("POSTGRES_HOST"):
        settings.database.host = os.environ["POSTGRES_HOST"]
    if os.environ.get("POSTGRES_PORT"):
        settings.database.port = int(os.environ["POSTGRES_PORT"])
    if os.environ.get("POSTGRES_DB"):
        settings.database.name = os.environ["POSTGRES_DB"]
    if os.environ.get("POSTGRES_USER"):
        settings.database.user = os.environ["POSTGRES_USER"]
    if os.environ.get("POSTGRES_PASSWORD"):
        settings.database.password = os.environ["POSTGRES_PASSWORD"]

    # Redis (infrastructure)
    if os.environ.get("REDIS_HOST"):
        settings.redis.host = os.environ["REDIS_HOST"]
    if os.environ.get("REDIS_PORT"):
        settings.redis.port = int(os.environ["REDIS_PORT"])
    if os.environ.get("REDIS_PASSWORD"):
        settings.redis.password = os.environ["REDIS_PASSWORD"]

    # API Keys (secrets)
    if os.environ.get("ANTHROPIC_API_KEY"):
        settings.llm.api_key = os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("OPENAI_API_KEY"):
        settings.embeddings.api_key = os.environ["OPENAI_API_KEY"]
    if os.environ.get("NEWS_API_KEY"):
        settings.news.api_key = os.environ["NEWS_API_KEY"]

    return settings


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings.

    Loads from:
    1. config.yaml (application settings)
    2. Environment variables (secrets + infrastructure overrides)
    """
    # Load .env file if present (for local development)
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)

    # Load YAML config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    yaml_config = _load_yaml_config(config_path)

    # Build settings from YAML
    settings = Settings(
        app=AppSettings(**yaml_config.get("app", {})),
        api=APISettings(**yaml_config.get("api", {})),
        database=DatabaseSettings(**yaml_config.get("database", {})),
        redis=RedisSettings(**yaml_config.get("redis", {})),
        llm=LLMSettings(**yaml_config.get("llm", {})),
        embeddings=EmbeddingsSettings(**yaml_config.get("embeddings", {})),
        news=NewsSettings(**yaml_config.get("news", {})),
        financial=FinancialSettings(**yaml_config.get("financial", {})),
        agent=AgentSettings(**yaml_config.get("agent", {})),
        monitoring=MonitoringSettings(**yaml_config.get("monitoring", {})),
        storage=StorageSettings(**yaml_config.get("storage", {})),
    )

    # Apply ENV var overrides for infrastructure and secrets
    settings = _apply_env_overrides(settings)

    return settings


def clear_settings_cache() -> None:
    """Clear the settings cache. Useful for testing."""
    get_settings.cache_clear()
