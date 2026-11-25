from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Platform database: AI cards (ai_instances, read-only) and monthly usage (ai_usage, written here).
    DATABASE_URL: str
    DATABASE_SSL: str = "require"  # asyncpg ssl mode; "disable" for local Postgres
    REDIS_URL: str = "redis://redis:6379/0"

    AI_DOMAIN: str = "ai.deploykar.tech"

    # Cloudflare Workers AI (open models). Never shown to customers.
    CF_AI_ACCOUNT_ID: str = ""
    CF_AI_TOKEN: str = ""

    TOKEN_CACHE_SECONDS: int = 60
    UNKNOWN_TOKEN_CACHE_SECONDS: int = 10
    USAGE_FLUSH_SECONDS: int = 30
    REQUESTS_PER_MINUTE: int = 120  # per AI card
    MAX_REQUEST_BYTES: int = 2_000_000
    MAX_OUTPUT_TOKENS: int = 8_000
    UPSTREAM_TIMEOUT_SECONDS: int = 120


def get_settings() -> Settings:
    return Settings()
