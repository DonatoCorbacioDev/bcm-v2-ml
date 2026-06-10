from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DB_URL: str
    CORS_ORIGINS: str = "http://localhost:3000"

    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT: float = 60.0
    REPORT_LANGUAGE: str = "italian"


settings = Settings()
