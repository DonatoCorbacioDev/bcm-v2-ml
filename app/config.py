from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DB_URL: str
    CORS_ORIGINS: str = "http://localhost:3000"

    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT: float = 120.0
    REPORT_LANGUAGE: str = "italian"

    # Shared secret expected on the X-Internal-Api-Key header. Empty disables
    # the check (local dev); must be set when the service is reachable
    # outside the backend's trusted network.
    INTERNAL_API_KEY: str = ""


settings = Settings()
