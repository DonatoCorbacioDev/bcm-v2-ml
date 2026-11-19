# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# Docker secrets (OWASP Secrets Management Cheat Sheet): in production each
# secret is a file here named after its setting (DB_URL, INTERNAL_API_KEY),
# not an environment variable that `docker inspect` would show. An
# environment variable of the same name still wins: production removes them.
SECRETS_DIR = os.environ.get("SECRETS_DIR", "/run/secrets")


def _secrets_dir() -> str | None:
    # Only when it exists: a laptop or the tests have none, and pydantic
    # would warn about a missing directory on every start.
    return SECRETS_DIR if os.path.isdir(SECRETS_DIR) else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", secrets_dir=_secrets_dir()
    )

    DB_URL: str
    # The database CA's certificate (PEM file): when set, the connection to
    # MySQL is TLS with the server's certificate checked against it (OWASP
    # Database Security, TLS). Required in production; empty on a laptop.
    DB_SSL_CA: str = ""
    CORS_ORIGINS: str = "http://localhost:3000"

    # "development" or "production". Used only to decide whether to warn at
    # startup about an unset INTERNAL_API_KEY; does not gate any behavior.
    ENVIRONMENT: str = "development"

    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT: float = 120.0
    REPORT_LANGUAGE: str = "italian"

    # Shared secret expected on the X-Internal-Api-Key header. Empty disables
    # the check (local dev); must be set when the service is reachable
    # outside the backend's trusted network.
    INTERNAL_API_KEY: str = ""

    # RSA public key (X.509, base64), used to verify the X-Internal-Claims JWT
    # the backend signs asserting org_id/manager_id for each request. Empty
    # disables verification (local dev) — org_id/manager_id are then trusted
    # directly from the query string, same posture as INTERNAL_API_KEY empty.
    # The matching private key lives only on the backend (ml.claims.private-key),
    # never here — see bcm-v2-backend's MlClaimsSigner for why this is
    # asymmetric rather than a second shared secret.
    INTERNAL_CLAIMS_PUBLIC_KEY: str = ""


settings = Settings()
