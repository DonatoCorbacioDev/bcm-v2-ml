# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import agent, anomalies, clause_risk, forecast, risk_scores

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.ENVIRONMENT == "production" and not settings.INTERNAL_API_KEY:
        raise RuntimeError(
            "ENVIRONMENT=production but INTERNAL_API_KEY is unset. "
            "Set INTERNAL_API_KEY to a strong random value to secure the ML API endpoints."
        )
    if settings.ENVIRONMENT == "production" and not settings.INTERNAL_CLAIMS_PUBLIC_KEY:
        raise RuntimeError(
            "ENVIRONMENT=production but INTERNAL_CLAIMS_PUBLIC_KEY is unset. "
            "Without it org_id/manager_id would be taken from the query string "
            "instead of the backend-signed claims."
        )
    yield


def docs_urls(environment: str) -> dict:
    """Interactive docs and the OpenAPI schema list every endpoint and its
    parameters: off in production (OWASP API8), where nobody browses them."""
    if environment == "production":
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {}


app = FastAPI(title="BCM ML Service", version="1.0.0", lifespan=lifespan, **docs_urls(settings.ENVIRONMENT))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router)
app.include_router(risk_scores.router)
app.include_router(anomalies.router)
app.include_router(agent.router)
app.include_router(clause_risk.router)


@app.get("/health")
def health():
    return {"status": "ok"}
