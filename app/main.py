import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import agent, anomalies, forecast, risk_scores

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.ENVIRONMENT == "production" and not settings.INTERNAL_API_KEY:
        raise RuntimeError(
            "ENVIRONMENT=production but INTERNAL_API_KEY is unset. "
            "Set INTERNAL_API_KEY to a strong random value to secure the ML API endpoints."
        )
    yield


app = FastAPI(title="BCM ML Service", version="1.0.0", lifespan=lifespan)

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


@app.get("/health")
def health():
    return {"status": "ok"}
