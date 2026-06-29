import asyncio
import logging
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from app.main import app, lifespan
from app.config import settings
from app.database import get_db


class MockSession:
    def query(self, *args):
        return self

    def filter(self, *args):
        return self

    def group_by(self, *args):
        return self

    def all(self):
        return []


def override_get_db():
    yield MockSession()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_default():
    response = client.get("/forecast")
    assert response.status_code == 200
    data = response.json()
    assert data == {"historical": [], "forecast": []}


def test_forecast_custom_months():
    response = client.get("/forecast?months=6")
    assert response.status_code == 200
    data = response.json()
    assert "historical" in data
    assert "forecast" in data


def test_forecast_months_too_low():
    response = client.get("/forecast?months=0")
    assert response.status_code == 422


def test_forecast_months_too_high():
    response = client.get("/forecast?months=25")
    assert response.status_code == 422


def test_risk_scores_empty():
    response = client.get("/risk-scores")
    assert response.status_code == 200
    assert response.json() == []


def test_agent_insights_success():
    with patch("app.services.agent._call_ollama", return_value="Generated report"):
        response = client.get("/agent/insights")
    assert response.status_code == 200
    data = response.json()
    assert data == {
        "riskScores": [],
        "forecast": {"historical": [], "forecast": []},
        "report": "Generated report",
        "error": None,
    }


def test_agent_insights_ollama_unavailable():
    with patch("app.services.agent._call_ollama", side_effect=httpx.ConnectError("refused")):
        response = client.get("/agent/insights")
    assert response.status_code == 200
    data = response.json()
    assert data["report"] is None
    assert "Ollama service unavailable" in data["error"]


def test_agent_insights_months_too_low():
    response = client.get("/agent/insights?months=0")
    assert response.status_code == 422


def test_agent_insights_months_too_high():
    response = client.get("/agent/insights?months=25")
    assert response.status_code == 422


def test_risk_scores_accepts_org_id():
    response = client.get("/risk-scores?org_id=1")
    assert response.status_code == 200
    assert response.json() == []


def test_forecast_accepts_org_id():
    response = client.get("/forecast?org_id=1")
    assert response.status_code == 200


def test_protected_endpoint_rejects_request_without_key_when_configured():
    original = settings.INTERNAL_API_KEY
    settings.INTERNAL_API_KEY = "secret"
    try:
        response = client.get("/risk-scores")
        assert response.status_code == 401
    finally:
        settings.INTERNAL_API_KEY = original


def test_protected_endpoint_accepts_request_with_correct_key():
    original = settings.INTERNAL_API_KEY
    settings.INTERNAL_API_KEY = "secret"
    try:
        response = client.get("/risk-scores", headers={"X-Internal-Api-Key": "secret"})
        assert response.status_code == 200
    finally:
        settings.INTERNAL_API_KEY = original


def test_health_does_not_require_key():
    original = settings.INTERNAL_API_KEY
    settings.INTERNAL_API_KEY = "secret"
    try:
        response = client.get("/health")
        assert response.status_code == 200
    finally:
        settings.INTERNAL_API_KEY = original


async def _run_lifespan():
    async with lifespan(app):
        pass


def test_lifespan_warns_when_production_without_internal_api_key(caplog):
    original_env = settings.ENVIRONMENT
    original_key = settings.INTERNAL_API_KEY
    settings.ENVIRONMENT = "production"
    settings.INTERNAL_API_KEY = ""
    try:
        with caplog.at_level(logging.WARNING):
            asyncio.run(_run_lifespan())
        assert any("INTERNAL_API_KEY is unset" in r.message for r in caplog.records)
    finally:
        settings.ENVIRONMENT = original_env
        settings.INTERNAL_API_KEY = original_key


def test_lifespan_no_warning_when_internal_api_key_set(caplog):
    original_env = settings.ENVIRONMENT
    original_key = settings.INTERNAL_API_KEY
    settings.ENVIRONMENT = "production"
    settings.INTERNAL_API_KEY = "secret"
    try:
        with caplog.at_level(logging.WARNING):
            asyncio.run(_run_lifespan())
        assert not any("INTERNAL_API_KEY is unset" in r.message for r in caplog.records)
    finally:
        settings.ENVIRONMENT = original_env
        settings.INTERNAL_API_KEY = original_key


def test_anomalies_empty():
    with patch("app.services.anomaly_detection.compute_anomalies", return_value=[]):
        response = client.get("/anomalies")
    assert response.status_code == 200
    assert response.json() == []


def test_anomalies_returns_flagged_records():
    fake = [{
        "financialValueId": 1, "contractId": 2, "customerName": "Acme",
        "month": 6, "year": 2025, "financialAmount": 999_999.0,
        "anomalyScore": -0.5, "severity": "HIGH",
    }]
    with patch("app.services.anomaly_detection.compute_anomalies", return_value=fake):
        response = client.get("/anomalies")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["severity"] == "HIGH"


def test_anomalies_accepts_org_id():
    with patch("app.services.anomaly_detection.compute_anomalies", return_value=[]):
        response = client.get("/anomalies?org_id=3")
    assert response.status_code == 200


def test_lifespan_no_warning_in_development(caplog):
    original_env = settings.ENVIRONMENT
    original_key = settings.INTERNAL_API_KEY
    settings.ENVIRONMENT = "development"
    settings.INTERNAL_API_KEY = ""
    try:
        with caplog.at_level(logging.WARNING):
            asyncio.run(_run_lifespan())
        assert not any("INTERNAL_API_KEY is unset" in r.message for r in caplog.records)
    finally:
        settings.ENVIRONMENT = original_env
        settings.INTERNAL_API_KEY = original_key


def test_risk_scores_merges_ml_scores_when_present():
    fake_results = [{"contractId": 1, "customerName": "Acme", "riskScore": 0.5, "level": "MEDIUM", "anomalies": []}]
    fake_ml = {1: {"mlScore": 0.9, "mlLevel": "HIGH"}}
    with patch("app.routers.risk_scores.risk_scoring.compute_risk_scores", return_value=fake_results), \
         patch("app.routers.risk_scores.ml_risk_scoring.compute_ml_risk_scores", return_value=fake_ml):
        response = client.get("/risk-scores")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["mlScore"] == 0.9
    assert data[0]["mlLevel"] == "HIGH"


def test_risk_scores_skips_ml_merge_when_no_matching_contract():
    fake_results = [{"contractId": 99, "customerName": "Test", "riskScore": 0.3, "level": "LOW", "anomalies": []}]
    fake_ml = {1: {"mlScore": 0.9, "mlLevel": "HIGH"}}
    with patch("app.routers.risk_scores.risk_scoring.compute_risk_scores", return_value=fake_results), \
         patch("app.routers.risk_scores.ml_risk_scoring.compute_ml_risk_scores", return_value=fake_ml):
        response = client.get("/risk-scores")
    assert response.status_code == 200
    data = response.json()
    assert "mlScore" not in data[0]
