from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db


class MockSession:
    def query(self, *args):
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
