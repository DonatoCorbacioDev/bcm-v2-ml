from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db


def override_get_db():
    yield None


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
