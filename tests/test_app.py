
from fastapi.testclient import TestClient

from src.app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_bootstrap():
    response = client.get("/bootstrap")
    assert response.status_code == 200
    payload = response.json()
    assert "project" in payload
    assert "sample_features" in payload
    assert "app_modules" in payload


def test_project_metadata():
    response = client.get("/project")
    assert response.status_code == 200
    assert "slug" in response.json()


def test_recommend():
    response = client.post("/recommend", json={"prompt": "Recommend next steps."})
    assert response.status_code == 200
    assert "actions" in response.json()
