from fastapi.testclient import TestClient

from isthmus.api import app
from conftest import FakeEmbeddingProvider


def test_document_and_query_workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("ISTHMUS_EMBEDDING_MODEL", "test")
    monkeypatch.setattr("isthmus.api.OpenAIEmbeddingProvider", lambda **kwargs: FakeEmbeddingProvider())
    monkeypatch.setattr("isthmus.api.sector_selector_from_environment", lambda: None)
    monkeypatch.setenv("ISTHMUS_DATABASE", str(tmp_path / "test.db"))
    client = TestClient(app)
    assert client.post("/sectors", json={"name": "Finance"}).status_code == 201
    upload = client.post(
        "/documents",
        data={"sector": "Finance"},
        files={"file": ("policy.txt", b"Approved travel expenses are reimbursed.", "text/plain")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["document"]["id"]
    response = client.post("/query", json={"question": "travel reimbursement"})
    assert response.status_code == 200
    assert response.json()["retrieval"]["context"][0]["document_id"] == document_id
    assert "embedding" not in response.json()["retrieval"]["context"][0]
    assert client.delete(f"/documents/{document_id}").status_code == 200
    assert client.post("/query", json={"question": "travel"}).status_code == 400


def test_upload_requires_existing_sector(tmp_path, monkeypatch):
    monkeypatch.setenv("ISTHMUS_EMBEDDING_MODEL", "test")
    monkeypatch.setattr("isthmus.api.OpenAIEmbeddingProvider", lambda **kwargs: FakeEmbeddingProvider())
    monkeypatch.setenv("ISTHMUS_DATABASE", str(tmp_path / "test.db"))
    response = TestClient(app).post(
        "/documents",
        data={"sector": "missing"},
        files={"file": ("policy.txt", b"content", "text/plain")},
    )
    assert response.status_code == 400
