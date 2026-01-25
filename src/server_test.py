import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from src.server import app
from src.database import Base
from src.config import DEFAULT_FREQUENCY_SECONDS


# ##################################################################
# create test client
# helper to create a test client with isolated database
def create_test_client():
    import src.database as db_module

    original_engine = db_module._engine

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        db_module._engine = engine
        Base.metadata.create_all(engine)

        try:
            yield TestClient(app, raise_server_exceptions=True)
        finally:
            db_module._engine = original_engine


# ##################################################################
# test health endpoint
# verify health check returns ok status
def test_health():
    for client in create_test_client():
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ##################################################################
# test list feeds empty
# verify empty feed list is returned initially
def test_list_feeds_empty():
    for client in create_test_client():
        response = client.get("/feeds")
        assert response.status_code == 200
        assert response.json() == []


# ##################################################################
# test add feed
# verify feed can be added via API
def test_add_feed():
    for client in create_test_client():
        response = client.post("/feeds", json={"url": "https://example.com/feed.xml", "name": "Test Feed"})
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://example.com/feed.xml"
        assert data["name"] == "Test Feed"
        assert data["frequency_seconds"] == DEFAULT_FREQUENCY_SECONDS
        assert data["entry_count"] == 0


# ##################################################################
# test add feed duplicate
# verify duplicate feed returns error
def test_add_feed_duplicate():
    for client in create_test_client():
        client.post("/feeds", json={"url": "https://example.com/feed.xml"})
        response = client.post("/feeds", json={"url": "https://example.com/feed.xml"})
        assert response.status_code == 400


# ##################################################################
# test delete feed
# verify feed can be deleted
def test_delete_feed():
    for client in create_test_client():
        add_response = client.post("/feeds", json={"url": "https://example.com/feed.xml"})
        feed_id = add_response.json()["id"]

        delete_response = client.delete(f"/feeds/{feed_id}")
        assert delete_response.status_code == 200

        list_response = client.get("/feeds")
        assert list_response.json() == []


# ##################################################################
# test delete feed not found
# verify deleting non-existent feed returns 404
def test_delete_feed_not_found():
    for client in create_test_client():
        response = client.delete("/feeds/99999")
        assert response.status_code == 404


# ##################################################################
# test stats endpoint
# verify stats returns expected fields including top_feeds_by_avg_score
def test_stats():
    for client in create_test_client():
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_entries" in data
        assert "total_feeds" in data
        assert "top_feeds" in data
        assert "top_feeds_by_avg_score" in data
        assert isinstance(data["top_feeds_by_avg_score"], list)
