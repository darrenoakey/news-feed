import tempfile
from pathlib import Path
import threading
import time

from sqlalchemy import create_engine
import uvicorn

from src.cli import feed_add, feed_list, feed_delete, get_base_url, apply_training_scores
from src.server import app
from src.database import Base
from src.config import SERVER_HOST


# ##################################################################
# test get base url
# verify base URL is constructed correctly
def test_get_base_url():
    url = get_base_url()
    assert url.startswith("http://")
    assert SERVER_HOST in url


# ##################################################################
# test feed operations with real server
# integration test for add, list, delete operations
def test_feed_operations_integration():
    import src.database as db_module
    import src.config as config_module

    original_engine = db_module._engine
    original_port = config_module.SERVER_PORT

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        db_module._engine = engine
        Base.metadata.create_all(engine)

        test_port = 19199
        config_module.SERVER_PORT = test_port

        config = uvicorn.Config(app, host=SERVER_HOST, port=test_port, log_level="error")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        time.sleep(1)

        try:
            result = feed_list()
            assert result == 0

            result = feed_add("https://example.com/test.xml", "Test Feed")
            assert result == 0

            result = feed_list()
            assert result == 0

            result = feed_delete(1)
            assert result == 0

        finally:
            server.should_exit = True
            thread.join(timeout=2)
            db_module._engine = original_engine
            config_module.SERVER_PORT = original_port


# ##################################################################
# test export rss ordering
# verify items are ordered by discovered_at (creation time), not scored_at
def test_export_rss_orders_by_discovered_at():
    import io
    import sys
    import xml.etree.ElementTree as ET
    from datetime import datetime, timezone, timedelta

    import src.database as db_module
    from src.database import Base, get_session
    from src.models import Feed, FeedEntry
    from src.cli import export_rss

    original_engine = db_module._engine

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        db_module._engine = engine
        Base.metadata.create_all(engine)

        try:
            # Create test data: items discovered in specific order
            now = datetime.now(timezone.utc)
            with get_session() as session:
                feed = Feed(url="https://test.com/feed.xml", name="Test Feed")
                session.add(feed)
                session.flush()

                # Entry 1: discovered first, scored last
                entry1 = FeedEntry(
                    feed_id=feed.id,
                    guid="entry1",
                    xml_content="<entry><title>First Discovered</title><link>https://test.com/1</link><summary>First</summary><published>Mon, 20 Jan 2026 00:00:00 +0000</published></entry>",
                    discovered_at=now - timedelta(hours=2),
                    score=9.5,
                    scored_at=now,  # Scored last
                )
                session.add(entry1)

                # Entry 2: discovered second, scored first
                entry2 = FeedEntry(
                    feed_id=feed.id,
                    guid="entry2",
                    xml_content="<entry><title>Second Discovered</title><link>https://test.com/2</link><summary>Second</summary><published>Mon, 20 Jan 2026 00:00:00 +0000</published></entry>",
                    discovered_at=now - timedelta(hours=1),
                    score=9.5,
                    scored_at=now - timedelta(hours=1),  # Scored first
                )
                session.add(entry2)

                # Entry 3: discovered last (most recent)
                entry3 = FeedEntry(
                    feed_id=feed.id,
                    guid="entry3",
                    xml_content="<entry><title>Third Discovered</title><link>https://test.com/3</link><summary>Third</summary><published>Mon, 20 Jan 2026 00:00:00 +0000</published></entry>",
                    discovered_at=now,  # Most recently discovered
                    score=9.5,
                    scored_at=now - timedelta(minutes=30),
                )
                session.add(entry3)

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                export_rss(
                    min_score=8.0,
                    limit=10,
                    title="Test",
                    description="Test feed",
                    link="https://test.com",
                )
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            # Parse RSS output
            root = ET.fromstring(output)
            items = root.findall(".//item")

            assert len(items) == 3

            # Items should be ordered by discovered_at desc (most recent first)
            titles = [item.find("title").text for item in items]
            assert titles == ["Third Discovered", "Second Discovered", "First Discovered"]

        finally:
            db_module._engine = original_engine


# ##################################################################
# test apply training scores
# verify apply_training_scores updates scores for matching entries
def test_apply_training_scores():
    from datetime import datetime, timezone

    import src.database as db_module
    from src.database import Base, get_session
    from src.models import Feed, FeedEntry

    original_engine = db_module._engine

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        db_module._engine = engine
        Base.metadata.create_all(engine)

        try:
            now = datetime.now(timezone.utc)
            with get_session() as session:
                feed = Feed(url="https://test.com/feed.xml", name="Test Feed")
                session.add(feed)
                session.flush()

                # Entry with URL that will be in training set
                entry1 = FeedEntry(
                    feed_id=feed.id,
                    guid="entry1",
                    xml_content="<entry><title>Test Article</title><link>https://example.com/trained</link></entry>",
                    discovered_at=now,
                    score=5.0,
                )
                session.add(entry1)

                # Entry with URL not in training set
                entry2 = FeedEntry(
                    feed_id=feed.id,
                    guid="entry2",
                    xml_content="<entry><title>Other Article</title><link>https://example.com/other</link></entry>",
                    discovered_at=now,
                    score=5.0,
                )
                session.add(entry2)

            # Apply training data directly (no API call needed)
            training_data = [{"url": "https://example.com/trained", "score": 9.5}]
            updated = apply_training_scores(training_data)
            assert updated == 1

            # Verify the score was updated
            with get_session() as session:
                entry1 = session.query(FeedEntry).filter(FeedEntry.guid == "entry1").first()
                entry2 = session.query(FeedEntry).filter(FeedEntry.guid == "entry2").first()

                assert entry1.score == 9.5  # Updated from training data
                assert entry2.score == 5.0  # Unchanged

        finally:
            db_module._engine = original_engine
