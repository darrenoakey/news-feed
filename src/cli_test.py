import tempfile
from pathlib import Path
import threading
import time

from sqlalchemy import create_engine
import uvicorn

from src.cli import feed_add, feed_list, feed_delete, get_base_url
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
# test export rss by label
# verify label-based export produces valid RSS XML
def test_export_rss_by_label():
    import io
    import sys
    import xml.etree.ElementTree as ET
    from datetime import datetime, timezone

    import src.database as db_module
    from src.database import Base, get_session
    from src.models import Feed, FeedEntry, TitleClassification
    from src.cli import export_rss_by_label

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

                entry1 = FeedEntry(
                    feed_id=feed.id,
                    guid="entry1",
                    xml_content="<entry><title>Great Article</title><link>https://test.com/1</link><summary>Summary</summary></entry>",
                    discovered_at=now,
                )
                session.add(entry1)
                session.flush()

                tc1 = TitleClassification(
                    entry_id=entry1.id,
                    title="Great Article",
                    predicted_label="great",
                    predicted_score=0.95,
                )
                session.add(tc1)

                entry2 = FeedEntry(
                    feed_id=feed.id,
                    guid="entry2",
                    xml_content="<entry><title>Other Article</title><link>https://test.com/2</link><summary>Summary</summary></entry>",
                    discovered_at=now,
                )
                session.add(entry2)
                session.flush()

                tc2 = TitleClassification(
                    entry_id=entry2.id,
                    title="Other Article",
                    predicted_label="other",
                    predicted_score=0.80,
                )
                session.add(tc2)

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                export_rss_by_label(
                    label="great",
                    limit=10,
                    title="Test",
                    description="Test feed",
                    link="https://test.com",
                )
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            root = ET.fromstring(output)
            items = root.findall(".//item")

            # Only the "great" entry should appear
            assert len(items) == 1
            assert items[0].find("title").text == "Great Article"

        finally:
            db_module._engine = original_engine
