from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.models import Feed, FeedEntry, PendingEntry
from src.config import DEFAULT_FREQUENCY_SECONDS


# ##################################################################
# create test session
# helper to create a fresh in-memory database session
def create_test_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return session_factory()


# ##################################################################
# test feed creation
# verify feed can be created with required fields
def test_feed_creation():
    session = create_test_session()
    feed = Feed(url="https://example.com/feed.xml", name="Test Feed")
    session.add(feed)
    session.commit()

    assert feed.id is not None
    assert feed.url == "https://example.com/feed.xml"
    assert feed.name == "Test Feed"
    assert feed.last_checked is None
    assert feed.frequency_seconds == DEFAULT_FREQUENCY_SECONDS
    assert feed.created_at is not None
    session.close()


# ##################################################################
# test feed entry creation
# verify feed entry can be created and linked to feed
def test_feed_entry_creation():
    session = create_test_session()
    feed = Feed(url="https://example.com/feed.xml", name="Test Feed")
    session.add(feed)
    session.commit()

    entry = FeedEntry(feed_id=feed.id, guid="unique-id-123", xml_content="<item><title>Test</title></item>")
    session.add(entry)
    session.commit()

    assert entry.id is not None
    assert entry.feed_id == feed.id
    assert entry.guid == "unique-id-123"
    assert entry.discovered_at is not None
    session.close()


# ##################################################################
# test pending entry creation
# verify pending entry can be created and linked to feed entry
def test_pending_entry_creation():
    session = create_test_session()
    feed = Feed(url="https://example.com/feed.xml", name="Test Feed")
    session.add(feed)
    session.commit()

    entry = FeedEntry(feed_id=feed.id, guid="unique-id-123", xml_content="<item><title>Test</title></item>")
    session.add(entry)
    session.commit()

    pending = PendingEntry(entry_id=entry.id)
    session.add(pending)
    session.commit()

    assert pending.id is not None
    assert pending.entry_id == entry.id
    assert pending.created_at is not None
    session.close()


# ##################################################################
# test feed cascade delete
# verify deleting feed cascades to entries and pending
def test_feed_cascade_delete():
    session = create_test_session()
    feed = Feed(url="https://example.com/feed.xml", name="Test Feed")
    session.add(feed)
    session.commit()

    entry = FeedEntry(feed_id=feed.id, guid="unique-id-123", xml_content="<item><title>Test</title></item>")
    session.add(entry)
    session.commit()

    pending = PendingEntry(entry_id=entry.id)
    session.add(pending)
    session.commit()

    session.delete(feed)
    session.commit()

    assert session.query(Feed).count() == 0
    assert session.query(FeedEntry).count() == 0
    assert session.query(PendingEntry).count() == 0
    session.close()
