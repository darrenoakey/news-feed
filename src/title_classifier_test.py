from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.models import Feed, FeedEntry, TitleClassification
from src.title_classifier import TitleClassifierService


# ##################################################################
# create test session
# helper to create a fresh in-memory database session
def create_test_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return session_factory()


# ##################################################################
# test title classification model creation
def test_title_classification_creation():
    session = create_test_session()
    feed = Feed(url="https://example.com/feed.xml", name="Test Feed")
    session.add(feed)
    session.commit()

    entry = FeedEntry(feed_id=feed.id, guid="guid-1", xml_content="<item><title>Test Title</title></item>")
    session.add(entry)
    session.commit()

    tc = TitleClassification(entry_id=entry.id, title="Test Title")
    session.add(tc)
    session.commit()

    assert tc.id is not None
    assert tc.entry_id == entry.id
    assert tc.title == "Test Title"
    assert tc.human_label is None
    assert tc.predicted_label is None
    assert tc.created_at is not None
    session.close()


# ##################################################################
# test title classification unique constraint
def test_title_classification_unique_entry():
    session = create_test_session()
    feed = Feed(url="https://example.com/feed.xml", name="Test Feed")
    session.add(feed)
    session.commit()

    entry = FeedEntry(feed_id=feed.id, guid="guid-1", xml_content="<item><title>Test</title></item>")
    session.add(entry)
    session.commit()

    tc1 = TitleClassification(entry_id=entry.id, title="Test")
    session.add(tc1)
    session.commit()

    tc2 = TitleClassification(entry_id=entry.id, title="Test")
    session.add(tc2)
    try:
        session.commit()
        assert False, "Should have raised IntegrityError"
    except Exception:
        session.rollback()
    session.close()


# ##################################################################
# test cascade delete from feed entry
def test_title_classification_cascade_delete():
    session = create_test_session()
    feed = Feed(url="https://example.com/feed.xml", name="Test Feed")
    session.add(feed)
    session.commit()

    entry = FeedEntry(feed_id=feed.id, guid="guid-1", xml_content="<item><title>Test</title></item>")
    session.add(entry)
    session.commit()

    tc = TitleClassification(entry_id=entry.id, title="Test", human_label="great")
    session.add(tc)
    session.commit()

    session.delete(feed)
    session.commit()

    assert session.query(TitleClassification).count() == 0
    session.close()


# ##################################################################
# test predict returns None without rules loaded
def test_predict_without_rules():
    svc = TitleClassifierService()
    result = svc.predict("some title")
    assert result is None
    assert not svc.is_trained


# ##################################################################
# test rolling accuracy tracking
def test_rolling_accuracy():
    svc = TitleClassifierService()

    # record some correct and incorrect predictions
    svc.record_result("great", "great")
    svc.record_result("good", "good")
    svc.record_result("other", "other")
    svc.record_result("other", "good")  # wrong

    metrics = svc.get_metrics()
    assert metrics is not None
    rolling = metrics["rolling"]
    assert rolling["total"] == 4
    assert rolling["accuracy"] == 0.75  # 3/4 correct


# ##################################################################
# test set training count
def test_set_training_count():
    svc = TitleClassifierService()
    svc.set_training_count(42)
    assert svc.training_count == 42
    metrics = svc.get_metrics()
    assert metrics is not None
    assert metrics["training_samples"] == 42
