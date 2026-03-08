from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, UniqueConstraint, Float
from sqlalchemy.orm import relationship

from src.database import Base
from src.config import DEFAULT_FREQUENCY_SECONDS


# ##################################################################
# utc now
# return current UTC time as timezone-aware datetime
def utc_now():
    return datetime.now(timezone.utc)


# ##################################################################
# feed model
# represents an RSS feed to be polled
class Feed(Base):
    __tablename__ = "feeds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    last_checked = Column(DateTime, nullable=True)
    frequency_seconds = Column(Integer, nullable=False, default=DEFAULT_FREQUENCY_SECONDS)
    created_at = Column(DateTime, nullable=False, default=utc_now)

    entries = relationship("FeedEntry", back_populates="feed", cascade="all, delete-orphan")


# ##################################################################
# feed entry model
# represents a single entry from an RSS feed
class FeedEntry(Base):
    __tablename__ = "feed_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False)
    guid = Column(String, nullable=False)
    xml_content = Column(Text, nullable=False)
    discovered_at = Column(DateTime, nullable=False, default=utc_now)
    score = Column(Float, nullable=True)
    scored_at = Column(DateTime, nullable=True)

    feed = relationship("Feed", back_populates="entries")
    title_classification = relationship("TitleClassification", back_populates="entry", cascade="all, delete-orphan", uselist=False)

    __table_args__ = (UniqueConstraint("feed_id", "guid", name="uq_feed_guid"),)


# ##################################################################
# title classification model
# stores human labels and predictions for title-based classification
class TitleClassification(Base):
    __tablename__ = "title_classifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(Integer, ForeignKey("feed_entries.id"), unique=True, nullable=False)
    title = Column(Text, nullable=False)
    human_label = Column(Text, nullable=True)
    predicted_label = Column(Text, nullable=True)
    predicted_score = Column(Float, nullable=True)
    classified_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utc_now)

    entry = relationship("FeedEntry", back_populates="title_classification")
