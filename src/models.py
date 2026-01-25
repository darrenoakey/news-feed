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
    pending = relationship("PendingEntry", back_populates="entry", cascade="all, delete-orphan")
    scored = relationship("ScoredEntry", back_populates="entry", cascade="all, delete-orphan")
    error = relationship("ErrorEntry", back_populates="entry", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("feed_id", "guid", name="uq_feed_guid"),)


# ##################################################################
# pending entry model
# queue of new entries waiting to be processed
class PendingEntry(Base):
    __tablename__ = "pending_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(Integer, ForeignKey("feed_entries.id"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=utc_now)

    entry = relationship("FeedEntry", back_populates="pending")


# ##################################################################
# scored entry model
# queue of successfully scored entries waiting for further processing
class ScoredEntry(Base):
    __tablename__ = "scored_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(Integer, ForeignKey("feed_entries.id"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=utc_now)

    entry = relationship("FeedEntry", back_populates="scored")


# ##################################################################
# error entry model
# queue of entries that failed scoring
class ErrorEntry(Base):
    __tablename__ = "error_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(Integer, ForeignKey("feed_entries.id"), nullable=False)
    error_message = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utc_now)

    entry = relationship("FeedEntry", back_populates="error")
