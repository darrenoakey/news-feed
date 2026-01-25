#!/usr/bin/env python3
import sys
from pathlib import Path

# add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

import setproctitle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import (
    SERVER_HOST,
    SERVER_PORT,
    MIN_FREQUENCY_SECONDS,
    MAX_FREQUENCY_SECONDS,
    DEFAULT_FREQUENCY_SECONDS,
    FREQUENCY_ADJUSTMENT_SECONDS,
    WORKER_SLEEP_SECONDS,
    SCORING_WORKER_SLEEP_SECONDS,
    SCORING_API_TIMEOUT_SECONDS,
    DISCORD_MIN_SCORE,
    DISCORD_WORKER_SLEEP_SECONDS,
    DISCORD_RATE_LIMIT_BACKOFF_SECONDS,
)
from src.database import get_session, init_db
from src.models import Feed, FeedEntry, PendingEntry, ScoredEntry, ErrorEntry
from src.rss import fetch_rss_entries
from src.scoring import extract_link_from_xml, get_score_for_url, ScoringError
from src.discord_sender import extract_title_from_xml, extract_summary_from_xml, send_news_item

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeedCreate(BaseModel):
    url: str
    name: Optional[str] = None


class FeedResponse(BaseModel):
    id: int
    url: str
    name: str
    last_checked: Optional[datetime]
    frequency_seconds: int
    created_at: datetime
    entry_count: int

    class Config:
        from_attributes = True


# ##################################################################
# background worker
# continuously poll feeds and discover new entries
async def background_worker():
    logger.info("Background worker started")
    while True:
        try:
            processed = await process_next_feed()
            if not processed:
                await asyncio.sleep(WORKER_SLEEP_SECONDS)
        except Exception as err:
            logger.exception("Background worker error: %s", err)
            await asyncio.sleep(WORKER_SLEEP_SECONDS)


# ##################################################################
# process next feed
# find and process the next feed that needs checking, returns True if processed
async def process_next_feed() -> bool:
    with get_session() as session:
        now = datetime.now(timezone.utc)

        # get feed that was checked longest ago (or never)
        feed = session.query(Feed).order_by(Feed.last_checked.asc().nullsfirst()).first()

        if feed is None:
            return False

        # check if feed is due for checking
        if feed.last_checked is not None:
            last_checked = feed.last_checked.replace(tzinfo=timezone.utc)
            next_check = last_checked + timedelta(seconds=feed.frequency_seconds)
            if next_check > now:
                return False

        logger.info("Checking feed: %s (%s)", feed.name, feed.url)

        try:
            entries = fetch_rss_entries(feed.url)
        except Exception as err:
            logger.error("Failed to fetch feed %s: %s", feed.url, err)
            feed.last_checked = now
            return True

        new_count = 0
        for entry in entries:
            existing = (
                session.query(FeedEntry).filter(FeedEntry.feed_id == feed.id, FeedEntry.guid == entry.guid).first()
            )

            if existing is None:
                feed_entry = FeedEntry(feed_id=feed.id, guid=entry.guid, xml_content=entry.xml_content)
                session.add(feed_entry)
                session.flush()

                pending = PendingEntry(entry_id=feed_entry.id)
                session.add(pending)
                new_count += 1
                logger.info("NEW: [%s] %s", feed.name, entry.title or entry.guid)

        feed.last_checked = now

        if new_count > 0:
            feed.frequency_seconds = max(MIN_FREQUENCY_SECONDS, feed.frequency_seconds - FREQUENCY_ADJUSTMENT_SECONDS)
            logger.info("Found %d new entries, frequency now %ds", new_count, feed.frequency_seconds)
        else:
            feed.frequency_seconds = min(MAX_FREQUENCY_SECONDS, feed.frequency_seconds + FREQUENCY_ADJUSTMENT_SECONDS)
            logger.info("No new entries, frequency now %ds", feed.frequency_seconds)

        return True


# ##################################################################
# scoring worker
# continuously process pending entries and score them
async def scoring_worker():
    logger.info("Scoring worker started")
    while True:
        try:
            processed = await process_next_pending()
            if not processed:
                await asyncio.sleep(SCORING_WORKER_SLEEP_SECONDS)
        except Exception as err:
            logger.exception("Scoring worker error: %s", err)
            await asyncio.sleep(SCORING_WORKER_SLEEP_SECONDS)


# ##################################################################
# process next pending
# find and score the next pending entry, returns True if something was processed
async def process_next_pending() -> bool:
    with get_session() as session:
        pending = session.query(PendingEntry).first()
        if pending is None:
            return False

        entry = pending.entry
        feed = entry.feed

        link = extract_link_from_xml(entry.xml_content)
        if link is None:
            link = entry.guid

        logger.info("SCORING: [%s] %s", feed.name, link)

        # save entry_id and feed_name before closing session
        entry_id = entry.id
        feed_name = feed.name

        session.delete(pending)

    # run scoring API call in thread pool to avoid blocking event loop
    try:
        result = await asyncio.to_thread(get_score_for_url, link, SCORING_API_TIMEOUT_SECONDS)
        score = result.get("rank", 0)
    except ScoringError as err:
        with get_session() as session:
            error = ErrorEntry(entry_id=entry_id, error_message=str(err))
            session.add(error)
        logger.error("SCORE ERROR: [%s] %s - %s", feed_name, link, err)
        return True

    with get_session() as session:
        now = datetime.now(timezone.utc)
        entry = session.get(FeedEntry, entry_id)

        if score == 0:
            error = ErrorEntry(entry_id=entry_id, error_message="Score returned 0")
            session.add(error)
            logger.warning("SCORE=0: [%s] %s", feed_name, link)
        else:
            entry.score = score
            entry.scored_at = now
            scored = ScoredEntry(entry_id=entry_id)
            session.add(scored)
            logger.info("SCORED: [%s] %s -> %.1f", feed_name, link, score)

    return True


# ##################################################################
# discord worker
# continuously publish high-scored entries to discord
async def discord_worker():
    logger.info("Discord worker started")
    backoff_until = None

    while True:
        try:
            # check if we're in backoff mode
            if backoff_until is not None:
                now = datetime.now(timezone.utc)
                if now < backoff_until:
                    wait_seconds = (backoff_until - now).total_seconds()
                    logger.info("Discord rate limited, waiting %.0fs", wait_seconds)
                    await asyncio.sleep(min(wait_seconds, 60))
                    continue
                backoff_until = None

            result = await process_next_scored()
            if result == "rate_limited":
                backoff_until = datetime.now(timezone.utc) + timedelta(seconds=DISCORD_RATE_LIMIT_BACKOFF_SECONDS)
                logger.warning("Discord rate limited, backing off for %ds", DISCORD_RATE_LIMIT_BACKOFF_SECONDS)
            elif not result:
                await asyncio.sleep(DISCORD_WORKER_SLEEP_SECONDS)
        except Exception as err:
            logger.exception("Discord worker error: %s", err)
            await asyncio.sleep(DISCORD_WORKER_SLEEP_SECONDS)


# ##################################################################
# process next scored
# find and publish the next scored entry, returns True/False/"rate_limited"
async def process_next_scored():
    with get_session() as session:
        scored = session.query(ScoredEntry).first()
        if scored is None:
            return False

        entry = scored.entry
        feed = entry.feed

        # save data before closing session
        entry_id = entry.id
        score = entry.score or 0
        xml_content = entry.xml_content
        feed_name = feed.name if feed else "Unknown"

        # remove from scored queue regardless of outcome
        session.delete(scored)

    # check score threshold
    if score < DISCORD_MIN_SCORE:
        logger.info("SKIP: [%s] score %.1f < %.1f threshold", feed_name, score, DISCORD_MIN_SCORE)
        return True

    # extract content for discord message
    title = extract_title_from_xml(xml_content) or f"Entry {entry_id}"
    link = extract_link_from_xml(xml_content) or ""
    summary = extract_summary_from_xml(xml_content)

    logger.info("DISCORD: [%s] %s (%.1f)", feed_name, title, score)

    # send to discord in thread pool to avoid blocking
    try:
        success = await asyncio.to_thread(send_news_item, title, link, score, feed_name, summary)
        if not success:
            logger.error("DISCORD FAILED: [%s] %s", feed_name, title)
        return True
    except Exception as err:
        err_str = str(err).lower()
        if "rate limit" in err_str or "too many" in err_str:
            return "rate_limited"
        logger.error("DISCORD ERROR: [%s] %s - %s", feed_name, title, err)
        return True


# ##################################################################
# lifespan
# startup and shutdown logic for the FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized")

    feed_task = asyncio.create_task(background_worker())
    scoring_task = asyncio.create_task(scoring_worker())
    discord_task = asyncio.create_task(discord_worker())
    yield

    feed_task.cancel()
    scoring_task.cancel()
    discord_task.cancel()
    try:
        await feed_task
    except asyncio.CancelledError:
        pass
    try:
        await scoring_task
    except asyncio.CancelledError:
        pass
    try:
        await discord_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="News Feed", lifespan=lifespan)


# ##################################################################
# health check
# simple endpoint to verify server is running
@app.get("/health")
def health():
    return {"status": "ok"}


# ##################################################################
# list feeds
# return all configured feeds with entry counts
@app.get("/feeds", response_model=list[FeedResponse])
def list_feeds():
    with get_session() as session:
        feeds = session.query(Feed).all()
        result = []
        for feed in feeds:
            entry_count = session.query(FeedEntry).filter(FeedEntry.feed_id == feed.id).count()
            result.append(
                FeedResponse(
                    id=feed.id,
                    url=feed.url,
                    name=feed.name,
                    last_checked=feed.last_checked,
                    frequency_seconds=feed.frequency_seconds,
                    created_at=feed.created_at,
                    entry_count=entry_count,
                )
            )
        return result


# ##################################################################
# add feed
# create a new feed to be polled
@app.post("/feeds", response_model=FeedResponse)
def add_feed(feed_create: FeedCreate):
    with get_session() as session:
        existing = session.query(Feed).filter(Feed.url == feed_create.url).first()
        if existing:
            raise HTTPException(status_code=400, detail="Feed already exists")

        name = feed_create.name or feed_create.url
        feed = Feed(url=feed_create.url, name=name, frequency_seconds=DEFAULT_FREQUENCY_SECONDS)
        session.add(feed)
        session.flush()

        return FeedResponse(
            id=feed.id,
            url=feed.url,
            name=feed.name,
            last_checked=feed.last_checked,
            frequency_seconds=feed.frequency_seconds,
            created_at=feed.created_at,
            entry_count=0,
        )


# ##################################################################
# delete feed
# remove a feed and all its entries
@app.delete("/feeds/{feed_id}")
def delete_feed(feed_id: int):
    with get_session() as session:
        feed = session.query(Feed).filter(Feed.id == feed_id).first()
        if feed is None:
            raise HTTPException(status_code=404, detail="Feed not found")

        session.delete(feed)
        return {"status": "deleted", "id": feed_id}


# ##################################################################
# get stats
# return database statistics
@app.get("/stats")
def get_stats():
    with get_session() as session:
        from sqlalchemy import func

        total_entries = session.query(FeedEntry).count()
        total_feeds = session.query(Feed).count()
        total_pending = session.query(PendingEntry).count()
        total_scored = session.query(ScoredEntry).count()
        total_errors = session.query(ErrorEntry).count()

        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        entries_today = session.query(FeedEntry).filter(FeedEntry.discovered_at >= today_start).count()
        scored_today = session.query(ScoredEntry).filter(ScoredEntry.created_at >= today_start).count()

        avg_per_feed = total_entries / total_feeds if total_feeds > 0 else 0

        top_feeds = (
            session.query(Feed.name, func.count(FeedEntry.id).label("count"))
            .join(FeedEntry, Feed.id == FeedEntry.feed_id)
            .group_by(Feed.id)
            .order_by(func.count(FeedEntry.id).desc())
            .limit(3)
            .all()
        )

        top_feeds_by_avg_score = (
            session.query(Feed.name, func.avg(FeedEntry.score).label("avg_score"))
            .join(FeedEntry, Feed.id == FeedEntry.feed_id)
            .filter(FeedEntry.score.isnot(None))
            .group_by(Feed.id)
            .order_by(func.avg(FeedEntry.score).desc())
            .limit(10)
            .all()
        )

        # Count feeds with 0 articles using a subquery
        feeds_with_entries = session.query(FeedEntry.feed_id).distinct().subquery()
        feeds_with_zero = session.query(Feed).filter(~Feed.id.in_(session.query(feeds_with_entries.c.feed_id))).count()

        return {
            "total_entries": total_entries,
            "total_feeds": total_feeds,
            "feeds_with_zero_articles": feeds_with_zero,
            "total_pending": total_pending,
            "total_scored": total_scored,
            "total_errors": total_errors,
            "entries_today": entries_today,
            "scored_today": scored_today,
            "average_entries_per_feed": round(avg_per_feed, 1),
            "top_feeds": [{"name": name, "count": count} for name, count in top_feeds],
            "top_feeds_by_avg_score": [
                {"name": name, "avg_score": round(avg_score, 2)} for name, avg_score in top_feeds_by_avg_score
            ],
        }


# ##################################################################
# main
# entry point for running the server directly
if __name__ == "__main__":
    setproctitle.setproctitle("news-feed")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
