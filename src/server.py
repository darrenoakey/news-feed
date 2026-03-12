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

from fastapi.responses import HTMLResponse

from src.config import (
    SERVER_HOST,
    SERVER_PORT,
    PROJECT_ROOT,
    MIN_FREQUENCY_SECONDS,
    MAX_FREQUENCY_SECONDS,
    DEFAULT_FREQUENCY_SECONDS,
    FREQUENCY_ADJUSTMENT_SECONDS,
    WORKER_SLEEP_SECONDS,
)
from src.database import get_session, init_db
from src.models import Feed, FeedEntry, TitleClassification
from src.rss import fetch_rss_entries
from src.discord_sender import extract_title_from_xml
from src.title_classifier import TitleClassifierService

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
            processed = await asyncio.to_thread(_process_next_feed)
            if not processed:
                await asyncio.sleep(WORKER_SLEEP_SECONDS)
        except Exception as err:
            logger.exception("Background worker error: %s", err)
            await asyncio.sleep(WORKER_SLEEP_SECONDS)


# ##################################################################
# process next feed
# find and process the next feed that needs checking, returns True if processed
def _process_next_feed() -> bool:
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
# title classifier instance
# shared classifier service for training and prediction
title_classifier = TitleClassifierService()

TENTATIVE_WORKER_SLEEP_SECONDS = 5
TENTATIVE_BATCH_SIZE = 50
RULES_RETRAIN_THRESHOLD = 20

_last_rules_train_count: int = 0
_rules_training_lock: bool = False


# ##################################################################
# tentative worker
# pre-classify entries using rules backend to maintain a pool for balanced card presentation
async def tentative_worker():
    logger.info("Tentative worker started")
    while True:
        try:
            await asyncio.to_thread(_tentative_cycle)
        except Exception as err:
            logger.exception("Tentative worker error: %s", err)
        await asyncio.sleep(TENTATIVE_WORKER_SLEEP_SECONDS)


def _tentative_cycle():
    with get_session() as session:
        # find entries with no TitleClassification row yet
        already_classified = session.query(TitleClassification.entry_id)
        unclassified = (
            session.query(FeedEntry)
            .filter(
                FeedEntry.id.notin_(already_classified),
            )
            .limit(TENTATIVE_BATCH_SIZE)
            .all()
        )

        if not unclassified:
            return

        created = 0
        for entry in unclassified:
            title = extract_title_from_xml(entry.xml_content) or entry.guid
            prediction = title_classifier.predict(title)
            svm_prediction = title_classifier.predict_svm(title)
            if prediction or svm_prediction:
                tc = TitleClassification(
                    entry_id=entry.id,
                    title=title,
                    predicted_label=prediction[0] if prediction else None,
                    predicted_score=prediction[1] if prediction else None,
                    svm_predicted_label=svm_prediction[0] if svm_prediction else None,
                    svm_predicted_score=svm_prediction[1] if svm_prediction else None,
                )
                session.add(tc)
                created += 1

        if created:
            logger.info("Tentative worker created %d classifications", created)


# ##################################################################
# maybe retrain rules
# trigger LLM rules refinement when enough new great/good labels accumulate
def _maybe_retrain_rules():
    global _last_rules_train_count, _rules_training_lock
    if _rules_training_lock:
        return

    with get_session() as session:
        from sqlalchemy import func
        count = (
            session.query(func.count(TitleClassification.id))
            .filter(TitleClassification.human_label.in_(("great", "good")))
            .scalar()
        )

    if count - _last_rules_train_count < RULES_RETRAIN_THRESHOLD:
        return

    _rules_training_lock = True
    try:
        logger.info("Rules retraining triggered: %d new great/good labels since last train", count - _last_rules_train_count)
        from src.classifier_trainer import train_tree, train_svm
        train_tree()
        train_svm()
        title_classifier.load_model()
        title_classifier.load_svm_model()
        _last_rules_train_count = count
        logger.info("Rules retraining complete, new count baseline: %d", count)
        _refresh_tentative_predictions()
    except Exception as err:
        logger.exception("Rules retraining failed: %s", err)
    finally:
        _rules_training_lock = False


# ##################################################################
# refresh tentative predictions
# re-classify all unlabeled tentative entries with the updated rules backend
def _refresh_tentative_predictions():
    with get_session() as session:
        tentative = (
            session.query(TitleClassification)
            .filter(TitleClassification.human_label.is_(None))
            .all()
        )
        tree_updated = 0
        svm_updated = 0
        for tc in tentative:
            prediction = title_classifier.predict(tc.title)
            if prediction:
                new_label, new_score = prediction
                if new_label != tc.predicted_label:
                    tc.predicted_label = new_label
                    tc.predicted_score = new_score
                    tree_updated += 1
            svm_prediction = title_classifier.predict_svm(tc.title)
            if svm_prediction:
                new_label, new_score = svm_prediction
                if new_label != tc.svm_predicted_label:
                    tc.svm_predicted_label = new_label
                    tc.svm_predicted_score = new_score
                    svm_updated += 1
        logger.info("Refreshed tentative predictions: tree=%d svm=%d / %d total", tree_updated, svm_updated, len(tentative))


# ##################################################################
# seed rolling history
# backfill rolling accuracy from existing labeled data that had predictions
def _seed_rolling_history():
    global _last_rules_train_count

    # load both models
    title_classifier.load_model()
    title_classifier.load_svm_model()
    if not title_classifier.is_trained and not title_classifier.is_svm_trained:
        logger.info("No classifiers available — tentative classification disabled until training runs")
        return
    with get_session() as session:
        from sqlalchemy import func
        labeled = (
            session.query(TitleClassification)
            .filter(TitleClassification.human_label.isnot(None))
            .order_by(TitleClassification.classified_at.asc())
            .all()
        )
        title_classifier.set_training_count(len(labeled))
        for tc in labeled:
            prediction = title_classifier.predict(tc.title)
            if prediction:
                title_classifier.record_result(prediction[0], tc.human_label, model="tree")
            svm_prediction = title_classifier.predict_svm(tc.title)
            if svm_prediction:
                title_classifier.record_result(svm_prediction[0], tc.human_label, model="svm")

        # initialize rules retrain baseline
        _last_rules_train_count = (
            session.query(func.count(TitleClassification.id))
            .filter(TitleClassification.human_label.in_(("great", "good")))
            .scalar()
        )

    logger.info("Seeded rolling history with %d entries, rules train baseline: %d", len(labeled), _last_rules_train_count)


# ##################################################################
# lifespan
# startup and shutdown logic for the FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized, seeding rolling history in background")

    # seed in background so server starts accepting requests immediately
    seed_task = asyncio.create_task(asyncio.to_thread(_seed_rolling_history))
    feed_task = asyncio.create_task(background_worker())
    tentative_task = asyncio.create_task(tentative_worker())
    yield

    for task in [seed_task, feed_task, tentative_task]:
        task.cancel()
        try:
            await task
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

        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        entries_today = session.query(FeedEntry).filter(FeedEntry.discovered_at >= today_start).count()

        avg_per_feed = total_entries / total_feeds if total_feeds > 0 else 0

        top_feeds = (
            session.query(Feed.name, func.count(FeedEntry.id).label("count"))
            .join(FeedEntry, Feed.id == FeedEntry.feed_id)
            .group_by(Feed.id)
            .order_by(func.count(FeedEntry.id).desc())
            .limit(3)
            .all()
        )

        # Count feeds with 0 articles using a subquery
        feeds_with_entries = session.query(FeedEntry.feed_id).distinct().subquery()
        feeds_with_zero = session.query(Feed).filter(~Feed.id.in_(session.query(feeds_with_entries.c.feed_id))).count()

        # Classification stats
        total_classified = session.query(TitleClassification).count()
        total_labeled = session.query(TitleClassification).filter(TitleClassification.human_label.isnot(None)).count()

        return {
            "total_entries": total_entries,
            "total_feeds": total_feeds,
            "feeds_with_zero_articles": feeds_with_zero,
            "entries_today": entries_today,
            "average_entries_per_feed": round(avg_per_feed, 1),
            "top_feeds": [{"name": name, "count": count} for name, count in top_feeds],
            "total_classified": total_classified,
            "total_labeled": total_labeled,
        }


# ##################################################################
# trainer page
# serve the training UI HTML
@app.get("/trainer", response_class=HTMLResponse)
def trainer_page():
    html_path = PROJECT_ROOT / "static" / "trainer.html"
    return HTMLResponse(content=html_path.read_text())


# ##################################################################
# classify request model
class ClassifyRequest(BaseModel):
    id: int
    label: str


# ##################################################################
# trainer api cards
# return 21 random unclassified feed entries for labeling (21 = divisible by 3 columns)
@app.get("/trainer/api/cards")
def trainer_cards(model: str = "tree"):
    with get_session() as session:
        from sqlalchemy import func
        from sqlalchemy.orm import joinedload

        import random

        # pick which predicted_label column to band by
        if model == "svm":
            label_col = TitleClassification.svm_predicted_label
        else:
            label_col = TitleClassification.predicted_label

        tentative_base = (
            session.query(TitleClassification)
            .join(FeedEntry, TitleClassification.entry_id == FeedEntry.id)
            .options(joinedload(TitleClassification.entry).joinedload(FeedEntry.feed))
            .filter(
                TitleClassification.human_label.is_(None),
                label_col.isnot(None),
            )
        )

        bands = [
            tentative_base.filter(label_col == "great"),
            tentative_base.filter(label_col == "good"),
            tentative_base.filter(label_col == "other"),
        ]

        tcs = []
        for band_query in bands:
            band_tcs = band_query.order_by(func.random()).limit(7).all()
            tcs.extend(band_tcs)
        random.shuffle(tcs)

        # re-score with latest model so cards reflect current predictions
        for tc in tcs:
            prediction = title_classifier.predict(tc.title)
            if prediction:
                new_label, new_score = prediction
                if new_label != tc.predicted_label or new_score != tc.predicted_score:
                    tc.predicted_label = new_label
                    tc.predicted_score = new_score
            svm_pred = title_classifier.predict_svm(tc.title)
            if svm_pred:
                new_label, new_score = svm_pred
                if new_label != tc.svm_predicted_label or new_score != tc.svm_predicted_score:
                    tc.svm_predicted_label = new_label
                    tc.svm_predicted_score = new_score

        results = []
        for tc in tcs:
            entry = tc.entry
            if model == "svm":
                pred_label = tc.svm_predicted_label
                pred_score = tc.svm_predicted_score
            else:
                pred_label = tc.predicted_label
                pred_score = tc.predicted_score
            card = {
                "id": tc.id,
                "entry_id": entry.id,
                "title": tc.title,
                "feed_name": entry.feed.name if entry.feed else "Unknown",
                "predicted_label": pred_label,
                "predicted_confidence": round(pred_score, 2) if pred_score is not None else None,
            }
            results.append(card)
        return results


# ##################################################################
# trainer api classify
# save a human label for a title classification entry
@app.post("/trainer/api/classify")
def trainer_classify(req: ClassifyRequest):
    if req.label not in ("great", "good", "other"):
        raise HTTPException(status_code=400, detail="Invalid label")
    with get_session() as session:
        tc = session.query(TitleClassification).filter(TitleClassification.id == req.id).first()
        if tc is None:
            raise HTTPException(status_code=404, detail="Classification entry not found")
        # record prediction vs actual for rolling accuracy (both models)
        prediction = title_classifier.predict(tc.title)
        if prediction:
            title_classifier.record_result(prediction[0], req.label, model="tree")
        svm_prediction = title_classifier.predict_svm(tc.title)
        if svm_prediction:
            title_classifier.record_result(svm_prediction[0], req.label, model="svm")
        tc.human_label = req.label
        tc.classified_at = datetime.now(timezone.utc)
    if req.label in ("great", "good"):
        import threading
        threading.Thread(target=_maybe_retrain_rules, daemon=True).start()
    return {"status": "ok"}


# ##################################################################
# trainer api metrics
# return current model training metrics
@app.get("/trainer/api/metrics")
def trainer_metrics(model: str = "tree"):
    metrics = title_classifier.get_metrics()
    if metrics is None:
        with get_session() as session:
            count = session.query(TitleClassification).filter(TitleClassification.human_label.isnot(None)).count()
        return {"training_samples": count, "status": "not_trained"}
    # return model-specific rolling/history/backend based on query param
    if model == "svm":
        return {
            "training_samples": metrics["training_samples"],
            "last_trained": metrics["last_trained"],
            "rolling": metrics.get("svm_rolling"),
            "history": metrics.get("svm_history"),
            "backend": metrics.get("svm_backend"),
        }
    return {
        "training_samples": metrics["training_samples"],
        "last_trained": metrics["last_trained"],
        "rolling": metrics.get("rolling"),
        "history": metrics.get("history"),
        "backend": metrics.get("backend"),
    }


# ##################################################################
# trainer api retrain
# force a retrain of the sklearn classifier
@app.post("/trainer/api/retrain")
def trainer_retrain(model: str = "tree"):
    import threading

    def _do_retrain():
        global _last_rules_train_count, _rules_training_lock
        if _rules_training_lock:
            return
        _rules_training_lock = True
        try:
            if model == "svm":
                from src.classifier_trainer import train_svm
                train_svm()
                title_classifier.load_svm_model()
            else:
                from src.classifier_trainer import train_tree
                train_tree()
                title_classifier.load_model()
            with get_session() as session:
                from sqlalchemy import func
                _last_rules_train_count = (
                    session.query(func.count(TitleClassification.id))
                    .filter(TitleClassification.human_label.in_(("great", "good")))
                    .scalar()
                )
            _refresh_tentative_predictions()
        except Exception as err:
            logger.exception("Manual retrain failed: %s", err)
        finally:
            _rules_training_lock = False

    threading.Thread(target=_do_retrain, daemon=True).start()
    return {"status": "ok", "message": f"Retrain ({model}) started in background"}


# ##################################################################
# trainer api stats
# return classification label counts
@app.get("/trainer/api/stats")
def trainer_stats():
    with get_session() as session:
        from sqlalchemy import func

        counts = (
            session.query(TitleClassification.human_label, func.count(TitleClassification.id))
            .group_by(TitleClassification.human_label)
            .all()
        )
        result = {"great": 0, "good": 0, "other": 0, "unclassified": 0}
        for label, count in counts:
            if label is None:
                result["unclassified"] = count
            elif label in result:
                result[label] = count
        # also count entries without any TitleClassification row
        total_entries = session.query(FeedEntry).count()
        total_classified_rows = session.query(TitleClassification).count()
        result["unclassified"] += total_entries - total_classified_rows
        return result


# ##################################################################
# main
# entry point for running the server directly
if __name__ == "__main__":
    setproctitle.setproctitle("news-feed")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
