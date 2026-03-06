# news-feed

RSS aggregator with ML-based scoring via external news-score service.

## Architecture

Two separate systems work together:
- **news-feed** (this project): RSS aggregation, storage, Discord publishing, RSS export
- **news-score** (separate project at `/Users/darrenoakey/src/news-score`): ML ranking service on port 19091

## Key Files

- `src/cli.py` - CLI commands including `export-rss`, `export-rss-great`, `export-rss-good`, `update-trained`
- `src/server.py` - Background workers: feed discovery, scoring, Discord publishing, tentative classification
- `src/scoring.py` - Client for news-score API
- `src/title_classifier.py` - TitleClassifierService wrapping sklearn decision tree
- `src/classifier_backend.py` - Decision tree model loading/prediction
- `src/classifier_trainer.py` - Training loop for the title classifier
- `static/trainer.html` - Web UI for human labeling at `/trainer`
- `local/feeds.db` - SQLite database with feed entries and scores

## Scoring Flow

1. Feed discovery worker finds new entries → creates `PendingEntry`
2. Scoring worker calls `http://localhost:19091/rank?url=...` → stores score in `FeedEntry.score`
3. Discord worker publishes high-scoring items
4. Tentative worker classifies all scored entries via `TitleClassifierService` → creates `TitleClassification` rows

## Title Classifier

- sklearn decision tree trained on human-labeled titles (great/good/other)
- Tentative worker runs every 5s, classifies 50 unclassified entries per cycle
- `/trainer` web UI presents balanced cards (7 per label) for human labeling
- `trainer_cards()` re-scores cards with latest model before returning
- Retraining triggers automatically after 20 new great/good labels
- Model stored at `local/title_classifier.joblib`

## Label-Based RSS Feeds

- `./run export-rss-great` — articles where `predicted_label == "great"`
- `./run export-rss-good` — articles where `predicted_label == "good"`
- Same dedup/generation as score-based `export-rss` (shared helpers `_deduplicate_entries` and `_build_rss_xml` in cli.py)

## Critical Gotcha: Score Sync

When a score is corrected in news-score (via `/correct_rank`), the news-feed database is NOT automatically updated. The `export_rss` function now auto-syncs training scores before export, but if scores seem stale:
- Run `./run update-trained` to sync all trained scores
- Or check `corrections` table in news-score: `sqlite3 /Users/darrenoakey/src/news-score/local/news_ranker.db "SELECT * FROM corrections"`

## Database Locations

- Feed database: `/Users/darrenoakey/src/news-feed/local/feeds.db`
- news-score database: `/Users/darrenoakey/src/news-score/local/news_ranker.db`

## Auto Daemon

Registered with `auto` using `.venv/bin/python` (not system python3) to prevent module import failures when system Python upgrades. If the service dies with `ModuleNotFoundError`, check that dependencies are installed in `.venv/`.

## RSS Export

- `--min-score`: Inclusive lower bound
- `--max-score`: Exclusive upper bound (enables non-overlapping feeds)
- Example: `--min-score 8 --max-score 9.5` gives scores in range [8, 9.5)

### Deduplication and HTML Handling

- **HTML stripping**: Titles and descriptions are stripped of HTML tags and normalized (uses `strip_html()`)
- **Nested HTML**: Uses `itertext()` to extract text from nested elements (handles `<h1>Title</h1>` inside `<title>`)
- **Deduplication**: Entries are deduplicated by URL (primary) OR exact title match (secondary)
- **Score averaging**: When duplicates exist (same article from multiple feeds), their scores are averaged

## Testing

```bash
./run test src/cli_test.py  # Run specific test file
./run lint                   # Run linting
```

## Entry XML Structure

Feed entries store original XML in `xml_content` field. Extract link with:
```python
root = ET.fromstring(entry.xml_content)
link = root.find("link").text
```
