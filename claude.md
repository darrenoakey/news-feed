# news-feed

RSS aggregator with ML-based title classification.

## Architecture

- **news-feed** (this project): RSS aggregation, storage, ML classification, RSS export

## Key Files

- `src/cli.py` - CLI commands including `export-rss-great`, `export-rss-good`
- `src/server.py` - Background workers: feed discovery, tentative classification
- `src/title_classifier.py` - TitleClassifierService wrapping sklearn decision tree
- `src/classifier_backend.py` - Decision tree model loading/prediction
- `src/classifier_trainer.py` - Training loop for the title classifier
- `static/trainer.html` - Web UI for human labeling at `/trainer`
- `local/feeds.db` - SQLite database with feed entries and classifications

## Classification Flow

1. Feed discovery worker finds new entries → creates `FeedEntry`
2. Tentative worker classifies all entries via `TitleClassifierService` → creates `TitleClassification` rows
3. `/trainer` web UI presents balanced cards (7 per label) for human labeling
4. Retraining triggers automatically after 20 new great/good labels
5. Model stored at `local/title_classifier.joblib`

## Title Classifier

- sklearn decision tree trained on human-labeled titles (great/good/other)
- Tentative worker runs every 5s, classifies 50 unclassified entries per cycle
- `trainer_cards()` re-scores cards with latest model before returning

## Label-Based RSS Feeds

- `./run export-rss-great` — articles where `predicted_label == "great"`
- `./run export-rss-good` — articles where `predicted_label == "good"`
- Dedup/generation uses shared helpers `_deduplicate_entries` and `_build_rss_xml` in cli.py

## Database Location

- Feed database: `/Users/darrenoakey/src/news-feed/local/feeds.db`

## Threading & Concurrency

- Both background workers use `asyncio.to_thread()` for blocking I/O (HTTP requests, SQLAlchemy) — never block the event loop directly
- `_seed_rolling_history` runs in a background thread on startup so the server accepts requests immediately
- SQLite WAL mode enabled for concurrent reader/writer support across threads

## Auto Daemon

Registered with `auto` using `.venv/bin/python` (not system python3) to prevent module import failures when system Python upgrades. If the service dies with `ModuleNotFoundError`, check that dependencies are installed in `.venv/`.

## RSS Export

### Deduplication and HTML Handling

- **HTML stripping**: Titles and descriptions are stripped of HTML tags and normalized (uses `strip_html()`)
- **Nested HTML**: Uses `itertext()` to extract text from nested elements (handles `<h1>Title</h1>` inside `<title>`)
- **Deduplication**: Entries are deduplicated by URL (primary) OR exact title match (secondary)

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
