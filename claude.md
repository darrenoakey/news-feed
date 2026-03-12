# news-feed

RSS aggregator with ML-based title classification.

## Architecture

- **news-feed** (this project): RSS aggregation, storage, ML classification, RSS export

## Key Files

- `src/cli.py` - CLI commands including `export-rss-great`, `export-rss-good`
- `src/server.py` - Background workers: feed discovery, tentative classification
- `src/title_classifier.py` - TitleClassifierService wrapping both tree and SVM backends
- `src/classifier_backend.py` - TreeBackend (decision tree) and SVMBackend (TF-IDF + LinearSVC)
- `src/classifier_trainer.py` - Training loops: `train_tree()` and `train_svm()`
- `static/trainer.html` - Web UI for human labeling at `/trainer`
- `local/feeds.db` - SQLite database with feed entries and classifications

## Classification Flow

1. Feed discovery worker finds new entries → creates `FeedEntry`
2. Tentative worker classifies all entries via `TitleClassifierService` → creates `TitleClassification` rows
3. `/trainer` web UI presents balanced cards (7 per label) for human labeling
4. Retraining triggers automatically after 20 new great/good labels
5. Models stored at `local/classifier_model.joblib` (tree) and `local/svm_model.joblib` (SVM)

## Title Classifiers

Two side-by-side classifiers, same human labels feed both:

- **Tree**: sklearn DecisionTreeClassifier with CountVectorizer (bigrams, binary). Model at `local/classifier_model.joblib`
- **SVM** (primary): TF-IDF + LinearSVC. Model at `local/svm_model.joblib`. Used for RSS feeds and portfolio

Trainer UI (`/trainer`) has Tree/SVM toggle to switch which model's predictions are displayed and retrained. Tentative worker classifies with both models. Rolling accuracy tracked per model.

DB columns: `predicted_label`/`predicted_score` (tree), `svm_predicted_label`/`svm_predicted_score` (SVM)

## Label-Based RSS Feeds

- `./run export-rss-great` — articles where `svm_predicted_label == "great"`
- `./run export-rss-good` — articles where `svm_predicted_label == "good"`
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
