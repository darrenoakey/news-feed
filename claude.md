# news-feed

RSS aggregator with ML-based scoring via external news-score service.

## Architecture

Two separate systems work together:
- **news-feed** (this project): RSS aggregation, storage, Discord publishing, RSS export
- **news-score** (separate project at `/Users/darrenoakey/src/news-score`): ML ranking service on port 19091

## Key Files

- `src/cli.py` - CLI commands including `export-rss`, `update-trained`
- `src/server.py` - Background workers: feed discovery, scoring, Discord publishing
- `src/scoring.py` - Client for news-score API
- `local/feeds.db` - SQLite database with feed entries and scores

## Scoring Flow

1. Feed discovery worker finds new entries → creates `PendingEntry`
2. Scoring worker calls `http://localhost:19091/rank?url=...` → stores score in `FeedEntry.score`
3. Discord worker publishes high-scoring items

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
