![](banner.jpg)

# News Feed

A personal RSS feed aggregator that collects articles from multiple sources, scores them using AI, and publishes curated content to Discord and as an RSS feed.

## Purpose

This project solves the problem of information overload from technical news sources. It:

- Aggregates RSS feeds from various tech news sources
- Uses AI to score articles based on relevance and quality
- Sends high-scoring articles to Discord
- Generates a curated RSS feed of the best content

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables for Discord webhook and AI scoring (see configuration section)

## Usage

The project provides a CLI through the `./run` script.

### Managing Feeds

Add a new RSS feed:
```bash
./run feed add https://example.com/rss.xml --name "Example Feed"
```

List all configured feeds:
```bash
./run feed list
```

Delete a feed by ID:
```bash
./run feed delete 5
```

### Viewing Statistics

Show database statistics including feed counts and entry totals:
```bash
./run stats
```

### Exporting Curated Content

Export high-scoring entries as an RSS feed:
```bash
./run export-rss --min-score 8.0 --limit 100
```

Customize the output:
```bash
./run export-rss --min-score 9.0 --limit 50 --title "My Curated Feed" --link "https://mysite.com/feed"
```

### Testing Discord Integration

Send the highest-rated article to Discord to test formatting:
```bash
./run test-discord-send
```

### Monitoring

Tail the server logs:
```bash
./run monitor
```

### Development Commands

Run a specific test:
```bash
./run test src/models_test.py::test_feed_creation
```

Run the linter:
```bash
./run lint
```

Run the full test suite:
```bash
./run check
```

Update scores for training set entries:
```bash
./run update-trained
```

## Examples

### Setting Up a New Feed Collection

```bash
# Add some tech news feeds
./run feed add https://dev.to/feed --name "Dev.to"
./run feed add https://techcrunch.com/feed/ --name "TechCrunch"
./run feed add https://www.wired.com/feed/rss --name "WIRED"

# Verify feeds are configured
./run feed list

# Check statistics
./run stats
```

### Generating a Curated Newsletter

```bash
# Export only the best articles (score >= 9.0) from the last batch
./run export-rss --min-score 9.0 --limit 20 --title "Weekly Tech Digest"
```

### Testing the Pipeline

```bash
# Run all tests
./run check

# Run specific test file
./run test src/scoring_test.py

# Check code quality
./run lint
```