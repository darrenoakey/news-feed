import html
import re
import sys
from html.parser import HTMLParser
from typing import Optional

import requests

from src import config


class _HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._text: list[str] = []

    def handle_data(self, data: str) -> None:
        self._text.append(data)

    def get_text(self) -> str:
        return " ".join(self._text)


def strip_html(text: str) -> str:
    """Strip HTML tags and decode entities from text."""
    if not text:
        return ""
    # First decode HTML entities
    text = html.unescape(text)
    # Then strip tags using the parser
    parser = _HTMLTextExtractor()
    try:
        parser.feed(text)
        result = parser.get_text()
    except Exception:
        # Fallback: simple regex strip
        result = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace
    return " ".join(result.split())


# ##################################################################
# get base url
# construct the base URL for API requests
def get_base_url() -> str:
    return f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"


# ##################################################################
# feed add
# add a new feed via the API
def feed_add(url: str, name: Optional[str] = None) -> int:
    payload = {"url": url}
    if name:
        payload["name"] = name

    try:
        response = requests.post(f"{get_base_url()}/feeds", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Is it running?", file=sys.stderr)
        return 1

    if response.status_code == 200:
        data = response.json()
        print(f"Added feed: {data['name']} (id={data['id']})")
        return 0
    elif response.status_code == 400:
        print(f"Error: {response.json().get('detail', 'Unknown error')}", file=sys.stderr)
        return 1
    else:
        print(f"Error: HTTP {response.status_code}", file=sys.stderr)
        return 1


# ##################################################################
# feed list
# list all feeds via the API
def feed_list() -> int:
    try:
        response = requests.get(f"{get_base_url()}/feeds", timeout=10)
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Is it running?", file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}", file=sys.stderr)
        return 1

    feeds = response.json()
    if not feeds:
        print("No feeds configured")
        return 0

    print(f"{'ID':<6} {'Name':<30} {'Entries':<10} {'Frequency':<12} URL")
    print("-" * 100)
    for feed in feeds:
        freq_min = feed["frequency_seconds"] // 60
        print(f"{feed['id']:<6} {feed['name'][:28]:<30} {feed['entry_count']:<10} {freq_min}m{'':<10} {feed['url']}")

    return 0


# ##################################################################
# feed delete
# delete a feed via the API
def feed_delete(feed_id: int) -> int:
    try:
        response = requests.delete(f"{get_base_url()}/feeds/{feed_id}", timeout=10)
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Is it running?", file=sys.stderr)
        return 1

    if response.status_code == 200:
        print(f"Deleted feed {feed_id}")
        return 0
    elif response.status_code == 404:
        print(f"Error: Feed {feed_id} not found", file=sys.stderr)
        return 1
    else:
        print(f"Error: HTTP {response.status_code}", file=sys.stderr)
        return 1


# ##################################################################
# stats
# show database statistics via the API
def stats() -> int:
    try:
        response = requests.get(f"{get_base_url()}/stats", timeout=10)
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Is it running?", file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}", file=sys.stderr)
        return 1

    data = response.json()

    print("News Feed Statistics")
    print("=" * 40)
    print(f"Total feeds:              {data['total_feeds']}")
    print(f"Feeds with 0 articles:    {data.get('feeds_with_zero_articles', 0)}")
    print(f"Total entries:            {data['total_entries']}")
    print(f"Entries today:            {data['entries_today']}")
    print()
    print("Scoring Pipeline:")
    print(f"  Pending (to score):     {data['total_pending']}")
    print(f"  Scored (in queue):      {data['total_scored']}")
    print(f"  Errors:                 {data['total_errors']}")
    print(f"  Scored today:           {data['scored_today']}")
    print()
    print(f"Average entries per feed: {data['average_entries_per_feed']}")
    print()
    print("Top 3 Most Active Feeds:")
    for i, feed in enumerate(data["top_feeds"], 1):
        print(f"  {i}. {feed['name']}: {feed['count']} entries")
    print()
    print("Top 10 Feeds by Average Score:")
    for i, feed in enumerate(data.get("top_feeds_by_avg_score", []), 1):
        print(f"  {i:2}. {feed['name']}: {feed['avg_score']:.2f}")

    return 0


# ##################################################################
# export rss
# generate RSS 2.0 XML for entries above a score threshold
# deduplicates by both URL and title (same story from different sources)
def export_rss(min_score: float, limit: int, title: str, description: str, link: str) -> int:
    import xml.etree.ElementTree as ET
    from datetime import datetime, timezone

    from src.database import get_session
    from src.models import FeedEntry, Feed
    from src.scoring import get_training_set, ScoringError

    # Sync training scores before export to reflect any corrections
    try:
        training_items = get_training_set()
        if training_items:
            apply_training_scores(training_items)
    except ScoringError:
        pass  # Continue with export even if sync fails

    with get_session() as session:
        # Fetch more entries than needed to account for duplicates
        # We'll dedupe in Python since title/URL are in xml_content
        entries = (
            session.query(FeedEntry)
            .join(Feed)
            .filter(FeedEntry.score.isnot(None))
            .filter(FeedEntry.score >= min_score)
            .order_by(FeedEntry.discovered_at.desc())
            .limit(limit * 5)  # Fetch extra to handle duplicates
            .all()
        )

        # Build RSS 2.0 XML
        rss = ET.Element("rss", version="2.0")
        channel = ET.SubElement(rss, "channel")

        ET.SubElement(channel, "title").text = title
        ET.SubElement(channel, "link").text = link
        ET.SubElement(channel, "description").text = description
        ET.SubElement(channel, "lastBuildDate").text = datetime.now(timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S +0000"
        )
        ET.SubElement(channel, "generator").text = "news-feed curated by Darren Oakey"

        # Track seen URLs and titles for deduplication
        seen_urls = set()
        seen_titles = set()
        item_count = 0

        for entry in entries:
            if item_count >= limit:
                break

            # Parse the stored XML to extract fields
            try:
                entry_root = ET.fromstring(entry.xml_content)
            except ET.ParseError:
                continue

            # Extract title and link for deduplication
            title_elem = entry_root.find("title")
            link_elem = entry_root.find("link")

            entry_title = title_elem.text.strip().lower() if title_elem is not None and title_elem.text else None
            entry_link = link_elem.text.strip().lower() if link_elem is not None and link_elem.text else None

            # Skip if we've seen this URL or title before
            if entry_link and entry_link in seen_urls:
                continue
            if entry_title and entry_title in seen_titles:
                continue

            # Mark as seen
            if entry_link:
                seen_urls.add(entry_link)
            if entry_title:
                seen_titles.add(entry_title)

            # Create RSS item
            item = ET.SubElement(channel, "item")
            item_count += 1

            # Title
            if title_elem is not None and title_elem.text:
                ET.SubElement(item, "title").text = title_elem.text

            # Link
            if link_elem is not None and link_elem.text:
                ET.SubElement(item, "link").text = link_elem.text

            # Description/summary (strip HTML)
            summary_elem = entry_root.find("summary")
            if summary_elem is not None and summary_elem.text:
                clean_summary = strip_html(summary_elem.text)
                if clean_summary:
                    ET.SubElement(item, "description").text = clean_summary

            # Publication date
            pub_elem = entry_root.find("published")
            if pub_elem is not None and pub_elem.text:
                ET.SubElement(item, "pubDate").text = pub_elem.text

            # GUID
            ET.SubElement(item, "guid", isPermaLink="false").text = entry.guid

            # Source (the feed name)
            ET.SubElement(item, "source", url=entry.feed.url).text = entry.feed.name

            # Score (for viewer display, RSS readers will ignore)
            if entry.score is not None:
                score_rounded = round(entry.score, 1)
                ET.SubElement(item, "score").text = str(score_rounded)

        # Output XML
        xml_str = ET.tostring(rss, encoding="unicode", xml_declaration=True)
        print(xml_str)

    return 0


# ##################################################################
# apply training scores
# update database entries with scores from training data
# returns number of entries updated
def apply_training_scores(training_items: list[dict]) -> int:
    import xml.etree.ElementTree as ET
    from datetime import datetime, timezone

    from src.database import get_session
    from src.models import FeedEntry

    if not training_items:
        return 0

    # Build URL -> score mapping
    url_to_score = {item["url"]: item["score"] for item in training_items}

    updated_count = 0
    with get_session() as session:
        entries = session.query(FeedEntry).all()

        for entry in entries:
            try:
                root = ET.fromstring(entry.xml_content)
                link_elem = root.find("link")
                if link_elem is None or not link_elem.text:
                    continue
                link = link_elem.text.strip()
            except ET.ParseError:
                continue

            if link in url_to_score:
                new_score = url_to_score[link]
                if entry.score != new_score:
                    entry.score = new_score
                    entry.scored_at = datetime.now(timezone.utc)
                    updated_count += 1

        session.commit()

    return updated_count


# ##################################################################
# update trained
# update scores for entries that are in the training set
def update_trained() -> int:
    from src.scoring import get_training_set, ScoringError

    try:
        training_items = get_training_set()
    except ScoringError as err:
        print(f"Error: Failed to fetch training set: {err}", file=sys.stderr)
        return 1

    if not training_items:
        print("No training data found")
        return 0

    print(f"Fetched {len(training_items)} trained URLs from scoring API")
    updated_count = apply_training_scores(training_items)
    print(f"Updated {updated_count} entries with training scores")
    return 0
