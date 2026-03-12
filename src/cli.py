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
    print(f"Average entries per feed: {data['average_entries_per_feed']}")
    print()
    print("Classification:")
    print(f"  Total classified:       {data.get('total_classified', 0)}")
    print(f"  Human labeled:          {data.get('total_labeled', 0)}")
    print()
    print("Top 3 Most Active Feeds:")
    for i, feed in enumerate(data["top_feeds"], 1):
        print(f"  {i}. {feed['name']}: {feed['count']} entries")

    return 0


# ##################################################################
# deduplicate entries
# group entries by URL/title for deduplication with score averaging
def _deduplicate_entries(entries: list) -> dict:
    import xml.etree.ElementTree as ET

    entry_groups: dict[tuple[Optional[str], Optional[str]], list[tuple]] = {}

    for entry in entries:
        try:
            entry_root = ET.fromstring(entry.xml_content)
        except ET.ParseError:
            continue

        title_elem = entry_root.find("title")
        link_elem = entry_root.find("link")

        raw_title = "".join(title_elem.itertext()) if title_elem is not None else None
        entry_title = strip_html(raw_title).lower() if raw_title else None
        entry_link = link_elem.text.strip().lower() if link_elem is not None and link_elem.text else None

        matched_key = None
        for existing_key, existing_entries in entry_groups.items():
            _, existing_root = existing_entries[0]
            existing_link_elem = existing_root.find("link")
            existing_title_elem = existing_root.find("title")
            existing_url = (
                existing_link_elem.text.strip().lower()
                if existing_link_elem is not None and existing_link_elem.text
                else None
            )
            existing_title_text = (
                "".join(existing_title_elem.itertext()) if existing_title_elem is not None else None
            )
            existing_title = strip_html(existing_title_text).lower() if existing_title_text else None

            if entry_link and existing_url and entry_link == existing_url:
                matched_key = existing_key
                break
            if entry_title and existing_title and entry_title == existing_title:
                matched_key = existing_key
                break

        if matched_key:
            entry_groups[matched_key].append((entry, entry_root))
        else:
            group_key = (entry_link or f"notitle-{len(entry_groups)}", entry_title)
            entry_groups[group_key] = [(entry, entry_root)]

    return entry_groups


# ##################################################################
# build rss xml
# generate RSS 2.0 XML from deduplicated entry groups and print it
def _build_rss_xml(
    entry_groups: dict, limit: int, title: str, description: str, link: str
) -> None:
    import xml.etree.ElementTree as ET
    from datetime import datetime, timezone

    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = title
    ET.SubElement(channel, "link").text = link
    ET.SubElement(channel, "description").text = description
    ET.SubElement(channel, "lastBuildDate").text = datetime.now(timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S +0000"
    )
    ET.SubElement(channel, "generator").text = "news-feed curated by Darren Oakey"

    item_count = 0
    for group_key, group_entries in entry_groups.items():
        if item_count >= limit:
            break

        entry, entry_root = group_entries[0]

        item = ET.SubElement(channel, "item")
        item_count += 1

        title_elem = entry_root.find("title")
        if title_elem is not None:
            raw_title = "".join(title_elem.itertext())
            if raw_title:
                clean_title = strip_html(raw_title)
                if clean_title:
                    ET.SubElement(item, "title").text = clean_title

        link_elem = entry_root.find("link")
        if link_elem is not None and link_elem.text:
            ET.SubElement(item, "link").text = link_elem.text

        summary_elem = entry_root.find("summary")
        if summary_elem is not None and summary_elem.text:
            clean_summary = strip_html(summary_elem.text)
            if clean_summary:
                ET.SubElement(item, "description").text = clean_summary

        pub_elem = entry_root.find("published")
        if pub_elem is not None and pub_elem.text:
            ET.SubElement(item, "pubDate").text = pub_elem.text

        ET.SubElement(item, "guid", isPermaLink="false").text = entry.guid
        ET.SubElement(item, "source", url=entry.feed.url).text = entry.feed.name

    xml_str = ET.tostring(rss, encoding="unicode", xml_declaration=True)
    print(xml_str)


# ##################################################################
# export rss by label
# generate RSS 2.0 XML for entries classified with a given label
# deduplicates by both URL and title (same story from different sources)
def export_rss_by_label(
    label: str, limit: int, title: str, description: str, link: str
) -> int:
    from src.database import get_session
    from src.models import FeedEntry, Feed, TitleClassification

    with get_session() as session:
        entries = (
            session.query(FeedEntry)
            .join(Feed)
            .join(TitleClassification, TitleClassification.entry_id == FeedEntry.id)
            .filter(TitleClassification.svm_predicted_label == label)
            .order_by(FeedEntry.discovered_at.desc())
            .limit(limit * 5)
            .all()
        )

        entry_groups = _deduplicate_entries(entries)
        _build_rss_xml(entry_groups, limit, title, description, link)

    return 0
