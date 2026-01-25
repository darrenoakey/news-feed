import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import feedparser
import requests

logger = logging.getLogger(__name__)


@dataclass
class RssEntry:
    guid: str
    title: str
    xml_content: str


# ##################################################################
# fetch rss entries
# fetch and parse RSS feed, returning list of entries with guid and xml
def fetch_rss_entries(url: str, timeout: int = 30) -> list[RssEntry]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    raw_xml = response.text
    parsed = feedparser.parse(raw_xml)

    entries = []
    for entry in parsed.entries:
        guid = extract_guid(entry)
        if guid is None:
            logger.warning("Skipping entry without guid in feed %s", url)
            continue

        title = extract_title(entry)
        xml_content = entry_to_xml(entry)
        entries.append(RssEntry(guid=guid, title=title, xml_content=xml_content))

    return entries


# ##################################################################
# extract guid
# get unique identifier from feed entry, trying id then link
def extract_guid(entry) -> Optional[str]:
    if hasattr(entry, "id") and entry.id:
        return entry.id
    if hasattr(entry, "link") and entry.link:
        return entry.link
    return None


# ##################################################################
# extract title
# get title from feed entry, with fallback to empty string
def extract_title(entry) -> str:
    if hasattr(entry, "title") and entry.title:
        return str(entry.title)
    return ""


# ##################################################################
# entry to xml
# convert feedparser entry to XML string representation
def entry_to_xml(entry) -> str:
    root = ET.Element("entry")

    for key in ["id", "title", "link", "summary", "published", "updated", "author"]:
        if hasattr(entry, key):
            value = getattr(entry, key)
            if value:
                child = ET.SubElement(root, key)
                child.text = str(value) if not isinstance(value, str) else value

    if hasattr(entry, "links") and entry.links:
        links_elem = ET.SubElement(root, "links")
        for link in entry.links:
            link_elem = ET.SubElement(links_elem, "link")
            for attr in ["href", "rel", "type"]:
                if attr in link:
                    link_elem.set(attr, link[attr])

    if hasattr(entry, "content") and entry.content:
        content_elem = ET.SubElement(root, "content")
        for content in entry.content:
            if "value" in content:
                content_elem.text = content["value"]
                break

    return ET.tostring(root, encoding="unicode")
