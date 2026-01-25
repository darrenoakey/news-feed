from src.rss import fetch_rss_entries, extract_guid, extract_title, entry_to_xml


# ##################################################################
# test fetch rss entries real
# verify we can fetch real RSS feeds from the internet
def test_fetch_rss_entries_openai():
    entries = fetch_rss_entries("https://openai.com/news/rss.xml")
    assert len(entries) > 0
    for entry in entries:
        assert entry.guid is not None
        assert len(entry.guid) > 0
        assert entry.xml_content is not None
        assert "<entry>" in entry.xml_content


# ##################################################################
# test fetch rss entries deepmind
# verify we can fetch DeepMind RSS feed
def test_fetch_rss_entries_deepmind():
    entries = fetch_rss_entries("https://deepmind.com/blog/feed/basic")
    assert len(entries) > 0
    for entry in entries:
        assert entry.guid is not None
        assert len(entry.guid) > 0


# ##################################################################
# test entry to xml structure
# verify xml output has expected structure
def test_entry_to_xml_structure():
    class FakeEntry:
        id = "test-123"
        title = "Test Title"
        link = "https://example.com/test"
        summary = "Test summary"

    xml = entry_to_xml(FakeEntry())
    assert "<entry>" in xml
    assert "<id>test-123</id>" in xml
    assert "<title>Test Title</title>" in xml
    assert "<link>https://example.com/test</link>" in xml


# ##################################################################
# test extract guid prefers id
# verify guid extraction prefers id over link
def test_extract_guid_prefers_id():
    class EntryWithBoth:
        id = "preferred-id"
        link = "https://example.com/fallback"

    guid = extract_guid(EntryWithBoth())
    assert guid == "preferred-id"


# ##################################################################
# test extract guid falls back to link
# verify guid extraction uses link when no id
def test_extract_guid_fallback_to_link():
    class EntryWithLink:
        link = "https://example.com/as-guid"

    guid = extract_guid(EntryWithLink())
    assert guid == "https://example.com/as-guid"


# ##################################################################
# test extract title
# verify title extraction works
def test_extract_title():
    class EntryWithTitle:
        title = "Test Article Title"

    title = extract_title(EntryWithTitle())
    assert title == "Test Article Title"


# ##################################################################
# test extract title fallback
# verify title returns empty string when no title
def test_extract_title_fallback():
    class EntryWithoutTitle:
        pass

    title = extract_title(EntryWithoutTitle())
    assert title == ""


# ##################################################################
# test fetch rss entries has titles
# verify entries from real feeds have titles
def test_fetch_rss_entries_has_titles():
    entries = fetch_rss_entries("https://openai.com/news/rss.xml")
    assert len(entries) > 0
    titles_found = sum(1 for e in entries if e.title)
    assert titles_found > 0
