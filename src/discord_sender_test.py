from src.discord_sender import extract_title_from_xml, extract_summary_from_xml, format_news_message


# ##################################################################
# test extract title from xml
# verify title extraction from xml content
def test_extract_title_from_xml():
    xml = "<entry><id>123</id><title>Test Title</title><link>https://example.com</link></entry>"
    title = extract_title_from_xml(xml)
    assert title == "Test Title"


# ##################################################################
# test extract title from xml no title
# verify returns None when no title element
def test_extract_title_from_xml_no_title():
    xml = "<entry><id>123</id><link>https://example.com</link></entry>"
    title = extract_title_from_xml(xml)
    assert title is None


# ##################################################################
# test extract title from xml invalid
# verify returns None for invalid xml
def test_extract_title_from_xml_invalid():
    xml = "not valid xml"
    title = extract_title_from_xml(xml)
    assert title is None


# ##################################################################
# test extract summary from xml
# verify summary extraction from xml content
def test_extract_summary_from_xml():
    xml = "<entry><id>123</id><summary>This is a test summary.</summary></entry>"
    summary = extract_summary_from_xml(xml)
    assert summary == "This is a test summary."


# ##################################################################
# test extract summary from xml no summary
# verify returns None when no summary element
def test_extract_summary_from_xml_no_summary():
    xml = "<entry><id>123</id><title>Test</title></entry>"
    summary = extract_summary_from_xml(xml)
    assert summary is None


# ##################################################################
# test format news message
# verify message formatting with all fields
def test_format_news_message():
    message = format_news_message(
        title="Test Article",
        link="https://example.com/article",
        score=8.5,
        feed_name="Test Feed",
        summary="This is a summary.",
    )
    assert "**8.5**" in message
    assert "Test Feed" in message
    assert "**Test Article**" in message
    assert "This is a summary." in message
    assert "https://example.com/article" in message


# ##################################################################
# test format news message no summary
# verify message formatting without summary
def test_format_news_message_no_summary():
    message = format_news_message(
        title="Test Article", link="https://example.com/article", score=9.0, feed_name="Tech News"
    )
    assert "**9.0**" in message
    assert "Tech News" in message
    assert "**Test Article**" in message
    assert "https://example.com/article" in message


# ##################################################################
# test format news message long summary
# verify summary truncation for long text
def test_format_news_message_long_summary():
    long_summary = "x" * 300  # More than 200 chars
    message = format_news_message(
        title="Test", link="https://example.com", score=7.5, feed_name="Feed", summary=long_summary
    )
    # Should be truncated with ...
    assert "..." in message
    # Should not contain full 300 chars
    assert "x" * 300 not in message
