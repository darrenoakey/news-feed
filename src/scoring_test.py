from src.scoring import extract_link_from_xml, get_score_for_url, get_training_set, ScoringError


# ##################################################################
# test extract link from xml
# verify link extraction from xml content
def test_extract_link_from_xml():
    xml = "<entry><id>123</id><title>Test</title><link>https://example.com/article</link></entry>"
    link = extract_link_from_xml(xml)
    assert link == "https://example.com/article"


# ##################################################################
# test extract link from xml no link
# verify returns None when no link element
def test_extract_link_from_xml_no_link():
    xml = "<entry><id>123</id><title>Test</title></entry>"
    link = extract_link_from_xml(xml)
    assert link is None


# ##################################################################
# test extract link from xml invalid
# verify returns None for invalid xml
def test_extract_link_from_xml_invalid():
    xml = "not valid xml"
    link = extract_link_from_xml(xml)
    assert link is None


# ##################################################################
# test get score for url real
# verify we can call the real scoring API
def test_get_score_for_url_real():
    result = get_score_for_url("https://openai.com/index/introducing-gpt-4o/", timeout=120)
    assert "rank" in result
    assert isinstance(result["rank"], (int, float))


# ##################################################################
# test scoring error
# verify ScoringError can be raised and caught
def test_scoring_error():
    try:
        raise ScoringError("Test error message")
    except ScoringError as err:
        assert str(err) == "Test error message"


# ##################################################################
# test get training set real
# verify we can call the real training set API
def test_get_training_set_real():
    result = get_training_set(timeout=30)
    assert isinstance(result, list)
    # If there are items, verify structure
    if len(result) > 0:
        item = result[0]
        assert "url" in item
        assert "score" in item
