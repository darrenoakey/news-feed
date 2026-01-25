import logging
import urllib.request
import urllib.parse
import urllib.error
import json
import xml.etree.ElementTree as ET
from typing import Optional

from src.config import SCORING_API_URL

logger = logging.getLogger(__name__)


# ##################################################################
# extract link from xml
# parse xml content and extract the link element
def extract_link_from_xml(xml_content: str) -> Optional[str]:
    try:
        root = ET.fromstring(xml_content)
        link_elem = root.find("link")
        if link_elem is not None and link_elem.text:
            return link_elem.text
        return None
    except ET.ParseError as err:
        logger.error("Failed to parse XML: %s", err)
        return None


# ##################################################################
# get score for url
# call the scoring API to get a score for a URL
def get_score_for_url(url: str, timeout: int = 30) -> dict:
    params = urllib.parse.urlencode({"url": url})
    target = f"{SCORING_API_URL}/rank?{params}"

    try:
        req = urllib.request.Request(target, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        raise ScoringError(f"HTTP error {err.code}: {err.reason}") from err
    except urllib.error.URLError as err:
        raise ScoringError(f"Connection error: {err.reason}") from err
    except json.JSONDecodeError as err:
        raise ScoringError(f"Invalid JSON response: {err}") from err
    except TimeoutError as err:
        raise ScoringError(f"Request timed out after {timeout}s") from err
    except OSError as err:
        if "timed out" in str(err).lower():
            raise ScoringError(f"Request timed out after {timeout}s") from err
        raise ScoringError(f"OS error: {err}") from err


# ##################################################################
# scoring error
# exception raised when scoring fails
class ScoringError(Exception):
    pass


# ##################################################################
# get training set
# fetch all trained URLs and their scores from the scoring API
def get_training_set(timeout: int = 30) -> list[dict]:
    target = f"{SCORING_API_URL}/training_set"

    try:
        req = urllib.request.Request(target, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
            return data.get("items", [])
    except urllib.error.HTTPError as err:
        raise ScoringError(f"HTTP error {err.code}: {err.reason}") from err
    except urllib.error.URLError as err:
        raise ScoringError(f"Connection error: {err.reason}") from err
    except json.JSONDecodeError as err:
        raise ScoringError(f"Invalid JSON response: {err}") from err
    except TimeoutError as err:
        raise ScoringError(f"Request timed out after {timeout}s") from err
    except OSError as err:
        if "timed out" in str(err).lower():
            raise ScoringError(f"Request timed out after {timeout}s") from err
        raise ScoringError(f"OS error: {err}") from err
