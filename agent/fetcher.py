"""
Link identification and call text fetching.

Responsibilities:
- Extract funding-call links from newsletter email text (domain whitelist + blacklist).
- Fetch call text from the linked page (HTML scrape or PDF download + conversion).
"""


def extract_call_links(email_text: str) -> list[str]:
    """Return URLs identified as funding-call links from the email body."""
    raise NotImplementedError


def fetch_call_text(url: str) -> tuple[str, str]:
    """
    Download and return the call text for a given URL.

    Returns (text, remark) where remark is an empty string on success or a
    warning/error message if content could not be retrieved.
    """
    raise NotImplementedError
