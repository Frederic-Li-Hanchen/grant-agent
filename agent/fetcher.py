"""
Link identification and call text fetching.

Responsibilities:
- Extract funding-call links from newsletter email text.
- Fetch call text from the linked page (HTML scrape or PDF download + conversion).
"""
import re
import io
import random
import time
import requests
from bs4 import BeautifulSoup
from typing import Any


# ---------------------------------------------------------------------------
# Link identification
# ---------------------------------------------------------------------------

def _join_wrapped_urls(email_text: str) -> str:
    """
    Plain-text email clients wrap long URLs across lines with leading whitespace
    inside angle brackets, e.g.:
        <https://www.example.de/
        SharedDocs/some-long-path.html>
    This function joins those fragments into single-line <URL> tokens so that
    subsequent URL extraction works on complete URLs.
    """
    # Match an opening '<' followed by 'http', then consume any characters
    # (including newlines + leading whitespace on continuation lines) up to '>'.
    def _join(m: re.Match) -> str:
        inner = m.group(1)
        # Collapse any whitespace (spaces, tabs, newlines) inside the URL
        joined = re.sub(r'\s+', '', inner)
        return f'<{joined}>'

    return re.sub(r'<(https?://[^>]+?)>', _join, email_text, flags=re.DOTALL)


def extract_call_links(email_text: str, config: dict[str, Any]) -> list[str]:
    """
    Return URLs identified as funding-call links from the newsletter email body.

    Strategy:
    1. Join multi-line <URL> tokens into single-line tokens.
    2. Find URLs that are preceded (within 500 characters) by a call-entry header
       matching the pattern: DD.MM.YYYY - DD.MM.YYYY | MINISTRY | Bekanntmachung.
       This pattern is specific to funding call entries in the Förderinfo newsletter
       and naturally excludes URLs from other sections (events, prizes, etc.).
    3. Apply URL blacklist as a safety net.
    4. Return a deduplicated, order-preserving list.

    Note: call_section_markers in config is kept for documentation purposes and can
    be used as a fallback for newsletter formats that do not use the date-range pattern.
    """
    cfg = config.get('link_identification', {})
    blacklist: list[str] = cfg.get('url_blacklist_patterns', [])

    # Step 1: normalise multi-line URLs
    normalised = _join_wrapped_urls(email_text)

    # Step 2: find URLs preceded by a call-entry date-range header
    # Pattern: DD.MM.YYYY - DD.MM.YYYY | *MINISTRY* | Bekanntmachung ... <URL>
    call_entry_re = re.compile(
        r'\d{2}\.\d{2}\.\d{4}\s*[-–]\s*\d{2}\.\d{2}\.\d{4}'              # date range
        r'\s*\|\s*\*?[\w\s]+\*?'                                            # ministry
        r'\s*\|\s*(?:Bekanntmachung|Förderaufruf|Förderwettbewerb)'        # call label
        r'.{0,500}?'                                                        # call title text (non-greedy)
        r'<(https?://[^\s>]+)>',                                            # URL
        re.DOTALL
    )

    urls: list[str] = [m.group(1) for m in call_entry_re.finditer(normalised)]

    # Step 3: apply blacklist
    def _is_blacklisted(url: str) -> bool:
        url_lower = url.lower()
        return any(
            isinstance(pattern, str) and pattern.lower() in url_lower
            for pattern in blacklist
        )

    urls = [u for u in urls if not _is_blacklisted(u)]

    # Step 4: deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)

    return result


# ---------------------------------------------------------------------------
# Call text fetching
# ---------------------------------------------------------------------------

def _text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pdfminer.six."""
    from pdfminer.high_level import extract_text

    return extract_text(io.BytesIO(pdf_bytes))


def _scrape_page_text(soup: BeautifulSoup) -> str:
    """
    Extract main body text from a BeautifulSoup-parsed page.
    Tries content containers in order: div#content → <main> → <body>.
    """
    container = (
        soup.find('div', id='content')
        or soup.find('main')
        or soup.find('body')
    )
    if container is None:
        return ''

    lines = []
    for element in container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
        text = element.get_text(separator=' ', strip=True)
        text = text.replace('\u00ad', '').replace('\u00a0', ' ').replace('\u2001', ' ')
        if text:
            lines.append(text)

    return '\n'.join(lines)


def _score_pdf_links(soup: BeautifulSoup, base_url: str, keywords: list[str]) -> list[tuple[int, str]]:
    """
    Find all PDF links on the page and score each by how many relevance keywords
    appear in its URL or anchor text. Returns a list of (score, absolute_url)
    sorted descending by score.
    """
    from urllib.parse import urljoin

    scored: list[tuple[int, str]] = []
    for tag in soup.find_all('a', href=True):
        href: str = tag['href']
        anchor_text: str = tag.get_text(strip=True)
        absolute = urljoin(base_url, href)

        if not (href.lower().endswith('.pdf') or '.pdf' in href.lower()):
            continue

        combined = (absolute + ' ' + anchor_text).lower()
        score = sum(1 for kw in keywords if kw.lower() in combined)
        scored.append((score, absolute))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def fetch_call_text(url: str, config: dict[str, Any]) -> tuple[str, str]:
    """
    Download and return the call text for a given URL.

    Returns (text, remark) where remark is an empty string on success or a
    warning/error message if content could not be retrieved.
    """
    cfg = config.get('fetch', {})
    timeout: int = cfg.get('request_timeout', 15)
    min_delay: float = cfg.get('min_request_delay', 1.0)
    max_delay: float = cfg.get('max_request_delay', 3.0)
    min_length: int = cfg.get('min_text_length', 200)
    pdf_keywords: list[str] = cfg.get('pdf_relevance_keywords', [])

    def _sleep():
        time.sleep(random.uniform(min_delay, max_delay))

    # --- Case 1: direct PDF URL ---
    if re.search(r'\.pdf(\?.*)?$', url, re.IGNORECASE) or '%20' in url and '.pdf' in url.lower():
        try:
            _sleep()
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            text = _text_from_pdf_bytes(response.content)
            if len(text.strip()) < min_length:
                return '', 'PDF downloaded but extracted text is too short or empty'
            return text, ''
        except requests.RequestException as e:
            return '', f'error fetching PDF: {e}'
        except Exception as e:
            return '', f'error converting PDF: {e}'

    # --- Case 2: HTML page ---
    try:
        _sleep()
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        return '', f'error fetching page: {e}'

    soup = BeautifulSoup(response.text, 'html.parser')
    text = _scrape_page_text(soup)

    if len(text.strip()) >= min_length:
        return text, ''

    # --- Case 3: page text too short — try PDF fallback ---
    scored_pdfs = _score_pdf_links(soup, url, pdf_keywords)
    if not scored_pdfs:
        if len(text.strip()) == 0:
            return '', 'page returned no text and no PDF links were found'
        # Return what little text we have with a warning
        return text, 'page text is shorter than expected'

    # Try candidates in descending score order
    for score, pdf_url in scored_pdfs:
        try:
            _sleep()
            pdf_response = requests.get(pdf_url, timeout=timeout)
            pdf_response.raise_for_status()
            pdf_text = _text_from_pdf_bytes(pdf_response.content)
            if len(pdf_text.strip()) >= min_length:
                return pdf_text, ''
        except Exception:
            continue

    return '', 'page text too short and all PDF candidates failed'
