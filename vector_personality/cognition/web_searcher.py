"""WebSearcher — DuckDuckGo Instant Answer + Wikipedia lookup.

No API key required. Uses public JSON endpoints only.
aiohttp is already a project dependency.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse

import aiohttp

logger = logging.getLogger(__name__)

_DDG_URL = "https://api.duckduckgo.com/"
_TIMEOUT = aiohttp.ClientTimeout(total=8)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

async def web_search(query: str) -> str:
    """Search the web and return a concise text snippet (≤500 chars).

    Strategy:
    1. DuckDuckGo Instant Answer API  (abstract, direct answer, or topic snippets)
    2. Wikipedia Italian summary
    3. Wikipedia English summary (fallback)

    Returns an empty string when nothing useful is found.
    """
    try:
        return await asyncio.wait_for(_search(query), timeout=10)
    except asyncio.TimeoutError:
        logger.warning("web_search timed out for query: %r", query)
        return ""
    except Exception as exc:
        logger.warning("web_search error: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _search(query: str) -> str:
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        result = await _ddg_instant(session, query)
        if result:
            return result

        result = await _wikipedia_summary(session, query, lang="it")
        if result:
            return result

        result = await _wikipedia_summary(session, query, lang="en")
        if result:
            return result

    return ""


async def _ddg_instant(session: aiohttp.ClientSession, query: str) -> str:
    """Try DuckDuckGo Instant Answer API."""
    try:
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "kl": "it-it",
        }
        async with session.get(_DDG_URL, params=params) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json(content_type=None)

        # Direct abstract (usually from Wikipedia)
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            source = data.get("AbstractSource") or "DuckDuckGo"
            return f"[{source}]\n{_truncate(abstract, 480)}"

        # Direct one-liner answer (date, definition, unit conversion, etc.)
        answer = (data.get("Answer") or "").strip()
        if answer:
            return _truncate(answer, 480)

        # RelatedTopics snippets
        topics = data.get("RelatedTopics") or []
        snippets = []
        for t in topics[:4]:
            text = (t.get("Text") or "").strip()
            if text:
                snippets.append(_truncate(text, 140))
        if snippets:
            return "\n".join(snippets)

        return ""
    except Exception as exc:
        logger.debug("DDG error: %s", exc)
        return ""


async def _wikipedia_summary(
    session: aiohttp.ClientSession, query: str, lang: str = "it"
) -> str:
    """Search Wikipedia (opensearch) then fetch REST summary."""
    try:
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        search_params = {
            "action": "opensearch",
            "search": query,
            "limit": "1",
            "format": "json",
            "redirects": "resolve",
        }
        async with session.get(search_url, params=search_params) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json(content_type=None)

        titles = data[1] if len(data) > 1 else []
        if not titles:
            return ""

        title = urllib.parse.quote(titles[0].replace(" ", "_"))
        summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

        async with session.get(summary_url) as resp:
            if resp.status != 200:
                return ""
            summary_data = await resp.json(content_type=None)

        extract = (summary_data.get("extract") or "").strip()
        page_title = summary_data.get("title", query)

        if extract:
            return f"[Wikipedia {lang.upper()}: {page_title}]\n{_truncate(extract, 460)}"

        return ""
    except Exception as exc:
        logger.debug("Wikipedia %s error: %s", lang, exc)
        return ""


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"
