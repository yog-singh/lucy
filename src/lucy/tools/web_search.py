"""The `web_search` tool: top results from a self-hosted SearXNG instance's JSON API.

The instance must have the JSON format enabled (`search.formats` in its
settings; see deploy/searxng/). SearXNG forwards the query to the search
engines it is configured to use.

Besides the results (ranked by SearXNG's score, with a date when the engine gives one),
the output starts with what SearXNG answers directly: infoboxes (from the Wikidata and
Wikipedia engines, for queries naming an entity), direct answers and spelling
corrections. Measured on this project's instance: an infobox's introduction comes from
the current Wikipedia article, but its Wikidata properties are sparse and sometimes
unresolved ids, which are dropped.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Any, Final

import httpx

from lucy.core.config import WebSearchSettings
from lucy.core.types import Risk, ToolOutput
from lucy.tools.common import (
    cap,
    describe_http_error,
    error,
    make_client,
    required_text,
    shorten,
    squash,
)

SERVICE: Final = "The search service"
SNIPPET_CHARS: Final = 300
SNIPPET_NOTE: Final = (
    "These are snippets of pages you have not read. A snippet shows what was true when it "
    "was written. Unless one states the answer in so many words and is recent enough for "
    "the question, read the most relevant page with fetch_page before answering."
)
"""Ends every result list. Measured: the same advice in the system prompt alone did not
stop the model answering from memory when an old snippet agreed with it."""
INFOBOX_CHARS: Final = 600
"""Most characters of an infobox's introduction."""
ATTRIBUTE_CHARS: Final = 100
MAX_ATTRIBUTES: Final = 8
MAX_INFOBOXES: Final = 2

_ENGINE_NAMES: Final = {"wikipedia": "Wikipedia", "wikidata": "Wikidata"}
_ENTITY_ID: Final = re.compile(r"^[QP]\d+$")
_DATE_PREFIX: Final = re.compile(r"^(\d{1,2}) ([A-Za-z]{3,9})\.? (\d{4}) (?:\.\.\.|…|·) ?")
_US_DATE_PREFIX: Final = re.compile(r"^([A-Za-z]{3,9})\.? (\d{1,2}), (\d{4}) (?:\.\.\.|…|·) ?")
""""Apr 6, 2026 · text", as the Yahoo engine gives it."""
_AGO_PREFIX: Final = re.compile(r"^(\d+ (?:minute|hour|day|week)s? ago) (?:\.\.\.|…|·) ?")
_MONTHS: Final = {
    name: number
    for number, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
    )
}


class WebSearchTool:
    """Search the web through SearXNG (risk tier read)."""

    name = "web_search"
    description = (
        "Search the web. Returns the top results, each with a title, address, short "
        "snippet and sometimes a date, and for a named person, organisation or thing an "
        "encyclopaedia summary when one exists. Use it for current events, recent news, "
        "prices, opening times, sports results and specific facts that an encyclopaedia "
        "would not have. To read one result in full, pass its address to fetch_page."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "recent": {
                "type": "boolean",
                "description": "True to search news articles, for events of the last days "
                "or weeks such as headlines or announcements. Leave it out for anything else, "
                "including weather, prices, opening times and current versions or office "
                "holders.",
            },
        },
        "required": ["query"],
    }
    risk = Risk.READ

    def __init__(
        self,
        settings: WebSearchSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not settings.searxng_url:
            raise ValueError("web_search needs searxng_url")
        self.settings = settings
        self.search_url = settings.searxng_url.rstrip("/") + "/search"
        self._client = make_client(settings.timeout, transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        query = required_text(args, "query")
        if query is None:
            return error("The web_search tool needs a 'query': what to search for.")
        news = args.get("recent") is True
        try:
            async with asyncio.timeout(self.settings.timeout):
                data = await self._search(query, news)
                if news and not _as_list(data.get("results")):
                    news = False  # no news found: the web at large may still know
                    data = await self._search(query, news)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                return error(
                    "The search service refused the request (HTTP 403); its JSON format is "
                    "probably not enabled."
                )
            return error(describe_http_error(exc, SERVICE))
        except (httpx.HTTPError, TimeoutError, ValueError) as exc:
            return error(describe_http_error(exc, SERVICE))
        if not isinstance(data, dict):
            return error(describe_http_error(TypeError(), SERVICE))
        return self._format(query, data, news=news)

    async def _search(self, query: str, news: bool) -> Any:
        """SearXNG's JSON answer; `news` asks for its news category (dated articles)."""
        params = {"q": query, "format": "json"} | ({"categories": "news"} if news else {})
        response = await self._client.get(self.search_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None

    def _format(self, query: str, data: dict[str, Any], *, news: bool = False) -> ToolOutput:
        lines: list[str] = []
        infoboxes = data.get("infoboxes")
        if isinstance(infoboxes, list):
            boxes = [box for box in map(_infobox, infoboxes) if box][:MAX_INFOBOXES]
            lines.extend(boxes)
        for answer in _as_list(data.get("answers")):
            text = answer.get("answer") if isinstance(answer, dict) else answer
            if isinstance(text, str) and text.strip():
                lines.append(f"Direct answer: {shorten(squash(text), SNIPPET_CHARS)}")
        corrections = [squash(c) for c in _as_list(data.get("corrections")) if isinstance(c, str)]
        if any(corrections):
            lines.append(f"Did you mean: {'; '.join(c for c in corrections if c)}?")

        results: list[str] = []
        seen: set[str] = set()
        items = [item for item in _as_list(data.get("results")) if isinstance(item, dict)]
        # SearXNG's own list is not always in score order (results from one engine can
        # come back as 1, 9, 15, ..., 2, 3), so rank by its score; ties keep their order.
        for item in sorted(items, key=_score, reverse=True):
            url, title = item.get("url"), squash(str(item.get("title") or ""))
            if not isinstance(url, str) or not url or url in seen:
                continue
            seen.add(url)
            content, dated = _split_date(squash(str(item.get("content") or "")))
            snippet = shorten(content, SNIPPET_CHARS)
            published = _published(item.get("publishedDate")) or dated
            if published:
                snippet = f"Published {published}. {snippet}".rstrip()
            entry = f"{len(results) + 1}. {title or url}\n   {url}"
            if snippet:
                entry += f"\n   {snippet}"
            results.append(entry)
            if len(results) >= self.settings.max_results:
                break

        if not results and not lines:
            failed = data.get("unresponsive_engines") or []
            if failed:
                return error(
                    f"No results for {query!r}; some search engines did not respond. "
                    "Try again later or rephrase."
                )
            return error(f"No results for {query!r}.")
        heading = f"{'News' if news else 'Web search'} results for {query!r}:"
        text = "\n".join([heading, *lines, *results])
        if not results:
            return ToolOutput(content=cap(text, self.settings.max_chars))
        room = max(0, self.settings.max_chars - len(SNIPPET_NOTE) - 1)
        return ToolOutput(content=f"{cap(text, room)}\n{SNIPPET_NOTE}")


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _score(item: dict[str, Any]) -> float:
    score = item.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return 0.0
    return float(score)


def _infobox(box: object) -> str | None:
    """A compact summary of one SearXNG infobox, or None if it has nothing usable.

    Infoboxes come from the Wikidata and Wikipedia engines for queries that name a
    person, organisation, place or thing: the title, the article's introduction (or a
    short description) and some Wikidata properties. Properties whose value contains
    an unresolved Wikidata id (such as "Q106028933") are left out; so are links other
    than the article and the official website.
    """
    if not isinstance(box, dict):
        return None
    title = squash(str(box.get("infobox") or ""))
    content = box.get("content")
    content = shorten(squash(content), INFOBOX_CHARS) if isinstance(content, str) else ""
    facts = []
    for attribute in _as_list(box.get("attributes"))[: MAX_ATTRIBUTES * 2]:
        fact = _fact(attribute)
        if fact:
            facts.append(fact)
    if not (content or facts):
        return None
    engines = [e for e in _as_list(box.get("engines")) if isinstance(e, str) and e.strip()]
    source = " and ".join(_ENGINE_NAMES.get(e, e.strip()) for e in engines[:3]) or "the web"
    lines = [f"Summary from {source}: {title or 'untitled'}"]
    urls = [u for u in _as_list(box.get("urls")) if isinstance(u, dict)]
    address = box.get("id")
    if not _is_web_address(address):
        address = next(
            (u.get("url") for u in urls if "wikipedia" in str(u.get("title")).lower()), None
        )
    if _is_web_address(address):
        lines.append(f"   {address}")
    if content:
        lines.append(f"   {content}")
    if facts:
        lines.append(f"   Facts: {'; '.join(facts[:MAX_ATTRIBUTES])}.")
    official = next((u.get("url") for u in urls if u.get("official") is True), None)
    if _is_web_address(official) and official != address:
        lines.append(f"   Official website: {official}")
    return "\n".join(lines)


def _fact(attribute: object) -> str | None:
    """The text "label: value" for an infobox attribute, or None if it is malformed or
    its value contains an unresolved Wikidata id (a list missing one item would mislead).
    """
    if not isinstance(attribute, dict):
        return None
    label, value = attribute.get("label"), attribute.get("value")
    if not isinstance(label, str) or not isinstance(value, str):
        return None
    label, value = squash(label), squash(value)
    if not label or not value or any(_ENTITY_ID.match(p.strip()) for p in value.split(",")):
        return None
    return f"{label}: {shorten(value, ATTRIBUTE_CHARS)}"


def _is_web_address(value: object) -> bool:
    return isinstance(value, str) and value.startswith(("https://", "http://"))


def _split_date(snippet: str) -> tuple[str, str | None]:
    """A snippet without the date some engines put in front ("9 Sept 2022 ... text",
    "Apr 6, 2026 · text"), and that date as ISO 8601 (or a relative "3 days ago"), or
    None."""
    for pattern, order in ((_DATE_PREFIX, (0, 1, 2)), (_US_DATE_PREFIX, (1, 0, 2))):
        match = pattern.match(snippet)
        if not match:
            continue
        day, month, year = (match.group(i + 1) for i in order)
        number = _MONTHS.get(month[:3].lower())
        try:
            when = date(int(year), number, int(day)).isoformat() if number else None
        except ValueError:
            when = None
        if when:
            return snippet[match.end() :], when
    match = _AGO_PREFIX.match(snippet)
    if match:
        return snippet[match.end() :], match.group(1)
    return snippet, None


def _published(value: object) -> str | None:
    """The date part of SearXNG's `publishedDate` (ISO 8601), or None if absent or invalid.

    Only some engines report one; it tells the model how current a result is.
    """
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10]).isoformat()
    except ValueError:
        return None
