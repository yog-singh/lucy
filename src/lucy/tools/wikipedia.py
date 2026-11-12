"""The `wikipedia` tool: the best matching article's introduction, from the MediaWiki API.

One request does it: a full-text search used as a generator, returning for
each match its search rank (`index`), plain-text introduction, address and
whether it is a disambiguation page. No API key is needed.

Disambiguation: when the best match is a disambiguation page, the result says
the name is ambiguous, quotes the page's short list of main meanings, and
shows the best matching real article, so the model can answer or ask again
more precisely.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

import httpx

from lucy.core.config import WikipediaSettings
from lucy.core.types import Risk, ToolOutput
from lucy.tools.common import (
    cap,
    describe_http_error,
    error,
    make_client,
    required_text,
    shorten,
    squash,
    tidy_paragraphs,
)

SERVICE: Final = "Wikipedia"


class WikipediaTool:
    """Search Wikipedia in one language edition (risk tier read)."""

    name = "wikipedia"
    description = (
        "Look something up on Wikipedia. Returns the introduction of the best matching "
        "article and the titles of other matches. Use it for encyclopaedic facts about "
        "people, places, organisations, events, science and history. Pass the name of the "
        "topic, for example 'Marie Curie' or 'Mount Kilimanjaro', not a whole question."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The topic to look up."},
        },
        "required": ["query"],
    }
    risk = Risk.READ

    def __init__(
        self,
        settings: WikipediaSettings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings or WikipediaSettings()
        self.api_url = f"https://{self.settings.language}.wikipedia.org/w/api.php"
        self._client = make_client(self.settings.timeout, transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        query = required_text(args, "query")
        if query is None:
            return error("The wikipedia tool needs a 'query': the topic to look up.")
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": str(self.settings.other_matches + 2),
            "prop": "extracts|pageprops|info",
            "exintro": "1",
            "explaintext": "1",
            "exlimit": "max",
            "ppprop": "disambiguation",
            "inprop": "url",
            "redirects": "1",
        }
        try:
            async with asyncio.timeout(self.settings.timeout):
                response = await self._client.get(self.api_url, params=params)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, TimeoutError, ValueError) as exc:
            return error(describe_http_error(exc, SERVICE))
        if not isinstance(data, dict):
            return error(describe_http_error(TypeError(), SERVICE))
        if "error" in data:
            info = (
                data["error"].get("info", "unknown error")
                if isinstance(data["error"], dict)
                else ""
            )
            return error(f"Wikipedia could not run the search: {squash(str(info))[:200]}")
        pages = (data.get("query") or {}).get("pages") or []
        pages = sorted(
            (p for p in pages if isinstance(p, dict) and p.get("title")),
            key=lambda p: p.get("index", 1_000),
        )
        if not pages:
            return error(f"Wikipedia has no article matching {query!r}.")
        return ToolOutput(content=self._format(query, pages))

    def _format(self, query: str, pages: list[dict[str, Any]]) -> str:
        def is_disambiguation(page: dict[str, Any]) -> bool:
            return "disambiguation" in (page.get("pageprops") or {})

        articles = [p for p in pages if not is_disambiguation(p)]
        top = pages[0]
        head: list[str] = []
        if is_disambiguation(top):
            meanings = tidy_paragraphs(top.get("extract") or "")
            head.append(
                f"Note: {top['title']!r} has several meanings on Wikipedia."
                + (f" Its list begins: {shorten(squash(meanings), 600)}" if meanings else "")
            )
        if not articles:
            head.append("No single article matched; ask again with a more specific name.")
            return cap("\n".join(head), self.settings.max_chars)
        best = articles[0]
        if is_disambiguation(top):
            head.append("Closest article (it may not be the one meant):")
        head.append(f"Article: {best['title']}")
        url = best.get("canonicalurl") or best.get("fullurl")
        if url:
            head.append(f"Address: {url}")
        others = [p["title"] for p in pages if p is not best and not is_disambiguation(p)]
        others = others[: self.settings.other_matches]
        tail = f"Other matching articles: {'; '.join(others)}" if others else ""

        intro = tidy_paragraphs(best.get("extract") or "") or "(This article has no introduction.)"
        fixed = "\n".join(head) + "\nIntroduction:\n" + ("\n" + tail if tail else "")
        room = self.settings.max_chars - len(fixed) - 1
        if len(intro) > room:
            intro = cap(intro, max(room, 100))
        text = "\n".join([*head, "Introduction:", intro] + ([tail] if tail else []))
        return cap(text, self.settings.max_chars)
