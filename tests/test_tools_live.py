"""The internet tools against the real services. Run with `pytest -m live`.

The weather test asks Open-Meteo about Porto. The SearXNG test needs the
instance from deploy/searxng/ on 127.0.0.1:8890 and is skipped when it is not
running.
"""

from __future__ import annotations

import httpx
import pytest

from lucy.core.config import WebSearchSettings
from lucy.tools.fetch_page import FetchPageTool
from lucy.tools.weather import WeatherTool
from lucy.tools.web_search import WebSearchTool
from lucy.tools.wikipedia import WikipediaTool

pytestmark = pytest.mark.live
SEARXNG = "http://127.0.0.1:8890"


async def test_live_wikipedia() -> None:
    tool = WikipediaTool()
    try:
        out = await tool.run({"query": "Eiffel Tower"})
    finally:
        await tool.aclose()
    assert not out.is_error, out.content
    assert out.content.startswith("Article: Eiffel Tower\nAddress: https://en.wikipedia.org/")
    assert "Paris" in out.content


async def test_live_searxng() -> None:
    try:
        httpx.get(SEARXNG, timeout=2)
    except httpx.HTTPError:
        pytest.skip(f"no SearXNG instance at {SEARXNG} (see deploy/searxng/README.md)")
    tool = WebSearchTool(WebSearchSettings(searxng_url=SEARXNG, timeout=20))
    try:
        out = await tool.run({"query": "Python programming language"})
    finally:
        await tool.aclose()
    assert not out.is_error, out.content
    assert out.content.startswith("Web search results for")
    assert "1. " in out.content and "https://" in out.content
    news = WebSearchTool(WebSearchSettings(searxng_url=SEARXNG, timeout=20))
    try:
        out = await news.run({"query": "Formula 1", "recent": True})
    finally:
        await news.aclose()
    assert not out.is_error, out.content
    assert "Published 20" in out.content


async def test_live_weather() -> None:
    tool = WeatherTool()
    try:
        out = await tool.run({"place": "Porto, Portugal", "days": 2})
    finally:
        await tool.aclose()
    assert not out.is_error, out.content
    assert out.content.startswith("Weather from Open-Meteo for Porto, Porto District, Portugal")
    assert "\nNow: " in out.content and "\nToday (" in out.content and "\nTomorrow (" in out.content


async def test_live_fetch_page() -> None:
    tool = FetchPageTool()
    try:
        out = await tool.run({"url": "https://example.com/"})
        refused = await tool.run({"url": "http://localhost:11434/api/tags"})
    finally:
        await tool.aclose()
    assert not out.is_error, out.content
    assert out.content.startswith("Web page: Example Domain")
    assert refused.is_error and "local name" in refused.content
