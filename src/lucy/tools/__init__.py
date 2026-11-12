"""Tools the chat model can call, and the catalogue that builds the enabled ones.

Many tools are active at once, so tools do not go through the one-adapter-per-
port registry: `CATALOGUE` maps each tool name to a factory taking the settings
(and, for tests, an HTTP transport). `build_tools` builds the tools listed in
`[tools] enabled`, in that order. Tools that hold an HTTP client have an
`aclose()` method; the app calls it on stop. The `time` tool needs no network:
it reads the system's time-zone database.

Every name in the catalogue has an options table in `ToolsSettings`
(`lucy.core.config`), which is where unknown names and missing options are
rejected when the settings are loaded.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

import httpx

from lucy.core.config import Settings
from lucy.core.ports import Tool
from lucy.tools.fetch_page import FetchPageTool
from lucy.tools.local_time import TimeTool
from lucy.tools.netguard import Resolver, system_resolver
from lucy.tools.weather import WeatherTool
from lucy.tools.web_search import WebSearchTool
from lucy.tools.wikipedia import WikipediaTool

Clock = Callable[[], datetime]
Factory = Callable[[Settings, httpx.AsyncBaseTransport | None, Resolver, Clock], Tool]

CATALOGUE: Final[dict[str, Factory]] = {
    "wikipedia": lambda s, transport, _resolver, _clock: WikipediaTool(
        s.tools.wikipedia, transport=transport
    ),
    "web_search": lambda s, transport, _resolver, _clock: WebSearchTool(
        s.tools.web_search, transport=transport
    ),
    "fetch_page": lambda s, transport, resolver, _clock: FetchPageTool(
        s.tools.fetch_page, transport=transport, resolver=resolver
    ),
    "time": lambda _s, _transport, _resolver, clock: TimeTool(clock=clock),
    "weather": lambda s, transport, _resolver, _clock: WeatherTool(
        s.tools.weather, transport=transport
    ),
}


def build_tools(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    resolver: Resolver = system_resolver,
    clock: Clock = lambda: datetime.now(UTC),
) -> list[Tool]:
    """The tools enabled in `settings`, in the order listed. Empty when none are.

    `transport` and `resolver` replace the network for tests. `clock` is "now" for
    the `time` tool (the app passes its own clock).
    """
    tools: list[Tool] = []
    for name in settings.tools.enabled:
        factory = CATALOGUE.get(name)
        if factory is None:
            raise ValueError(f"unknown tool {name!r}; known tools: {', '.join(CATALOGUE)}")
        tools.append(factory(settings, transport, resolver, clock))
    return tools
