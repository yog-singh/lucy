"""The `wikipedia` and `web_search` tools against recorded responses.

The JSON bodies in tests/data/ were captured once from the live services
(en.wikipedia.org and a local SearXNG instance); the SearXNG ones are cut to
8 results, and the empty one had its results removed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from lucy.core.config import WebSearchSettings, WikipediaSettings
from lucy.tools.common import TRUNCATION_NOTE, USER_AGENT
from lucy.tools.web_search import SNIPPET_NOTE, WebSearchTool
from lucy.tools.wikipedia import WikipediaTool

DATA = Path(__file__).parent / "data"
SEARXNG = "http://127.0.0.1:8890"


def recorded(name: str) -> Any:
    return json.loads((DATA / name).read_text())


def transport(
    handler: Callable[[httpx.Request], httpx.Response], seen: list[httpx.Request] | None = None
) -> httpx.MockTransport:
    def record(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return handler(request)

    return httpx.MockTransport(record)


def json_reply(body: Any, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(status, json=body)


def failing(exc: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


# --- wikipedia ------------------------------------------------------------------


async def test_wikipedia_returns_best_article_and_other_matches() -> None:
    seen: list[httpx.Request] = []
    tool = WikipediaTool(
        transport=transport(json_reply(recorded("wikipedia_eiffel_tower.json")), seen)
    )
    out = await tool.run({"query": "Eiffel Tower"})
    await tool.aclose()

    assert not out.is_error
    lines = out.content.splitlines()
    assert lines[0] == "Article: Eiffel Tower"
    assert lines[1] == "Address: https://en.wikipedia.org/wiki/Eiffel_Tower"
    assert lines[2] == "Introduction:"
    assert "lattice tower on the Champ de Mars in Paris" in out.content
    assert lines[-1] == (
        "Other matching articles: Eiffel Tower replicas and derivatives; "
        "List of names on the Eiffel Tower; Gustave Eiffel"
    )
    assert "disambiguation" not in out.content
    (request,) = seen
    assert request.url.host == "en.wikipedia.org"
    assert request.url.params["gsrsearch"] == "Eiffel Tower"
    assert request.url.params["explaintext"] == "1"
    assert request.headers["user-agent"] == USER_AGENT


async def test_wikipedia_disambiguation_is_flagged_and_closest_article_shown() -> None:
    tool = WikipediaTool(transport=transport(json_reply(recorded("wikipedia_mercury.json"))))
    out = await tool.run({"query": "Mercury"})
    assert not out.is_error
    assert out.content.startswith("Note: 'Mercury' has several meanings on Wikipedia.")
    assert "Mercury (planet), the closest planet to the Sun" in out.content
    assert "Closest article (it may not be the one meant):\nArticle: Freddie Mercury" in out.content
    assert out.content.splitlines()[-1] == (
        "Other matching articles: Mercury (planet); Mercury (element); Mercury Records"
    )


async def test_wikipedia_only_disambiguation_pages() -> None:
    body = recorded("wikipedia_mercury.json")
    body["query"]["pages"] = [p for p in body["query"]["pages"] if p["title"] == "Mercury"]
    tool = WikipediaTool(transport=transport(json_reply(body)))
    out = await tool.run({"query": "Mercury"})
    assert not out.is_error
    assert "several meanings" in out.content
    assert "ask again with a more specific name" in out.content


async def test_wikipedia_no_match_is_an_error_result() -> None:
    tool = WikipediaTool(transport=transport(json_reply(recorded("wikipedia_no_match.json"))))
    out = await tool.run({"query": "qwxzvbnmplkjh"})
    assert out.is_error
    assert out.content == "Wikipedia has no article matching 'qwxzvbnmplkjh'."


async def test_wikipedia_language_selects_the_edition() -> None:
    seen: list[httpx.Request] = []
    tool = WikipediaTool(
        WikipediaSettings(language="de"),
        transport=transport(json_reply(recorded("wikipedia_no_match.json")), seen),
    )
    await tool.run({"query": "Eiffelturm"})
    assert seen[0].url.host == "de.wikipedia.org"


async def test_wikipedia_result_is_capped() -> None:
    tool = WikipediaTool(
        WikipediaSettings(max_chars=800),
        transport=transport(json_reply(recorded("wikipedia_eiffel_tower.json"))),
    )
    out = await tool.run({"query": "Eiffel Tower"})
    assert len(out.content) <= 800
    assert TRUNCATION_NOTE in out.content
    assert out.content.startswith("Article: Eiffel Tower")
    assert out.content.splitlines()[-1].startswith("Other matching articles:")


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (json_reply({"x": 1}, status=503), "Wikipedia answered with an error (HTTP 503)."),
        (failing(httpx.ConnectError("no route")), "Wikipedia is not reachable."),
        (failing(httpx.ReadTimeout("slow")), "Wikipedia did not answer in time."),
        (lambda r: httpx.Response(200, text="<html>"), "Wikipedia returned something unexpected"),
        (
            json_reply({"error": {"code": "x", "info": "Search request is longer than allowed"}}),
            "Wikipedia could not run the search: Search request is longer than allowed",
        ),
    ],
)
async def test_wikipedia_failures_are_error_results(
    handler: Callable[[httpx.Request], httpx.Response], message: str
) -> None:
    tool = WikipediaTool(transport=transport(handler))
    out = await tool.run({"query": "anything"})
    assert out.is_error
    assert out.content.startswith(message)


async def test_wikipedia_needs_a_query() -> None:
    tool = WikipediaTool(transport=transport(failing(AssertionError("no request expected"))))
    for args in ({}, {"query": "  "}, {"query": 3}):
        out = await tool.run(args)
        assert out.is_error and "needs a 'query'" in out.content


# --- web_search -----------------------------------------------------------------


def search_tool(
    handler: Callable[[httpx.Request], httpx.Response], **options: Any
) -> WebSearchTool:
    settings = WebSearchSettings(searxng_url=SEARXNG + "/", **options)
    return WebSearchTool(settings, transport=transport(handler))


async def test_web_search_lists_top_results() -> None:
    seen: list[httpx.Request] = []
    settings = WebSearchSettings(searxng_url=SEARXNG)
    tool = WebSearchTool(
        settings, transport=transport(json_reply(recorded("searxng_iceland.json")), seen)
    )
    out = await tool.run({"query": "population of iceland"})
    await tool.aclose()

    assert not out.is_error
    lines = out.content.splitlines()
    assert lines[0] == "Web search results for 'population of iceland':"
    assert lines[1:3] == ["1. Iceland - Wikipedia", "   https://en.wikipedia.org/wiki/Iceland"]
    assert lines[3].startswith("   Iceland is a Nordic island country")
    assert sum(line[:2] in {f"{n}." for n in range(1, 10)} for line in lines) == 5
    assert "Iceland Population (1950-2026) - Macrotrends" in out.content
    (request,) = seen
    assert str(request.url).startswith("http://127.0.0.1:8890/search?")
    assert request.url.params["format"] == "json"
    assert request.url.params["q"] == "population of iceland"
    assert request.headers["user-agent"] == USER_AGENT


async def test_web_search_respects_max_results_and_snippet_length() -> None:
    body = recorded("searxng_iceland.json")
    body["results"][0]["content"] = "word " * 500
    out = await search_tool(json_reply(body), max_results=2).run({"query": "q"})
    assert "2. " in out.content and "3. " not in out.content
    snippet = out.content.splitlines()[3]
    assert len(snippet) <= 304 and snippet.endswith("…")


async def test_web_search_includes_direct_answers() -> None:
    body = recorded("searxng_iceland.json")
    body["answers"] = [{"answer": "About 390,000 people.", "url": None}, "plain answer"]
    out = await search_tool(json_reply(body)).run({"query": "q"})
    assert "Direct answer: About 390,000 people.\nDirect answer: plain answer" in out.content


async def test_web_search_shows_published_dates() -> None:
    body = recorded("searxng_iceland.json")
    body["results"][0]["publishedDate"] = "2026-08-19T18:05:44"
    body["results"][1]["publishedDate"] = "not a date"
    body["results"][2]["publishedDate"] = "2025-01-02"
    body["results"][2]["content"] = ""
    lines = (await search_tool(json_reply(body)).run({"query": "q"})).content.splitlines()
    assert lines[3].startswith("   Published 2026-08-19. Iceland is a Nordic island country")
    assert "Published" not in lines[6]
    assert lines[9] == "   Published 2025-01-02."


async def search(body: Any, **options: Any) -> str:
    out = await search_tool(json_reply(body), **options).run({"query": "q"})
    assert not out.is_error
    return out.content


def result(title: str, score: Any = 1.0, content: str = "", **fields: Any) -> dict[str, Any]:
    slug = title.lower().replace(" ", "-")
    return {"title": title, "url": f"https://example.org/{slug}", "content": content} | {
        "score": score,
        **fields,
    }


async def test_web_search_ranks_results_by_score() -> None:
    # Recorded: SearXNG listed this engine's results as positions 1, 9, 15, 17, 18, 2, 3, ...
    body = recorded("searxng_current_ceo_apple.json")
    assert [r["positions"][0] for r in body["results"][:6]] == [1, 9, 15, 17, 18, 2]
    lines = (await search(body)).splitlines()
    titles = [line for line in lines if line[:2] in {"1.", "2.", "3.", "4.", "5."}]
    assert titles == [
        "1. Tim Cook - Wikipedia",
        "2. Tim Cook - Apple Leadership",
        "3. Tim Cook to become Apple Executive Chairman John Ternus to ...",
        "4. Apple Leadership",
        "5. Tim Cook | Biography, Apple CEO, & Business Executive - Britannica",
    ]
    # The engine's date prefix ("20 Apr 2026 ...") becomes the published date.
    assert lines[lines.index(titles[2]) + 2] == (
        "   Published 2026-04-20. John Ternus to become Apple CEO ... Tim Cook and John "
        "Ternus at Apple Park. CUPERTINO, CALIFORNIA ..."
    )


async def test_web_search_score_ties_and_bad_scores_keep_their_order() -> None:
    body = {
        "results": [
            result("A", score="high"),
            result("B", score=None),
            result("C", score=0.5),
            result("D", score=True),
            result("E"),
        ]
    }
    lines = (await search(body)).splitlines()
    assert [line[3:] for line in lines if line[:1].isdigit()] == ["E", "C", "A", "B", "D"]


async def test_web_search_shows_the_infobox_before_the_results() -> None:
    lines = (await search(recorded("searxng_tim_cook.json"))).splitlines()
    assert lines[1:3] == [
        "Summary from Wikidata and Wikipedia: Tim Cook",
        "   https://en.wikipedia.org/wiki/Tim_Cook",
    ]
    assert lines[3].startswith(
        "   Timothy Donald Cook is an American business executive who served as the chief "
        "executive officer (CEO) of Apple from August 24, 2011 to August 31, 2026"
    )
    assert len(lines[3]) <= 3 + 600 and lines[3].endswith("…")
    assert lines[4:6] == [
        "   Facts: Date of birth: Tuesday, November 1, 1960; Country of citizenship: "
        "United States.",
        "   Official website: https://www.apple.com/leadership/tim-cook/",
    ]
    assert lines[6] == "1. Tim Cook - Wikipedia"
    # Only the article and the official site are linked.
    assert "twitter.com" not in "\n".join(lines[:6]) and "wikidata.org" not in "\n".join(lines)


async def test_web_search_infobox_facts_and_short_description() -> None:
    text = await search(recorded("searxng_python.json"))
    assert text.splitlines()[1:4] == [
        "Summary from Wikidata: Python",
        "   https://en.wikipedia.org/wiki/Python_(programming_language)",
        "   general-purpose programming language",
    ]
    assert "Software version identifier: 3.15.2." in text
    assert "Official website: https://www.python.org/" in text


async def test_web_search_drops_infobox_facts_with_unresolved_ids() -> None:
    # Recorded: the chief executive came back as a bare Wikidata id, and one founder too.
    body = recorded("searxng_apple_inc.json")
    values = [a["value"] for a in body["infoboxes"][0]["attributes"]]
    assert "Q106028933" in values and "Q332591, Steve Wozniak, Steve Jobs" in values
    text = await search(body)
    assert "Q106028933" not in text and "Q332591" not in text
    assert "Chief executive officer" not in text and "Founder" not in text
    assert "Headquarters location: Cupertino; Founder" not in text
    assert "Headquarters location: Cupertino; Legal form: corporation." in text


async def test_web_search_infoboxes_are_bounded() -> None:
    box = {
        "infobox": "Big",
        "id": "https://en.wikipedia.org/wiki/Big",
        "content": "word " * 400,
        "attributes": [{"label": f"L{n}", "value": "v " * 100} for n in range(20)],
        "engines": ["wikidata"],
    }
    text = await search({"results": [], "infoboxes": [box, box | {"infobox": "Two"}, box]})
    assert text.count("Summary from Wikidata:") == 2
    facts = next(line for line in text.splitlines() if line.startswith("   Facts:"))
    assert facts.count("; ") == 7 and "L7:" in facts and "L8:" not in facts
    assert all(len(part.removeprefix("   Facts: ")) <= 105 for part in facts.split("; "))
    assert max(len(line) for line in text.splitlines() if "word" in line) <= 603


@pytest.mark.parametrize(
    "infoboxes",
    [
        "not a list",
        [None, 3, "text", {}],
        [{"infobox": "Empty"}],  # nothing to say
        [{"infobox": "X", "content": 5, "attributes": [{"label": "L", "value": ""}]}],
    ],
)
async def test_web_search_ignores_unusable_infoboxes(infoboxes: Any) -> None:
    body = recorded("searxng_iceland.json") | {"infoboxes": infoboxes}
    assert (await search(body)).splitlines()[1] == "1. Iceland - Wikipedia"


async def test_web_search_tolerates_malformed_infobox_fields() -> None:
    box = {
        "infobox": None,
        "id": "not an address",
        "content": "  A short   description. ",
        "attributes": [
            None,
            {"label": 1, "value": "x"},
            {"label": " ", "value": "x"},
            {"label": "Good", "value": "yes"},
            {"label": "Id", "value": "P31"},
        ],
        "urls": [
            None,
            {"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/X"},
            {"title": "Official website", "url": "javascript:alert(1)", "official": True},
        ],
        "engines": [None, "wikipedia", 7],
    }
    text = await search({"results": [], "infoboxes": [box]})
    assert text.splitlines()[1:] == [
        "Summary from Wikipedia: untitled",
        "   https://en.wikipedia.org/wiki/X",
        "   A short description.",
        "   Facts: Good: yes.",
    ]
    box = {"infobox": "Y", "content": "Text.", "urls": "nope", "engines": "nope"}
    assert (await search({"results": [], "infoboxes": [box]})).splitlines()[1:] == [
        "Summary from the web: Y",
        "   Text.",
    ]


async def test_web_search_reports_spelling_corrections() -> None:
    body = recorded("searxng_iceland.json") | {"corrections": ["population of iceland", 3, ""]}
    assert (await search(body)).splitlines()[1] == "Did you mean: population of iceland?"
    body = recorded("searxng_iceland.json") | {"corrections": "nope", "answers": "nope"}
    assert (await search(body)).splitlines()[1] == "1. Iceland - Wikipedia"


@pytest.mark.parametrize(
    ("content", "published", "snippet"),
    [
        ("9 Sept 2022 ... Since then.", None, "Published 2022-09-09. Since then."),
        ("27 May 2026 ... The main branch.", None, "Published 2026-05-27. The main branch."),
        ("3 days ago ... 2026 Season.", None, "Published 3 days ago. 2026 Season."),
        ("1 day ago · Text.", None, "Published 1 day ago. Text."),
        ("Apr 6, 2026 · Text.", None, "Published 2026-04-06. Text."),  # Yahoo
        ("Sept. 30, 2025 ... Text.", None, "Published 2025-09-30. Text."),
        ("Feb 30, 2026 · Text.", None, "Feb 30, 2026 · Text."),
        ("31 Feb 2026 ... Text.", None, "31 Feb 2026 ... Text."),  # not a date: left alone
        ("9 Smarch 2022 ... Text.", None, "9 Smarch 2022 ... Text."),
        ("Posted 2022. Text.", None, "Posted 2022. Text."),
        ("9 Sept 2022 ... Text.", "2022-09-10T08:00:00", "Published 2022-09-10. Text."),
        ("9 Sept 2022 ...", None, "Published 2022-09-09."),
    ],
)
async def test_web_search_reads_dates_at_the_start_of_snippets(
    content: str, published: str | None, snippet: str
) -> None:
    body = {"results": [result("A", content=content, publishedDate=published)]}
    assert (await search(body)).splitlines()[3] == f"   {snippet}"


async def test_web_search_results_end_with_the_note_about_snippets() -> None:
    assert (await search(recorded("searxng_iceland.json"))).endswith(f"\n{SNIPPET_NOTE}")
    # Without a result list (an infobox or direct answer only) there are no snippets.
    body = {"results": [], "answers": ["42"]}
    assert (await search(body)).splitlines() == ["Web search results for 'q':", "Direct answer: 42"]


async def test_web_search_infobox_alone_is_a_result() -> None:
    body = recorded("searxng_no_results.json") | {
        "infoboxes": recorded("searxng_tim_cook.json")["infoboxes"]
    }
    out = await search_tool(json_reply(body)).run({"query": "Tim Cook"})
    assert not out.is_error and "Summary from Wikidata and Wikipedia: Tim Cook" in out.content


async def test_web_search_cap_keeps_the_infobox() -> None:
    text = await search(recorded("searxng_tim_cook.json"), max_chars=1200)
    assert len(text) <= 1200 and text.endswith(f"{TRUNCATION_NOTE}\n{SNIPPET_NOTE}")
    assert "Facts: Date of birth" in text


async def test_web_search_no_results() -> None:
    out = await search_tool(json_reply(recorded("searxng_no_results.json"))).run({"query": "zz"})
    assert out.is_error
    assert out.content.startswith("No results for 'zz'; some search engines did not respond.")
    body = recorded("searxng_no_results.json") | {"unresponsive_engines": []}
    out = await search_tool(json_reply(body)).run({"query": "zz"})
    assert out.is_error and out.content == "No results for 'zz'."


async def test_web_search_result_is_capped() -> None:
    out = await search_tool(json_reply(recorded("searxng_iceland.json")), max_chars=400).run(
        {"query": "population of iceland"}
    )
    assert len(out.content) <= 400
    # The note about snippets survives the cap.
    assert out.content.endswith(f"{TRUNCATION_NOTE}\n{SNIPPET_NOTE}")


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (json_reply({}, status=403), "The search service refused the request (HTTP 403)"),
        (json_reply({}, status=500), "The search service answered with an error (HTTP 500)."),
        (failing(httpx.ConnectError("refused")), "The search service is not reachable."),
        (failing(httpx.ConnectTimeout("slow")), "The search service did not answer in time."),
        (lambda r: httpx.Response(200, text="not json"), "The search service returned something"),
        (json_reply([1, 2]), "The search service returned something unexpected"),
    ],
)
async def test_web_search_failures_are_error_results(
    handler: Callable[[httpx.Request], httpx.Response], message: str
) -> None:
    out = await search_tool(handler).run({"query": "q"})
    assert out.is_error
    assert out.content.startswith(message)


async def test_web_search_overall_timeout() -> None:
    import asyncio

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    settings = WebSearchSettings(searxng_url=SEARXNG, timeout=0.05)
    tool = WebSearchTool(settings, transport=httpx.MockTransport(slow))
    out = await tool.run({"query": "q"})
    assert out.is_error and "did not answer in time" in out.content


async def test_web_search_needs_a_query_and_an_address() -> None:
    out = await search_tool(failing(AssertionError())).run({"query": ""})
    assert out.is_error and "needs a 'query'" in out.content
    with pytest.raises(ValueError, match="searxng_url"):
        WebSearchTool(WebSearchSettings())


def replying(seen: list[httpx.Request], *bodies: Any) -> Callable[[httpx.Request], httpx.Response]:
    """A handler that records each request and answers with the next body."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=bodies[len(seen) - 1])

    return handler


async def test_web_search_recent_asks_for_news_and_falls_back_to_the_web() -> None:
    seen: list[httpx.Request] = []
    news = {"results": [result("Launch today", content="Text.", publishedDate="2026-11-11")]}
    out = await search_tool(replying(seen, news)).run({"query": "launch", "recent": True})
    assert out.content.startswith("News results for 'launch':\n1. Launch today")
    assert "Published 2026-11-11. Text." in out.content
    assert [r.url.params.get("categories") for r in seen] == ["news"]

    # No news results: the same query once more, without the category.
    seen.clear()
    tool = search_tool(replying(seen, {"results": []}, recorded("searxng_iceland.json")))
    out = await tool.run({"query": "q", "recent": True})
    assert out.content.startswith("Web search results for 'q':")
    assert [r.url.params.get("categories") for r in seen] == ["news", None]


@pytest.mark.parametrize("recent", [False, "yes", None])
async def test_web_search_without_recent_searches_everything(recent: object) -> None:
    seen: list[httpx.Request] = []
    tool = search_tool(replying(seen, recorded("searxng_iceland.json")))
    out = await tool.run({"query": "q", "recent": recent})
    assert out.content.startswith("Web search results for 'q':")
    assert "categories" not in seen[0].url.params
