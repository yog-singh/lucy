"""The `time` tool: places resolved from the system time-zone database, the clock
injected, daylight saving and odd offsets handled, and no network."""

from __future__ import annotations

from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lucy.core.config import LLMSettings, Settings, ToolsSettings
from lucy.core.types import Risk
from lucy.tools import build_tools
from lucy.tools.local_time import TimeTool, parse_clock, resolve_place

KOLKATA = ZoneInfo("Asia/Kolkata")
LONDON = ZoneInfo("Europe/London")
NOW = datetime(2026, 11, 12, 6, 12, tzinfo=UTC)  # 11:42 in Kolkata, a Thursday


def tool(now: datetime = NOW, local: ZoneInfo = KOLKATA) -> TimeTool:
    return TimeTool(clock=lambda: now, local_zone=local)


async def ask(place: str, *, at: str | None = None, **kwargs: object) -> str:
    args = {"place": place} | ({"at": at} if at is not None else {})
    result = await tool(**kwargs).run(args)  # type: ignore[arg-type]
    assert not result.is_error, result.content
    return result.content


async def test_time_in_a_city_states_zone_offset_and_difference() -> None:
    text = await ask("Tokyo")
    assert text == (
        "In Tokyo (time zone Asia/Tokyo, UTC+09:00) it is now Thursday 12 November 2026, "
        "15:12 (3:12 pm) JST. That is 3 hours 30 minutes ahead of the user's local time "
        "(Thursday 12 November 2026, 11:42 (11:42 am) IST)."
    )


def test_the_tool_is_read_tier_and_needs_a_place() -> None:
    t = tool()
    assert t.name == "time" and t.risk is Risk.READ
    assert t.parameters["required"] == ["place"]


@pytest.mark.parametrize(
    ("place", "zone"),
    [
        ("Asia/Tokyo", "Asia/Tokyo"),  # a zone name
        ("europe/paris", "Europe/Paris"),  # in any case
        ("New York", "America/New_York"),  # the city part of a zone name
        ("São Paulo", "America/Sao_Paulo"),  # accents ignored
        ("Tokyo, Japan", "Asia/Tokyo"),  # "city, country"
        ("Springfield, Illinois", "America/Chicago"),  # unknown city, known state
        ("India", "Asia/Kolkata"),  # a country with one zone (iso3166.tab, zone.tab)
        ("Nepal", "Asia/Kathmandu"),
        ("Germany", "Europe/Berlin"),  # two zones with the same clock
        ("UK", "Europe/London"),  # built-in names
        ("California", "America/Los_Angeles"),
        ("San Francisco", "America/Los_Angeles"),
        ("Mumbai", "Asia/Kolkata"),
        ("PST", "America/Los_Angeles"),
        ("eastern time", "America/New_York"),
        ("Istanbul", "Europe/Istanbul"),  # two zones share the name and agree
        ("UTC", "UTC"),
    ],
)
def test_places_resolve(place: str, zone: str) -> None:
    assert resolve_place(place, NOW).zone == zone


def test_a_country_with_several_clocks_lists_them() -> None:
    resolved = resolve_place("United States", NOW)
    assert resolved.zone is None
    assert resolved.problem.startswith("United States has several time zones: ")
    assert "America/New_York (UTC-05:00)" in resolved.problem
    assert "America/Los_Angeles (UTC-08:00)" in resolved.problem
    assert resolved.problem.endswith("Ask again with a city or region there.")
    assert resolve_place("USA", NOW).problem == resolved.problem


async def test_unknown_place_is_an_error_with_close_matches() -> None:
    result = await tool().run({"place": "Tokio"})
    assert result.is_error
    assert "No time zone known for 'Tokio'. Close matches: Tokyo." in result.content
    result = await tool().run({"place": "Qwxyz"})
    assert result.is_error and "Close matches" not in result.content
    assert "zone name such as 'America/Chicago'" in result.content


@pytest.mark.parametrize("args", [{}, {"place": "  "}, {"place": 3}])
async def test_a_missing_place_is_an_error(args: dict[str, object]) -> None:
    result = await tool().run(args)
    assert result.is_error and "needs a 'place'" in result.content


async def test_daylight_saving_on_both_sides_of_a_transition() -> None:
    # New York leaves daylight saving on 1 November 2026, London on 25 October.
    before = await ask("New York", now=datetime(2026, 10, 31, 12, tzinfo=UTC), local=LONDON)
    after = await ask("New York", now=datetime(2026, 11, 2, 12, tzinfo=UTC), local=LONDON)
    assert "(time zone America/New_York, UTC-04:00) it is now Saturday 31 October" in before
    assert "08:00 (8:00 am) EDT. That is 4 hours behind" in before
    assert "(time zone America/New_York, UTC-05:00) it is now Monday 2 November" in after
    assert "07:00 (7:00 am) EST. That is 5 hours behind" in after
    summer = await ask("New York", now=datetime(2026, 7, 1, 12, tzinfo=UTC), local=LONDON)
    assert "UTC-04:00" in summer and "13:00 (1:00 pm) BST" in summer


async def test_half_hour_and_45_minute_offsets() -> None:
    assert "11:57 (11:57 am). That is 15 minutes ahead of" in await ask("Kathmandu")
    # Adelaide is UTC+10:30 in the southern summer and UTC+09:30 in winter.
    assert "UTC+10:30) it is now Thursday 12 November 2026, 16:42 (4:42 pm) ACDT" in await ask(
        "Adelaide"
    )
    july = await ask("Adelaide", now=datetime(2026, 7, 1, 2, 30, tzinfo=UTC))
    assert "UTC+09:30) it is now Wednesday 1 July 2026, 12:00 (12:00 pm) ACST" in july
    assert "4 hours ahead of the user's local time" in july
    chatham = await ask("Pacific/Chatham")
    assert "UTC+13:45" in chatham and "8 hours 15 minutes ahead" in chatham


async def test_the_date_line() -> None:
    late = datetime(2026, 11, 12, 23, 30, tzinfo=UTC)
    auckland = await ask("Auckland", now=late, local=LONDON)
    assert "it is now Friday 13 November 2026, 12:30 (12:30 pm) NZDT" in auckland
    assert auckland.endswith("It is already the next day there.")
    honolulu = await ask("Honolulu")
    assert "it is now Wednesday 11 November 2026, 20:12 (8:12 pm) HST" in honolulu
    assert "15 hours 30 minutes behind" in honolulu
    assert honolulu.endswith("It is still the previous day there.")


async def test_same_zone_as_the_user() -> None:
    assert "That is the same as the user's local time" in await ask("India")


async def test_converting_a_clock_time_at_the_user_s_location() -> None:
    text = await ask("New York", at="9am")
    assert text == (
        "When it is Thursday 12 November 2026, 09:00 (9:00 am) IST at the user's location, "
        "it is Wednesday 11 November 2026, 22:30 (10:30 pm) EST in New York (time zone "
        "America/New_York, UTC-05:00). That is 10 hours 30 minutes behind the user's local "
        "time. It is still the previous day there."
    )
    assert "it is now" in await ask("Tokyo", at="now")
    result = await tool().run({"place": "Tokyo", "at": "teatime"})
    assert result.is_error and "Could not read 'teatime'" in result.content


async def test_converting_uses_the_user_s_offset_on_that_day() -> None:
    # 01:30 UTC on the morning London leaves daylight saving: 09:00 that day is GMT.
    now = datetime(2026, 10, 25, 0, 30, tzinfo=UTC)
    text = await ask("Tokyo", at="09:00", now=now, local=LONDON)
    assert "09:00 (9:00 am) GMT at the user's location, it is Sunday 25 October 2026, " in text
    assert "18:00 (6:00 pm) JST" in text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("9", time(9)),
        ("9am", time(9)),
        ("9 a.m.", time(9)),
        ("12am", time(0)),
        ("12 pm", time(12)),
        ("9:30 pm", time(21, 30)),
        ("21:05", time(21, 5)),
        ("7.15", time(7, 15)),
        ("noon", time(12)),
        ("midnight", time(0)),
        ("13pm", None),
        ("24:00", None),
        ("9:75", None),
        ("soon", None),
    ],
)
def test_parse_clock(text: str, expected: time | None) -> None:
    assert parse_clock(text) == expected


async def test_default_clock_and_zone_are_the_system_ones() -> None:
    result = await TimeTool().run({"place": "UTC"})
    assert not result.is_error and "(time zone UTC, UTC+00:00)" in result.content


def test_build_tools_passes_the_clock() -> None:
    settings = Settings(
        llm=LLMSettings(roles={"chat": {"adapter": "fake"}}),
        tools=ToolsSettings(enabled=["time"]),
    )
    (built,) = build_tools(settings, clock=lambda: NOW)
    assert isinstance(built, TimeTool) and built._clock() == NOW


def test_the_module_makes_no_network_requests() -> None:
    source = (Path(__file__).parent.parent / "src/lucy/tools/local_time.py").read_text()
    assert "httpx" not in source and "socket" not in source and "urllib" not in source
