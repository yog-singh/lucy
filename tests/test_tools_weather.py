"""The `weather` tool against recorded Open-Meteo responses.

The JSON bodies in tests/data/weather_*.json were captured once from the live
geocoding and forecast APIs (public places only); the geocoding lists are cut to
six matches.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from lucy.core.config import LLMSettings, Settings, ToolsSettings, WeatherSettings
from lucy.core.types import Risk
from lucy.tools import build_tools
from lucy.tools.common import USER_AGENT
from lucy.tools.weather import (
    FORECAST_URL,
    GEOCODING_URL,
    MAX_DAYS,
    WEATHER_CODES,
    WeatherTool,
    describe_code,
)

DATA = Path(__file__).parent / "data"


def recorded(name: str) -> Any:
    return json.loads((DATA / name).read_text())


PORTO = recorded("weather_geocode_porto.json")
CAMBRIDGE = recorded("weather_geocode_cambridge.json")
NONE = recorded("weather_geocode_none.json")
FORECAST = recorded("weather_forecast_porto.json")

Handler = Callable[[httpx.Request], httpx.Response]


def service(
    geocode: Any = PORTO,
    forecast: Any = FORECAST,
    seen: list[httpx.Request] | None = None,
    *,
    geocode_status: int = 200,
    forecast_status: int = 200,
) -> httpx.MockTransport:
    """Open-Meteo's two endpoints; `geocode` may be a list of bodies, one per call."""
    bodies = geocode if isinstance(geocode, list) else None

    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        url = str(request.url).split("?")[0]
        if url == GEOCODING_URL:
            calls = [r for r in (seen or []) if str(r.url).startswith(GEOCODING_URL)]
            body = bodies[len(calls) - 1] if bodies is not None else geocode
            return httpx.Response(geocode_status, json=body)
        assert url == FORECAST_URL
        return httpx.Response(forecast_status, json=forecast)

    return httpx.MockTransport(handle)


async def ask(args: dict[str, Any], transport: httpx.AsyncBaseTransport, **settings: Any) -> Any:
    tool = WeatherTool(WeatherSettings(**settings), transport=transport)
    try:
        return await tool.run(args)
    finally:
        await tool.aclose()


async def test_current_conditions_and_daily_forecast() -> None:
    out = await ask({"place": "Porto"}, service())
    assert not out.is_error
    assert out.content == (
        "Weather from Open-Meteo for Porto, Porto District, Portugal, observed at 08:00 local "
        "time on Thursday 12 November.\n"
        "Now: 16°C, feels like 15°C, thunderstorms, wind 12 km/h from the south-west, gusts "
        "38 km/h, humidity 93%, recent precipitation 0.5 mm.\n"
        "Today (Thursday 12 November): high 17°C, low 16°C, thunderstorms, chance of rain "
        "100%, 42.6 mm expected.\n"
        "Tomorrow (Friday 13 November): high 17°C, low 15°C, thunderstorms, chance of rain "
        "100%, 31.5 mm expected.\n"
        "Saturday 14 November: high 16°C, low 14°C, thunderstorms, chance of rain 99%, "
        "36.5 mm expected."
    )
    assert len(out.content) < 1000


async def test_the_exact_requests_sent() -> None:
    seen: list[httpx.Request] = []
    await ask({"place": "Porto", "days": 2}, service(seen=seen))
    geocode, forecast = seen
    assert str(geocode.url).split("?")[0] == GEOCODING_URL
    assert dict(geocode.url.params) == {
        "name": "Porto",
        "count": "10",
        "language": "en",
        "format": "json",
    }
    assert str(forecast.url).split("?")[0] == FORECAST_URL
    params = dict(forecast.url.params)
    # Coordinates rounded to two decimals (Open-Meteo returned 41.1485, -8.61097).
    assert params["latitude"] == "41.15" and params["longitude"] == "-8.61"
    assert params["timezone"] == "Europe/Lisbon"
    assert params["forecast_days"] == "2"
    assert "temperature_unit" not in params
    assert set(params) == {
        "latitude",
        "longitude",
        "current",
        "daily",
        "timezone",
        "forecast_days",
    }
    for request in seen:
        assert request.method == "GET"
        assert request.headers["user-agent"] == USER_AGENT
        assert not {"cookie", "authorization", "referer", "from"} & set(request.headers)


async def test_imperial_units() -> None:
    seen: list[httpx.Request] = []
    imperial = copy.deepcopy(FORECAST)
    imperial["current"] |= {"temperature_2m": 60.1, "apparent_temperature": 59.5}
    imperial["current"] |= {"wind_speed_10m": 7.6, "wind_gusts_10m": 23.7, "precipitation": 0.02}
    out = await ask({"place": "Porto"}, service(forecast=imperial, seen=seen), units="imperial")
    params = dict(seen[1].url.params)
    assert params["temperature_unit"] == "fahrenheit"
    assert params["wind_speed_unit"] == "mph"
    assert params["precipitation_unit"] == "inch"
    assert "Now: 60°F, feels like 60°F, thunderstorms, wind 8 mph from the south-west, " in (
        out.content
    )
    assert "gusts 24 mph" in out.content and "recent precipitation 0.02 in" in out.content
    assert "high 17°F" in out.content and "42.6 in expected" in out.content


@pytest.mark.parametrize(
    ("days", "sent"),
    [(None, "3"), (1, "1"), (7, "7"), (30, str(MAX_DAYS)), (0, "1"), (-2, "1"), ("5", "5")]
    + [("lots", "3"), (True, "3"), (2.7, "2")],
)
async def test_days_are_bounded(days: object, sent: str) -> None:
    seen: list[httpx.Request] = []
    args = {"place": "Porto"} | ({} if days is None else {"days": days})
    await ask(args, service(seen=seen))
    assert seen[1].url.params["forecast_days"] == sent


async def test_ambiguous_name_lists_the_other_notable_places() -> None:
    out = await ask({"place": "Cambridge"}, service(geocode=CAMBRIDGE))
    assert out.content.startswith("Weather from Open-Meteo for Cambridge, England, United Kingdom")
    assert out.content.splitlines()[-1] == (
        "Other places with this name: Cambridge, Massachusetts, United States; "
        "Cambridge, Ontario, Canada; Cambridge, Maryland, United States; "
        "Cambridge, Ohio, United States."
    )
    # Porto's namesakes are villages: not listed.
    assert "Other places" not in (await ask({"place": "Porto"}, service())).content


async def test_the_match_named_as_asked_wins_over_a_bigger_prefix_match() -> None:
    body = copy.deepcopy(PORTO)
    body["results"].insert(0, body["results"][0] | {"name": "Porto Velho", "population": 9e5})
    out = await ask({"place": "porto, portugal"}, service(geocode=body))
    assert "for Porto, Porto District, Portugal" in out.content


async def test_a_region_without_a_comma_is_retried_with_one() -> None:
    seen: list[httpx.Request] = []
    transport = service(geocode=[NONE, CAMBRIDGE], seen=seen)
    out = await ask({"place": "Cambridge England"}, transport)
    assert not out.is_error
    names = [r.url.params["name"] for r in seen if str(r.url).startswith(GEOCODING_URL)]
    assert names == ["Cambridge England", "Cambridge, England"]


async def test_unknown_place() -> None:
    seen: list[httpx.Request] = []
    out = await ask({"place": "Qwxyzzy"}, service(geocode=NONE, seen=seen))
    assert out.is_error
    assert out.content == (
        "No place called 'Qwxyzzy' was found. Try the nearest town or city, or add the region "
        "or country."
    )
    assert len(seen) == 1  # one word: no retry, and no forecast request


@pytest.mark.parametrize("args", [{}, {"place": ""}, {"place": 5}])
async def test_a_missing_place_is_an_error(args: dict[str, Any]) -> None:
    out = await ask(args, service())
    assert out.is_error and "needs a 'place'" in out.content


async def test_forecast_failure_after_geocoding() -> None:
    reason = {"error": True, "reason": "Forecast days is invalid. Allowed range 0 to 16."}
    out = await ask({"place": "Porto"}, service(forecast=reason, forecast_status=400))
    assert out.is_error
    assert out.content == (
        "The weather service (Open-Meteo) rejected the request: Forecast days is invalid. "
        "Allowed range 0 to 16."
    )
    out = await ask({"place": "Porto"}, service(forecast="busy", forecast_status=503))
    assert out.is_error and "answered with an error (HTTP 503)" in out.content


async def test_geocoding_failure() -> None:
    out = await ask({"place": "Porto"}, service(geocode={}, geocode_status=500))
    assert out.is_error and "answered with an error (HTTP 500)" in out.content


@pytest.mark.parametrize(
    "exc", [httpx.ConnectTimeout("slow"), httpx.ConnectError("down"), TimeoutError()]
)
async def test_network_failures_are_errors(exc: Exception) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise exc

    out = await ask({"place": "Porto"}, httpx.MockTransport(fail))
    assert out.is_error
    assert "Open-Meteo" in out.content


async def test_the_whole_call_is_bounded_by_the_timeout() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json=PORTO)

    out = await ask({"place": "Porto"}, httpx.MockTransport(slow), timeout=0.05)
    assert out.is_error and "did not answer in time" in out.content


@pytest.mark.parametrize(
    ("geocode", "forecast"),
    [
        ("not json object", FORECAST),
        ({"results": "nope"}, FORECAST),
        ({"results": [{"name": "Porto"}]}, FORECAST),  # no coordinates: no match
        (PORTO, {"daily": {}}),
        (PORTO, {"current": {"time": "yesterday"}}),
        (PORTO, ["a", "list"]),
    ],
)
async def test_malformed_responses_are_errors(geocode: Any, forecast: Any) -> None:
    out = await ask({"place": "Porto"}, service(geocode=geocode, forecast=forecast))
    assert out.is_error


async def test_missing_values_degrade_gracefully() -> None:
    sparse = {"current": {"time": "2026-11-12T08:00", "weather_code": 1234}, "daily": {}}
    out = await ask({"place": "Porto"}, service(forecast=sparse))
    assert not out.is_error
    assert out.content.splitlines()[1] == "Now: ?°C, feels like ?°C, unknown conditions."


@pytest.mark.parametrize("code", sorted(WEATHER_CODES))
def test_every_weather_code_has_a_phrase(code: int) -> None:
    phrase = describe_code(code)
    assert phrase == WEATHER_CODES[code] and phrase and phrase == phrase.lower()
    assert describe_code(float(code)) == phrase


@pytest.mark.parametrize("code", [4, 100, -1, None, "61", True, 61.5])
def test_unknown_weather_codes(code: object) -> None:
    assert describe_code(code) == "unknown conditions"


def test_tool_shape_and_catalogue() -> None:
    settings = Settings(
        llm=LLMSettings(roles={"chat": {"adapter": "fake"}}),
        tools=ToolsSettings(enabled=["weather"], weather=WeatherSettings(units="imperial")),
    )
    (tool,) = build_tools(settings)
    assert isinstance(tool, WeatherTool) and tool.risk is Risk.READ
    assert tool.settings.units == "imperial"
    assert tool.parameters["required"] == ["place"]
    assert set(tool.parameters["properties"]) == {"place", "days"}


def geocode_names(seen: list[httpx.Request]) -> list[str]:
    return [r.url.params["name"] for r in seen if str(r.url).startswith(GEOCODING_URL)]


async def test_country_nicknames_are_spelled_out() -> None:
    seen: list[httpx.Request] = []
    out = await ask({"place": "Cambridge, UK"}, service(geocode=CAMBRIDGE, seen=seen))
    assert not out.is_error
    assert geocode_names(seen) == ["Cambridge, United Kingdom"]


async def test_last_resort_filters_matches_of_the_name_by_region() -> None:
    seen: list[httpx.Request] = []
    out = await ask({"place": "Cambridge, Ontario"}, service(geocode=[NONE, CAMBRIDGE], seen=seen))
    assert out.content.startswith("Weather from Open-Meteo for Cambridge, Ontario, Canada")
    assert geocode_names(seen) == ["Cambridge, Ontario", "Cambridge"]

    seen.clear()
    transport = service(geocode=[NONE, NONE, CAMBRIDGE], seen=seen)
    out = await ask({"place": "Cambridge Maryland"}, transport)
    assert out.content.startswith("Weather from Open-Meteo for Cambridge, Maryland, United States")
    assert geocode_names(seen) == ["Cambridge Maryland", "Cambridge, Maryland", "Cambridge"]

    seen.clear()
    out = await ask({"place": "Cambridge, Narnia"}, service(geocode=[NONE, CAMBRIDGE], seen=seen))
    assert out.is_error and "No place called 'Cambridge, Narnia'" in out.content
