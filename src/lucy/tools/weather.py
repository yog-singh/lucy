"""The `weather` tool: current conditions and a short daily forecast from Open-Meteo.

Two requests, no API key: Open-Meteo's geocoding API turns the place name into
coordinates (it ranks matches mostly by population and understands "Cambridge,
Massachusetts"), then its forecast API returns current conditions and daily values
in the place's own time zone. Only the place name and the coordinates, rounded to
two decimals (about a kilometre), leave the machine, with the generic User-Agent.

Open-Meteo's free API is for non-commercial use (fewer than 10,000 calls a day),
and its data is licensed CC BY 4.0, so the result names Open-Meteo as its source.
"""

from __future__ import annotations

import asyncio
import unicodedata
from datetime import date, datetime
from typing import Any, Final

import httpx

from lucy.core.config import WeatherSettings
from lucy.core.types import Risk, ToolOutput
from lucy.tools.common import describe_http_error, error, make_client, required_text

GEOCODING_URL: Final = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL: Final = "https://api.open-meteo.com/v1/forecast"
SERVICE: Final = "The weather service (Open-Meteo)"
DEFAULT_DAYS: Final = 3
"""Today and the next two days."""
MAX_DAYS: Final = 7
MAX_OTHERS: Final = 4
"""Other places with the same name listed after the best match."""
MIN_OTHER_POPULATION: Final = 10_000

CURRENT: Final = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
    "weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
)
DAILY: Final = (
    "weather_code,temperature_2m_max,temperature_2m_min,"
    "precipitation_probability_max,precipitation_sum"
)

WEATHER_CODES: Final[dict[int, str]] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with light hail",
    97: "heavy thunderstorms",
    99: "thunderstorms with heavy hail",
}
"""The WMO weather codes Open-Meteo documents, in plain words."""

_UNITS: Final = {
    "metric": {"temperature": "°C", "wind": "km/h", "rain": "mm", "params": {}},
    "imperial": {
        "temperature": "°F",
        "wind": "mph",
        "rain": "in",
        "params": {
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
        },
    },
}
_COUNTRY_NAMES: Final = {
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "greatbritain": "United Kingdom",
    "usa": "United States",
    "us": "United States",
    "uae": "United Arab Emirates",
}
"""Country nicknames Open-Meteo's geocoder does not know, by `_key`."""
_COMPASS: Final = (
    "north",
    "north-east",
    "east",
    "south-east",
    "south",
    "south-west",
    "west",
    "north-west",
)


def describe_code(code: object) -> str:
    """The weather code in words; an unknown or missing code says so."""
    if isinstance(code, int | float) and not isinstance(code, bool) and code == int(code):
        return WEATHER_CODES.get(int(code), "unknown conditions")
    return "unknown conditions"


class WeatherTool:
    """Weather now and for the next days in a named place (risk tier read)."""

    name = "weather"
    description = (
        "Current weather and a daily forecast for a city or town, from Open-Meteo. Pass "
        "the place as the user would say it, adding the region or country when the name "
        "is common, for example 'Porto' or 'Cambridge, Massachusetts'. It cannot tell where "
        "the user is: for 'here', pass the place you know the user is in, or ask. Use "
        "'days' for longer forecasts (default 3: today and the next two days; at most 7)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "place": {
                "type": "string",
                "description": "A city or town, optionally with region or country.",
            },
            "days": {
                "type": "integer",
                "description": "Days of forecast, counting today (1 to 7, default 3).",
            },
        },
        "required": ["place"],
    }
    risk = Risk.READ

    def __init__(
        self,
        settings: WeatherSettings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings or WeatherSettings()
        self._client = make_client(self.settings.timeout, transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        place = required_text(args, "place")
        if place is None:
            return error("The weather tool needs a 'place': a city or town.")
        days = _days(args.get("days"))
        try:
            async with asyncio.timeout(self.settings.timeout):
                matches = await self._find(place)
                if not matches:
                    return error(
                        f"No place called {place!r} was found. Try the nearest town or city, "
                        "or add the region or country."
                    )
                best, others = _choose(place, matches)
                forecast = await self._forecast(best, days)
        except httpx.HTTPStatusError as exc:
            return error(_api_error(exc))
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, KeyError) as exc:
            return error(describe_http_error(exc, SERVICE))
        try:
            return ToolOutput(content=self._format(best, others, forecast))
        except (TypeError, KeyError, ValueError, IndexError) as exc:
            return error(describe_http_error(exc, SERVICE))

    async def _find(self, place: str) -> list[dict[str, Any]]:
        """Matches for `place`, best first. Open-Meteo understands "Cambridge,
        Massachusetts" and "Austin, TX" but not "London, UK" or "Springfield Illinois",
        so those are retried: a known country nickname is spelled out, a missing comma
        added, and as a last resort the name alone is searched and the matches kept
        whose region or country is the one given."""
        parts = [p.strip() for p in place.split(",") if p.strip()]
        name, qualifiers = parts[0], [_COUNTRY_NAMES.get(_key(q), q) for q in parts[1:]]
        query = ", ".join([name, *qualifiers])
        matches = await self._geocode(query)
        if matches or (not qualifiers and " " not in name):
            return matches
        if not qualifiers:  # "Springfield Illinois": the last word is the region
            name, last = name.rsplit(" ", 1)
            qualifiers = [_COUNTRY_NAMES.get(_key(last), last)]
            matches = await self._geocode(f"{name}, {qualifiers[0]}")
            if matches:
                return matches
        wanted = {_key(q) for q in qualifiers}
        return [
            m
            for m in await self._geocode(name)
            if wanted
            <= {_key(str(m.get(k) or "")) for k in ("admin1", "admin2", "country", "country_code")}
        ]

    async def _geocode(self, place: str) -> list[dict[str, Any]]:
        params = {"name": place, "count": "10", "language": "en", "format": "json"}
        response = await self._client.get(GEOCODING_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("geocoding answer is not an object")
        results = data.get("results") or []
        if not isinstance(results, list):
            raise TypeError("geocoding results are not a list")
        return [
            r
            for r in results
            if isinstance(r, dict)
            and isinstance(r.get("latitude"), int | float)
            and isinstance(r.get("longitude"), int | float)
            and r.get("name")
        ]

    async def _forecast(self, place: dict[str, Any], days: int) -> dict[str, Any]:
        units = _UNITS[self.settings.units]
        params: dict[str, str] = {
            "latitude": f"{place['latitude']:.2f}",
            "longitude": f"{place['longitude']:.2f}",
            "current": CURRENT,
            "daily": DAILY,
            "timezone": place.get("timezone") or "auto",
            "forecast_days": str(days),
            **units["params"],  # type: ignore[dict-item]
        }
        response = await self._client.get(FORECAST_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("current"), dict):
            raise TypeError("forecast answer has no current conditions")
        return data

    def _format(
        self, place: dict[str, Any], others: list[dict[str, Any]], data: dict[str, Any]
    ) -> str:
        units = _UNITS[self.settings.units]
        deg, wind, rain = units["temperature"], units["wind"], units["rain"]
        current = data["current"]
        observed = datetime.fromisoformat(str(current["time"]))
        lines = [
            f"Weather from Open-Meteo for {_place_name(place)}, observed at "
            f"{observed:%H:%M} local time on {observed:%A} {observed.day} {observed:%B}."
        ]
        now = [
            f"Now: {_whole(current.get('temperature_2m'))}{deg}",
            f"feels like {_whole(current.get('apparent_temperature'))}{deg}",
            describe_code(current.get("weather_code")),
        ]
        speed, gusts = current.get("wind_speed_10m"), current.get("wind_gusts_10m")
        if isinstance(speed, int | float):
            text = f"wind {_whole(speed)} {wind}"
            direction = current.get("wind_direction_10m")
            if isinstance(direction, int | float) and speed >= 1:
                text += f" from the {_COMPASS[round(direction / 45) % 8]}"
            if isinstance(gusts, int | float) and gusts >= speed + 10:
                text += f", gusts {_whole(gusts)} {wind}"
            now.append(text)
        humidity = current.get("relative_humidity_2m")
        if isinstance(humidity, int | float):
            now.append(f"humidity {_whole(humidity)}%")
        wet = current.get("precipitation")
        if isinstance(wet, int | float):
            now.append(f"recent precipitation {wet:g} {rain}" if wet else "no precipitation")
        lines.append(", ".join(now) + ".")

        daily = data.get("daily") if isinstance(data.get("daily"), dict) else {}
        dates = daily.get("time") or []
        for i, day in enumerate(dates):
            when = date.fromisoformat(str(day))
            label = _day_label(when, observed.date())
            high = _at(daily, "temperature_2m_max", i)
            low = _at(daily, "temperature_2m_min", i)
            parts = [
                f"{label}: high {_whole(high)}{deg}, low {_whole(low)}{deg}",
                describe_code(_at(daily, "weather_code", i)),
            ]
            chance = _at(daily, "precipitation_probability_max", i)
            if isinstance(chance, int | float):
                parts.append(f"chance of rain {_whole(chance)}%")
            total = _at(daily, "precipitation_sum", i)
            if isinstance(total, int | float) and total > 0:
                parts.append(f"{total:g} {rain} expected")
            lines.append(", ".join(parts) + ".")
        if others:
            names = "; ".join(_place_name(o) for o in others)
            lines.append(f"Other places with this name: {names}.")
        return "\n".join(lines)


def _days(value: object) -> int:
    """The number of forecast days asked for, within 1 to MAX_DAYS."""
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int | float):
        return DEFAULT_DAYS
    return max(1, min(MAX_DAYS, int(value)))


def _key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if c.isalnum()).lower()


def _choose(
    place: str, matches: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The best match (the first whose name is the one asked for, else the first) and
    other notable places with the same name, in Open-Meteo's order: those with at least
    MIN_OTHER_POPULATION people and 5% of the best match's population (so Porto in
    Portugal does not list hamlets, but Cambridge lists Cambridge, Massachusetts)."""
    wanted = _key(place.split(",")[0])
    named = [m for m in matches if _key(str(m["name"])) == wanted]
    best = named[0] if named else matches[0]
    floor = max(MIN_OTHER_POPULATION, 0.05 * _population(best))
    seen = {_place_name(best)}
    others = []
    for match in matches:
        name = _place_name(match)
        if (
            match is not best
            and _key(str(match["name"])) == _key(str(best["name"]))
            and _population(match) >= floor
            and name not in seen
        ):
            seen.add(name)
            others.append(match)
    return best, others[:MAX_OTHERS]


def _population(place: dict[str, Any]) -> float:
    value = place.get("population")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _place_name(place: dict[str, Any]) -> str:
    parts = [str(place["name"])]
    for key in ("admin1", "country"):
        value = place.get(key)
        if isinstance(value, str) and value and value not in parts:
            parts.append(value)
    return ", ".join(parts)


def _whole(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "?"
    return str(round(value))


def _at(daily: dict[str, Any], key: str, index: int) -> object:
    values = daily.get(key)
    return values[index] if isinstance(values, list) and index < len(values) else None


def _day_label(day: date, today: date) -> str:
    name = f"{day:%A} {day.day} {day:%B}"
    offset = (day - today).days
    if offset == 0:
        return f"Today ({name})"
    if offset == 1:
        return f"Tomorrow ({name})"
    return name


def _api_error(exc: httpx.HTTPStatusError) -> str:
    """Open-Meteo explains a rejected request as {"error": true, "reason": "..."}."""
    try:
        reason = exc.response.json().get("reason")
    except (ValueError, AttributeError):
        reason = None
    if isinstance(reason, str) and reason.strip():
        return f"{SERVICE} rejected the request: {reason.strip()[:200]}"
    return describe_http_error(exc, SERVICE)
