"""The `time` tool: the current date and time in a named place, from the system's
time-zone database. It makes no network request.

The chat model gets clock arithmetic across time zones wrong with thinking off
(measured: "what time is it in Tokyo" right 0 of 4 times although the prompt
states the local time, zone and offset), so the conversion is done here.

A place is resolved locally, in this order: a small built-in table of common
names that are not zone names (countries, US and Australian states, large
cities, "Eastern time", "PST"); an IANA zone name ("Asia/Tokyo"); the city part
of a zone name ("Tokyo", "Sao Paulo"); a country from the database's
`iso3166.tab` and `zone.tab` (when the country has one zone). "Tokyo, Japan"
is tried whole and then part by part. An unknown or ambiguous place gives a
short error with close matches, so the model can ask again.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
import zoneinfo
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from functools import cache
from pathlib import Path
from typing import Any, Final, NamedTuple
from zoneinfo import ZoneInfo

from lucy.core.types import Risk, ToolOutput
from lucy.tools.common import error, required_text

_AREAS: Final = frozenset(
    {"Africa", "America", "Antarctica", "Asia", "Atlantic", "Australia", "Europe", "Indian"}
    | {"Pacific"}
)
"""Zone-name prefixes whose last part is a city (not "US/Pacific" or "Etc/GMT+5")."""

_SEVERAL: Final = 8
"""Most zones listed for a place with several."""

# Common names that are not zone names and not a country with a single zone in
# iso3166.tab. States and countries spanning zones map to the zone most people
# there use. Keys are compared after `_key`.
_ALIASES: Final[dict[str, str]] = {
    # Abbreviations and zone nicknames; the region's clock, daylight saving included.
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "pacific time": "America/Los_Angeles",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "mountain time": "America/Denver",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "central time": "America/Chicago",
    "est": "America/New_York",
    "edt": "America/New_York",
    "eastern time": "America/New_York",
    "bst": "Europe/London",
    "cet": "Europe/Berlin",
    "cest": "Europe/Berlin",
    "central european time": "Europe/Berlin",
    "jst": "Asia/Tokyo",
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
    # Countries whose database name differs from the everyday one.
    "uk": "Europe/London",
    "united kingdom": "Europe/London",
    "great britain": "Europe/London",
    "england": "Europe/London",
    "scotland": "Europe/London",
    "wales": "Europe/London",
    "northern ireland": "Europe/London",
    "south korea": "Asia/Seoul",
    "korea": "Asia/Seoul",
    "north korea": "Asia/Pyongyang",
    "uae": "Asia/Dubai",
    "czech republic": "Europe/Prague",
    "holland": "Europe/Amsterdam",
    "the netherlands": "Europe/Amsterdam",
    "china": "Asia/Shanghai",
    # US states and Washington, D.C.
    "washington dc": "America/New_York",
    "washington d c": "America/New_York",
    "alabama": "America/Chicago",
    "alaska": "America/Anchorage",
    "arizona": "America/Phoenix",
    "arkansas": "America/Chicago",
    "california": "America/Los_Angeles",
    "colorado": "America/Denver",
    "connecticut": "America/New_York",
    "delaware": "America/New_York",
    "florida": "America/New_York",
    "georgia usa": "America/New_York",
    "hawaii": "Pacific/Honolulu",
    "idaho": "America/Boise",
    "illinois": "America/Chicago",
    "indiana": "America/Indiana/Indianapolis",
    "iowa": "America/Chicago",
    "kansas": "America/Chicago",
    "kentucky": "America/New_York",
    "louisiana": "America/Chicago",
    "maine": "America/New_York",
    "maryland": "America/New_York",
    "massachusetts": "America/New_York",
    "michigan": "America/Detroit",
    "minnesota": "America/Chicago",
    "mississippi": "America/Chicago",
    "missouri": "America/Chicago",
    "montana": "America/Denver",
    "nebraska": "America/Chicago",
    "nevada": "America/Los_Angeles",
    "new hampshire": "America/New_York",
    "new jersey": "America/New_York",
    "new mexico": "America/Denver",
    "new york state": "America/New_York",
    "north carolina": "America/New_York",
    "north dakota": "America/Chicago",
    "ohio": "America/New_York",
    "oklahoma": "America/Chicago",
    "oregon": "America/Los_Angeles",
    "pennsylvania": "America/New_York",
    "rhode island": "America/New_York",
    "south carolina": "America/New_York",
    "south dakota": "America/Chicago",
    "tennessee": "America/Chicago",
    "texas": "America/Chicago",
    "utah": "America/Denver",
    "vermont": "America/New_York",
    "virginia": "America/New_York",
    "washington state": "America/Los_Angeles",
    "west virginia": "America/New_York",
    "wisconsin": "America/Chicago",
    "wyoming": "America/Denver",
    # Australian states, Canadian provinces.
    "new south wales": "Australia/Sydney",
    "victoria australia": "Australia/Melbourne",
    "queensland": "Australia/Brisbane",
    "western australia": "Australia/Perth",
    "south australia": "Australia/Adelaide",
    "tasmania": "Australia/Hobart",
    "northern territory": "Australia/Darwin",
    "ontario": "America/Toronto",
    "quebec": "America/Toronto",
    "british columbia": "America/Vancouver",
    "alberta": "America/Edmonton",
    "manitoba": "America/Winnipeg",
    "saskatchewan": "America/Regina",
    "nova scotia": "America/Halifax",
    "newfoundland": "America/St_Johns",
    # Large cities that are not zone names.
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "san diego": "America/Los_Angeles",
    "las vegas": "America/Los_Angeles",
    "portland oregon": "America/Los_Angeles",
    "dallas": "America/Chicago",
    "houston": "America/Chicago",
    "austin": "America/Chicago",
    "boston": "America/New_York",
    "miami": "America/New_York",
    "atlanta": "America/New_York",
    "philadelphia": "America/New_York",
    "washington": "America/New_York",
    "montreal": "America/Toronto",
    "ottawa": "America/Toronto",
    "calgary": "America/Edmonton",
    "rio de janeiro": "America/Sao_Paulo",
    "beijing": "Asia/Shanghai",
    "shenzhen": "Asia/Shanghai",
    "guangzhou": "Asia/Shanghai",
    "osaka": "Asia/Tokyo",
    "kyoto": "Asia/Tokyo",
    "mumbai": "Asia/Kolkata",
    "bombay": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "new delhi": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "bengaluru": "Asia/Kolkata",
    "chennai": "Asia/Kolkata",
    "hyderabad": "Asia/Kolkata",
    "pune": "Asia/Kolkata",
    "abu dhabi": "Asia/Dubai",
    "doha": "Asia/Qatar",
    "hanoi": "Asia/Bangkok",
    "saigon": "Asia/Ho_Chi_Minh",
    "bali": "Asia/Makassar",
    "frankfurt": "Europe/Berlin",
    "munich": "Europe/Berlin",
    "milan": "Europe/Rome",
    "barcelona": "Europe/Madrid",
    "geneva": "Europe/Zurich",
    "edinburgh": "Europe/London",
    "manchester": "Europe/London",
    "cape town": "Africa/Johannesburg",
    "auckland": "Pacific/Auckland",
}


_COUNTRY_ALIASES: Final = {
    k: "unitedstates" for k in ("us", "usa", "america", "unitedstatesofamerica")
}
"""Names of the United States, which has several zones (keys as `_key` makes them)."""


class _Place(NamedTuple):
    zone: str | None
    """The zone, or None when the place is unknown or ambiguous."""
    problem: str = ""


def _key(text: str) -> str:
    """`text` for matching: lower case, accents and punctuation removed, words joined."""
    text = unicodedata.normalize("NFKD", text.replace("&", " and "))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return "".join(c for c in text if c.isalnum())


class _Database(NamedTuple):
    zones: frozenset[str]
    cities: dict[str, list[str]]
    """`_key` of a zone's city part, and the zones with that city."""
    countries: dict[str, tuple[str, list[str]]]
    """`_key` of a country name, the name, and its zones."""
    canonical: frozenset[str]
    """Zones listed in zone.tab (the database's own choice among aliases)."""


@cache
def _database() -> _Database:
    zones = frozenset(zoneinfo.available_timezones())
    cities: dict[str, list[str]] = {}
    for zone in sorted(zones):
        area, _, rest = zone.partition("/")
        if area in _AREAS and rest:
            cities.setdefault(_key(rest.rsplit("/", 1)[-1]), []).append(zone)
    names: dict[str, str] = {}
    by_code: dict[str, list[str]] = {}
    for line in _tab_lines("iso3166.tab"):
        code, _, name = line.partition("\t")
        names[code] = name.strip()
    for line in _tab_lines("zone.tab"):
        fields = line.split("\t")
        if len(fields) >= 3 and fields[2] in zones:
            by_code.setdefault(fields[0], []).append(fields[2])
    countries: dict[str, tuple[str, list[str]]] = {}
    for code, name in names.items():
        if code in by_code:
            for variant in {name, re.sub(r"\s*\(.*\)", "", name)}:
                countries.setdefault(_key(variant), (variant, by_code[code]))
    canonical = frozenset(zone for listed in by_code.values() for zone in listed)
    return _Database(zones, cities, countries, canonical)


def _tab_lines(name: str) -> list[str]:
    """Non-comment lines of a table shipped with the system time-zone database; empty
    when it is not there (resolving countries then needs the built-in table)."""
    for directory in zoneinfo.TZPATH:
        path = Path(directory) / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        return [line for line in text.splitlines() if line and not line.startswith("#")]
    return []


@cache
def _alias_keys() -> dict[str, str]:
    return {_key(name): zone for name, zone in _ALIASES.items()}


def resolve_place(place: str, at: datetime) -> _Place:
    """The time zone of `place`, or why there is none. `at` decides whether zones that
    share a city name differ (they rarely do)."""
    db = _database()
    parts = [p for p in (s.strip() for s in place.split(",")) if p]
    for attempt in [place.strip()] + (parts if len(parts) > 1 else []):
        key = _key(attempt)
        if not key:
            continue
        if key in _alias_keys():
            return _Place(_alias_keys()[key])
        name = attempt.replace(" ", "_").lower()
        exact = sorted(z for z in db.zones if z.lower() == name)
        if exact and "/" in exact[0]:
            return _Place(exact[0])
        if key in db.cities:
            return _choose(db.cities[key], attempt, at)
        country = db.countries.get(_COUNTRY_ALIASES.get(key, key))
        if country:
            return _country(*country, at)
        if exact:  # "UTC", "GMT" and old names such as "Japan"
            return _Place(exact[0])
    return _Place(None, _unknown(place, parts, db))


def _unknown(place: str, parts: list[str], db: _Database) -> str:
    names = {_key(name): name for name in _ALIASES}
    names.update(
        {k: zones[0].rsplit("/", 1)[-1].replace("_", " ") for k, zones in db.cities.items()}
    )
    names.update({k: name for k, (name, _) in db.countries.items()})
    close = difflib.get_close_matches(_key(parts[-1] if parts else place), names, n=5, cutoff=0.75)
    hint = f" Close matches: {', '.join(names[k] for k in close)}." if close else ""
    return (
        f"No time zone known for {place!r}.{hint} Try a nearby large city, the country, "
        "or a zone name such as 'America/Chicago'."
    )


def _country(name: str, zones: list[str], at: datetime) -> _Place:
    """The zone of a country, or a list of its distinct clocks at `at`, each shown by a
    well-known zone (one in the built-in table) where there is one, else zone.tab's
    first (the database lists the most populous zone first)."""
    known = list(dict.fromkeys(_ALIASES.values()))
    ranked = sorted(zones, key=lambda z: known.index(z) if z in known else len(known))
    clocks: dict[timedelta | None, str] = {}
    for zone in ranked:
        clocks.setdefault(at.astimezone(ZoneInfo(zone)).utcoffset(), zone)
    if len(clocks) == 1:
        return _Place(zones[0])
    order = sorted(clocks.items(), key=lambda item: item[0] or timedelta(), reverse=True)
    listed = [f"{zone} ({_offset(offset)})" for offset, zone in order]
    more = " and more" if len(listed) > _SEVERAL else ""
    return _Place(
        None,
        f"{name} has several time zones: {', '.join(listed[:_SEVERAL])}{more}. Ask again "
        "with a city or region there.",
    )


def _choose(zones: list[str], place: str, at: datetime) -> _Place:
    """One of the zones sharing a city name: the listed one when they agree at `at`."""
    if len({at.astimezone(ZoneInfo(z)).utcoffset() for z in zones}) == 1:
        preferred = [z for z in zones if z in _database().canonical] or zones
        return _Place(min(preferred, key=len))
    return _Place(None, f"{place!r} is ambiguous: {', '.join(zones)}. Ask again with one of them.")


_CLOCK: Final = re.compile(r"^(\d{1,2})(?:[:.h](\d{2}))?\s*([ap])?\.?\s*m?\.?$")


def parse_clock(text: str) -> time | None:
    """A time of day from "9", "9am", "9:30 pm", "21:00", "noon" or "midnight", or None."""
    text = text.strip().lower()
    if text in {"noon", "midday"}:
        return time(12)
    if text == "midnight":
        return time(0)
    match = _CLOCK.match(text)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    if match.group(3):
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if match.group(3) == "p" else 0)
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def _offset(delta: timedelta | None) -> str:
    minutes = round((delta or timedelta()).total_seconds() / 60)
    sign = "+" if minutes >= 0 else "-"
    hours, rest = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours:02d}:{rest:02d}"


def _moment(when: datetime) -> str:
    """e.g. "Thursday 12 November 2026, 15:12 (3:12 pm) JST"."""
    twelve = f"{(when.hour % 12) or 12}:{when.minute:02d} {'am' if when.hour < 12 else 'pm'}"
    zone = when.tzname() or ""
    zone = f" {zone}" if zone.isalpha() else ""
    return f"{when:%A} {when.day} {when:%B %Y}, {when:%H:%M} ({twelve}){zone}"


def _difference(there: datetime, here: datetime) -> str:
    gap = (there.utcoffset() or timedelta()) - (here.utcoffset() or timedelta())
    minutes = round(gap.total_seconds() / 60)
    if minutes == 0:
        return "the same as the user's local time"
    hours, rest = divmod(abs(minutes), 60)
    parts = [f"{hours} hour{'' if hours == 1 else 's'}"] if hours else []
    if rest:
        parts.append(f"{rest} minute{'' if rest == 1 else 's'}")
    side = "ahead of" if minutes > 0 else "behind"
    return f"{' '.join(parts)} {side} the user's local time"


def _day_note(there: date, here: date) -> str:
    if there > here:
        return " It is already the next day there."
    if there < here:
        return " It is still the previous day there."
    return ""


class TimeTool:
    """The date and time in a named place (risk tier read; no network)."""

    name = "time"
    description = (
        "The current date and time in a city, country or time zone, worked out on this "
        "machine, for a place other than where the user is. Do not call it when no other "
        "place is named: the date and time you are given are the user's own. Pass the "
        "place, for example 'Tokyo', 'New York', 'India' or 'Europe/Paris'. To convert a "
        "clock time at the user's location, also pass it as 'at', for example '9am'."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "place": {"type": "string", "description": "A city, country or IANA time zone."},
            "at": {
                "type": "string",
                "description": "Optional: a time of day today at the user's location to "
                "convert, such as '9am' or '17:30'. Leave it out for the time now.",
            },
        },
        "required": ["place"],
    }
    risk = Risk.READ

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        local_zone: tzinfo | None = None,
    ) -> None:
        """`clock` is "now" (timezone-aware; default the system clock). `local_zone` is
        the user's zone (default the system's local zone)."""
        self._clock = clock or (lambda: datetime.now(UTC))
        self._local_zone = local_zone

    def _here(self, instant: datetime) -> datetime:
        if self._local_zone is not None:
            return instant.astimezone(self._local_zone)
        return instant.astimezone()

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        place = required_text(args, "place")
        if place is None:
            return error("The time tool needs a 'place': a city, country or time zone.")
        here = self._here(self._clock())
        at = required_text(args, "at")
        converting = at is not None and at.lower() not in {"now", "right now"}
        if converting:
            clock_time = parse_clock(at or "")
            if clock_time is None:
                return error(f"Could not read {at!r} as a time of day; use a form like '9am'.")
            local_zone = self._local_zone or here.tzinfo
            here = datetime.combine(here.date(), clock_time, tzinfo=local_zone)
        resolved = resolve_place(place, here)
        if resolved.zone is None:
            return error(resolved.problem)
        there = here.astimezone(ZoneInfo(resolved.zone))
        where = f"{place} (time zone {resolved.zone}, {_offset(there.utcoffset())})"
        if converting:
            first = (
                f"When it is {_moment(here)} at the user's location, it is {_moment(there)} "
                f"in {where}."
            )
        else:
            first = f"In {where} it is now {_moment(there)}."
        compare = f"That is {_difference(there, here)}"
        if not converting:
            compare += f" ({_moment(here)})"
        return ToolOutput(content=f"{first} {compare}.{_day_note(there.date(), here.date())}")
