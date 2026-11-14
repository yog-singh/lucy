"""Dates and times for the calendar tools, resolved in code.

The chat model gets date arithmetic wrong, so the calendar tools take the user's
own words and resolve them here, in the user's local time zone:

- "now", "today", "tomorrow", "day after tomorrow", "yesterday";
- a weekday ("friday", "this friday"): the coming one, today if today is that day;
- "this week" (today to Sunday), "next week" (Monday to Sunday), "this weekend";
- a date: ISO "2026-11-20", or "20 November", "November 20th", with an optional
  year (without one, the nearest such date to today);
- any of those days with a time ("friday 3pm", "tomorrow at 15:30", "2026-11-20T15:00")
  or a part of the day ("friday afternoon": 12:00 to 18:00; "tonight").

Anything else, and phrases people use for two different days ("next friday",
"next weekend", "11/12"), is refused with a message that says what is accepted
and, where it helps, which dates were possible, so the model asks the user
instead of guessing. Weeks start on Monday.

Times are built with the zone's rules, so days around a daylight-saving change
have 23 or 25 hours, and a clock time that does not exist that day (the hour the
clocks skip) is refused.

`resolve_moment` reads one future instant for a reminder: a length of time from
now ("in 20 minutes", number words included), or a day and time as above, plus a
part of the day with a clock time ("tomorrow morning at 9"). There a bare hour
("at 5") is the soonest future 05:00 or 17:00, unlike `resolve`, which reads it
as 05:00.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lucy.tools.local_time import parse_clock

ACCEPTED: Final = (
    "Accepted: 'today', 'tomorrow', 'day after tomorrow', 'yesterday', a weekday such as "
    "'friday', 'this week', 'next week', 'this weekend', or a date such as '2026-11-20' or "
    "'20 November'; optionally with a time ('friday 3pm', 'tomorrow 15:30', "
    "'2026-11-20T15:00') or a part of the day ('friday afternoon')."
)

WEEKDAYS: Final = {
    name: i
    for i, names in enumerate(
        [
            ("monday", "mon"),
            ("tuesday", "tue", "tues"),
            ("wednesday", "wed"),
            ("thursday", "thu", "thur", "thurs"),
            ("friday", "fri"),
            ("saturday", "sat"),
            ("sunday", "sun"),
        ]
    )
    for name in names
}
MONTHS: Final = {
    name: i + 1
    for i, full in enumerate(
        "january february march april may june july august september october november "
        "december".split()
    )
    for name in (full, full[:3], "sept" if full == "september" else full[:3])
}
PARTS: Final = {
    "morning": (time(6), time(12)),
    "afternoon": (time(12), time(18)),
    "evening": (time(18), None),
    "night": (time(18), None),
}
"""Parts of the day: start and end (None: midnight at the end of the day)."""


class DateError(ValueError):
    """Input that cannot be resolved; the message is for the model."""


@dataclass(frozen=True)
class Span:
    """A resolved stretch of local time; `end` is exclusive."""

    start: datetime
    end: datetime
    kind: str
    """"point" (one moment, end == start), "part" (part of a day), "days" (whole days)."""
    days: int = 0
    """For "days": how many."""

    @property
    def is_point(self) -> bool:
        return self.kind == "point"


def system_zone() -> tzinfo:
    """The machine's time zone with its daylight-saving rules (not just today's offset)."""
    name = os.environ.get("TZ", "").lstrip(":")
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    try:
        target = str(Path("/etc/localtime").resolve())
        if "zoneinfo/" in target:
            return ZoneInfo(target.split("zoneinfo/", 1)[1])
    except (OSError, ZoneInfoNotFoundError, ValueError):
        pass
    return datetime.now().astimezone().tzinfo or UTC


def local(day: date, clock: time, zone: tzinfo, *, strict: bool = True) -> datetime:
    """`day` at `clock` in `zone`. A clock time skipped by a daylight-saving change is
    refused when `strict`, else moved forward."""
    moment = datetime.combine(day, clock, tzinfo=zone)
    back = moment.astimezone(UTC).astimezone(zone)
    if back.replace(tzinfo=None) != moment.replace(tzinfo=None):
        if strict:
            raise DateError(
                f"{clock:%H:%M} does not exist on {day_text(day)}: the clocks change then. "
                "Ask the user for another time."
            )
        return back
    return moment


def midnight(day: date, zone: tzinfo) -> datetime:
    return local(day, time(0), zone, strict=False)


def day_text(day: date) -> str:
    """e.g. "Tuesday 17 November 2026"."""
    return f"{day:%A} {day.day} {day:%B %Y}"


def moment_text(when: datetime) -> str:
    """e.g. "Tuesday 17 November 2026, 15:00"."""
    return f"{day_text(when.date())}, {when:%H:%M}"


def when_text(start: datetime, end: datetime, all_day: bool) -> str:
    """An event's time in words, every date with its weekday, e.g. "Tuesday 17 November
    2026, 15:00 to 16:00" or "Friday 20 November 2026 to Sunday 22 November 2026, all day"."""
    if all_day:
        last = (end - timedelta(microseconds=1)).date() if end > start else start.date()
        if last == start.date():
            return f"{day_text(start.date())}, all day"
        return f"{day_text(start.date())} to {day_text(last)}, all day"
    if end.date() == start.date() or (end - start <= timedelta(days=1) and end.time() == time(0)):
        until = "midnight" if end.date() != start.date() else f"{end:%H:%M}"
        return f"{moment_text(start)} to {until}"
    return f"{moment_text(start)} to {moment_text(end)}"


def span_text(span: Span) -> str:
    """A resolved range in words, e.g. "Friday 20 November 2026, 12:00 to 18:00"."""
    if span.kind == "days":
        return when_text(span.start, span.end, True).removesuffix(", all day")
    return when_text(span.start, span.end, False)


def _clean(text: str) -> str:
    text = text.strip().lower().replace(",", " ")
    text = re.sub(r"[.!?]+$", "", text)
    text = re.sub(r"\b(at|on|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve(text: str, now: datetime, *, base: date | None = None) -> Span:
    """`text` as local time, relative to `now` (aware, in the user's zone). A bare time
    of day ("16:00") falls on `base` (default today)."""
    zone = now.tzinfo
    assert zone is not None
    raw = text.strip()
    if not raw:
        raise DateError(f"An empty date was given. {ACCEPTED}")
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return _iso(raw, zone)
    words = _clean(raw)
    if words in {"now", "right now"}:
        return Span(now, now, "point")
    if re.search(r"\b\d{1,4}/\d{1,2}(/\d{2,4})?\b", words):
        raise DateError(
            f"{raw!r} is ambiguous: day and month can be read either way. Use a form like "
            f"'{now:%Y-%m-%d}' or '{now.day} {now:%B}'."
        )
    if words == "tonight":
        words = "today evening"
    tokens = words.split()
    # Try every way of reading the words as a day, then a time or part of the day
    # (or the time first: "3pm friday").
    for k in range(len(tokens), -1, -1):
        for day_words, rest in (
            (tokens[:k], tokens[k:]),
            (tokens[len(tokens) - k :], tokens[: len(tokens) - k]),
        ):
            if not day_words and not rest:
                continue
            days = _days(" ".join(day_words), now.date()) if day_words else None
            if day_words and days is None:
                continue
            first, count = days or (base or now.date(), 1)
            if not rest:
                if days is None:
                    continue
                return Span(
                    midnight(first, zone), midnight(first + timedelta(count), zone), "days", count
                )
            extra = " ".join(rest)
            clock = parse_clock(extra)
            part = PARTS.get(extra)
            if clock is None and part is None:
                continue
            if count != 1:
                raise DateError(
                    f"{raw!r} names several days and a time; pass one day with the time."
                )
            if clock is not None:
                moment = local(first, clock, zone)
                return Span(moment, moment, "point")
            assert part is not None
            begin, finish = part
            end = (
                local(first, finish, zone, strict=False)
                if finish
                else midnight(first + timedelta(1), zone)
            )
            return Span(local(first, begin, zone, strict=False), end, "part")
    raise DateError(f"Could not read {raw!r} as a date or time. {ACCEPTED}")


def _iso(raw: str, zone: tzinfo) -> Span:
    try:
        if len(raw) == 10:
            day = date.fromisoformat(raw)
            return Span(midnight(day, zone), midnight(day + timedelta(1), zone), "days", 1)
        moment = datetime.fromisoformat(raw.replace(" ", "T", 1) if " " in raw[:11] else raw)
    except ValueError:
        moment = None
    if moment is None:
        # "2026-11-20 3pm": an ISO date with a spoken time.
        day_part, _, rest = raw.partition(" ")
        clock = parse_clock(rest) if rest else None
        try:
            day = date.fromisoformat(day_part)
        except ValueError:
            day = None
        if day is None or clock is None:
            raise DateError(f"Could not read {raw!r} as a date or time. {ACCEPTED}") from None
        moment = local(day, clock, zone)
        return Span(moment, moment, "point")
    if moment.tzinfo is not None:
        moment = moment.astimezone(zone)
    else:
        moment = local(moment.date(), moment.time(), zone)
    return Span(moment, moment, "point")


def _days(words: str, today: date) -> tuple[date, int] | None:
    """The first day and number of days `words` names, or None if it names none."""
    words = (
        words.removeprefix("this ").strip() if words not in {"this week", "this weekend"} else words
    )
    if words == "today":
        return today, 1
    if words == "tomorrow":
        return today + timedelta(1), 1
    if words in {"day after tomorrow", "the day after tomorrow"}:
        return today + timedelta(2), 1
    if words == "yesterday":
        return today - timedelta(1), 1
    if words in WEEKDAYS:
        return today + timedelta((WEEKDAYS[words] - today.weekday()) % 7), 1
    monday = today - timedelta(today.weekday())
    if words == "this week":
        return today, (monday + timedelta(7) - today).days
    if words == "next week":
        return monday + timedelta(7), 7
    if words == "this weekend" or words == "weekend":
        saturday = monday + timedelta(5)
        first = max(saturday, today)
        return first, (monday + timedelta(7) - first).days
    if words.startswith("next ") or words.startswith("last "):
        _ambiguous(words, today)
    return _calendar_date(words, today)


def _ambiguous(words: str, today: date) -> None:
    """Raise for "next friday" and the like, naming the days it could mean."""
    rest = words.split(" ", 1)[1]
    if words.startswith("next ") and rest in WEEKDAYS:
        coming = today + timedelta((WEEKDAYS[rest] - today.weekday() - 1) % 7 + 1)
        raise DateError(
            f"{words!r} is ambiguous: it could mean {day_text(coming)} or "
            f"{day_text(coming + timedelta(7))}. Ask the user which, then pass the date, "
            f"e.g. '{coming + timedelta(7):%Y-%m-%d}'."
        )
    if words == "next weekend":
        saturday = today + timedelta((5 - today.weekday()) % 7)
        if today.weekday() == 6:
            saturday += timedelta(7)
        raise DateError(
            f"'next weekend' is ambiguous: it could start {day_text(saturday)} or "
            f"{day_text(saturday + timedelta(7))}. Ask the user which, then pass the date."
        )


_DAY_MONTH: Final = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)? (?:of )?([a-z]+)(?: (\d{4}))?$")
_MONTH_DAY: Final = re.compile(r"^([a-z]+) (\d{1,2})(?:st|nd|rd|th)?(?: (\d{4}))?$")


def _calendar_date(words: str, today: date) -> tuple[date, int] | None:
    match = _DAY_MONTH.match(words)
    if match:
        day_s, month_s, year_s = match.groups()
    else:
        match = _MONTH_DAY.match(words)
        if not match:
            return None
        month_s, day_s, year_s = match.groups()
    month = MONTHS.get(month_s)
    if month is None:
        return None
    day = int(day_s)
    try:
        if year_s:
            return date(int(year_s), month, day), 1
        candidates = []
        for year in (today.year - 1, today.year, today.year + 1):
            try:
                candidates.append(date(year, month, day))
            except ValueError:
                continue
        if not candidates:
            raise ValueError
    except ValueError:
        raise DateError(f"{words!r} is not a date on the calendar.") from None
    return min(candidates, key=lambda d: abs((d - today).days)), 1


# --- Moments, for reminders and timers ------------------------------------------

MOMENT_ACCEPTED: Final = (
    "Accepted: a length of time from now ('in 20 minutes', 'in an hour and a half'), a day "
    "with a time ('tomorrow 9am', 'friday at 17:30', 'tomorrow morning at 9', '20 November "
    "3pm'), or a time later today ('5pm', '17:30')."
)
_NUMBERS: Final = {
    word: i
    for i, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS: Final = {
    word: 10 * (i + 2)
    for i, word in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())
}
_UNITS: Final = {
    **dict.fromkeys(("second", "seconds", "sec", "secs", "s"), timedelta(seconds=1)),
    **dict.fromkeys(("minute", "minutes", "min", "mins", "m"), timedelta(minutes=1)),
    **dict.fromkeys(("hour", "hours", "hr", "hrs", "h"), timedelta(hours=1)),
}
_LONG_UNITS: Final = frozenset(
    "day days week weeks fortnight fortnights month months year years".split()
)
_PHRASES: Final = (
    (r"\bthree quarters? of an hour\b", "45 minutes"),
    (r"\b(?:a )?quarter (?:of )?(?:an )?hour\b", "15 minutes"),
    (r"\b(?:a |one )?half (?:an |a )?hour\b", "30 minutes"),
    (r"\bhalf (?:a )?minute\b", "30 seconds"),
    (r"\ba couple of\b", "2"),
    (r"\band a half\b", "+half"),
)
_PART_WORDS: Final = re.compile(r"\b(?:in the |this |at )?(morning|afternoon|evening|night)\b")
_MERIDIEM: Final = re.compile(r"\d\s*[ap]\.?m?\.?(?![a-z])|\d[:.h]\d|\b(?:noon|midday|midnight)\b")


@dataclass(frozen=True)
class Moment:
    """A resolved instant for a reminder, in the user's zone."""

    at: datetime
    offset: timedelta | None = None
    """Set when the words were a length of time from now ("in 20 minutes")."""
    note: str | None = None
    """Something the model must say back, e.g. how a bare hour was read."""


def _number_tokens(words: str) -> list[str]:
    """Tokens with spelled-out numbers ("twenty five", "an hour") turned into digits."""
    tokens = words.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token, nxt = tokens[i], tokens[i + 1] if i + 1 < len(tokens) else ""
        if token in _TENS:
            value = _TENS[token]
            if nxt in _NUMBERS and 0 < _NUMBERS[nxt] < 10:
                value += _NUMBERS[nxt]
                i += 1
            out.append(str(value))
        elif token in _NUMBERS:
            out.append(str(_NUMBERS[token]))
        elif token in {"a", "an"} and (nxt in _UNITS or nxt in _LONG_UNITS):
            out.append("1")
        else:
            out.append(token)
        i += 1
    return out


def parse_offset(text: str) -> timedelta | None:
    """A length of time ("in 20 minutes", "an hour and a half", "twenty five minutes",
    "1h30"), or None if `text` is not one. Raises `DateError` for days or weeks, and
    for lengths of zero."""
    words = text.strip().lower().replace("-", " ").replace(",", " ")
    words = re.sub(r"[.!?]+$", "", words).strip()
    words = re.sub(r"(\d)([a-z])", r"\1 \2", words)  # "10min", "1h30"
    words = re.sub(r"\b([hm])(\d)", r"\1 \2", words)
    words = re.sub(r"^(?:in|for|after)\s+", "", words)
    words = re.sub(r"\s+(?:from now|time|'s time)$", "", words)
    for pattern, replacement in _PHRASES:
        words = re.sub(pattern, replacement, words)
    tokens = _number_tokens(words)
    total, last_unit = timedelta(), None
    i = 0
    while i < len(tokens):
        if tokens[i] == "and" and last_unit is not None:
            i += 1
            continue
        if not re.fullmatch(r"\d+(?:\.\d+)?", tokens[i]):
            return None
        value = float(tokens[i])
        i += 1
        if i < len(tokens) and tokens[i] == "+half":
            value, i = value + 0.5, i + 1
        word = tokens[i] if i < len(tokens) else ""
        if word in _LONG_UNITS:
            raise DateError(
                f"{text!r} is days or weeks away; a reminder takes a day and a time instead, "
                "such as 'tuesday 9am' or '20 November at 17:00'. Ask the user which day and "
                "what time."
            )
        unit = _UNITS.get(word)
        if unit is None:
            # A bare number closing a length: "1 hour 30" is minutes, "1h30" too.
            if word or last_unit is None or last_unit == timedelta(seconds=1):
                return None
            hours = last_unit == timedelta(hours=1)
            total += value * (timedelta(minutes=1) if hours else timedelta(seconds=1))
            break
        i += 1
        if i < len(tokens) and tokens[i] == "+half":
            value, i = value + 0.5, i + 1
        total += value * unit
        last_unit = unit
    if last_unit is None:
        return None
    total = timedelta(seconds=round(total.total_seconds()))
    if total <= timedelta():
        raise DateError(f"{text!r} is no time at all. Ask the user how long.")
    return total


def resolve_moment(text: str, now: datetime) -> Moment:
    """`text` as one future instant for a reminder, relative to `now` (aware, in the
    user's zone).

    A length of time ("in 20 minutes") is added to the instant (so it is right across
    a daylight-saving change). Otherwise a day and time as `resolve` reads them, plus
    a part of the day with a clock time ("tomorrow morning at 9", "9 in the
    evening"). A bare hour without am or pm ("at 5") is the soonest future 5:00 or
    17:00, and the `note` says so; with a day named ("friday at 5") it is refused,
    asking the user for morning or evening. Refused: a day or part of the day without a time,
    lengths in days or weeks, and times already past.
    """
    zone = now.tzinfo
    assert zone is not None
    raw = text.strip()
    if not raw:
        raise DateError(f"No time was given. Ask the user when. {MOMENT_ACCEPTED}")
    offset = parse_offset(raw)
    if offset is not None:
        return Moment((now.astimezone(UTC) + offset).astimezone(zone), offset=offset)
    iso = bool(re.match(r"^\d{4}-\d{2}-\d{2}", raw))
    words = raw.lower().replace(",", " ")
    words = re.sub(r"[.!?]+$", "", words)
    words = re.sub(r"\s*o'?\s?clock\b", "", words)
    part: str | None = None
    if re.search(r"\btonight\b", words):
        part, words = "evening", re.sub(r"\btonight\b", " ", words)
    elif not iso and (match := _PART_WORDS.search(words)):
        part, words = match.group(1), words[: match.start()] + " " + words[match.end() :]
    words = re.sub(r"\s+", " ", words).strip()
    if words in {"now", "right now"}:
        raise DateError("A reminder needs a time later than now. Ask the user when.")
    if not words:
        assert part is not None
        raise DateError(
            f"{raw!r} is a part of the day, not a time. Ask the user what time to remind them."
        )
    try:
        span = resolve(raw if iso else words, now)
    except DateError as exc:
        if str(exc).startswith("Could not read"):
            raise DateError(f"Could not read {raw!r} as a time. {MOMENT_ACCEPTED}") from None
        raise
    if not span.is_point:
        what = "a part of the day" if part or span.kind == "part" else "a day without a time"
        raise DateError(
            f"{raw!r} is {what} ({span_text(span)}), not a time. Ask the user what time to "
            "remind them."
        )
    at, note = span.start, None
    explicit = iso or bool(_MERIDIEM.search(words))
    if part and not explicit:
        if part != "morning" and 1 <= at.hour <= 11:
            at = local(at.date(), time(at.hour + 12, at.minute), zone)
        elif part in {"evening", "night"} and at.hour == 12 and at.minute == 0:
            at = midnight(at.date() + timedelta(1), zone)
    elif not part and not explicit and at.minute == 0 and 1 <= at.hour <= 12:
        if parse_clock(_clean(words)) is None:  # a day was named: ask, do not guess
            raise DateError(_ask_meridiem(raw, at))
        at, note = _bare_hour(at, now)
    if at <= now:
        raise DateError(
            f"{moment_text(at)} is in the past (it is now {moment_text(now)}). Ask the user "
            "for a later time."
        )
    return Moment(at, note=note)


def _ask_meridiem(raw: str, at: datetime) -> str:
    """The refusal for a bare hour on a named day ("friday at 5"): ask morning or evening."""
    if at.hour == 12:
        readings = f"12:00 (noon) on {at:%A} or midnight at the end of {at:%A}"
    else:
        readings = f"{at.hour:02d}:00 or {at.hour + 12:02d}:00 on {at:%A} {at.day} {at:%B}"
    return (
        f"{raw!r} gives no am or pm. Ask the user whether they mean the morning or the "
        f"evening: {readings}."
    )


def _bare_hour(at: datetime, now: datetime) -> tuple[datetime, str | None]:
    """The soonest future reading of a bare hour with no day named (5:00 or 17:00; 12:00
    or midnight), today or tomorrow; and a note saying how it was read."""
    zone = now.tzinfo
    assert zone is not None
    hours = (12, 24) if at.hour == 12 else (at.hour, at.hour + 12)
    days = [now.date(), now.date() + timedelta(1)]
    candidates = []
    for day in days:
        for hour in hours:
            try:
                candidates.append(local(day + timedelta(hour // 24), time(hour % 24), zone))
            except DateError:
                continue
    future = [c for c in candidates if c > now]
    if not future:
        return at, None  # past: refused by the caller
    chosen = min(future)
    readings = " and ".join("midnight" if h == 24 else f"{h:02d}:00" for h in hours)
    return chosen, (
        f"No am or pm was given, so {at.hour} was read as {chosen:%H:%M}, the sooner of "
        f"{readings}. Say this time back so the user can correct it."
    )
