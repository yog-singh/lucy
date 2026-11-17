"""Tools the chat model can call, and the catalogue that builds the enabled ones.

Many tools are active at once, so tools do not go through the one-adapter-per-
port registry: `CATALOGUE` maps each tool name to a factory taking the settings
(and, for tests, an HTTP transport). `build_tools` builds the tools listed in
`[tools] enabled`, in that order. Tools that hold an HTTP client have an
`aclose()` method; the app calls it on stop. The `time` tool needs no network:
it reads the system's time-zone database. The calendar tools (`calendar`,
`add_calendar_event`, `edit_calendar_event`, `delete_calendar_event`; the last two
always ask the user first) need none either: they use the macOS calendar store, and
load EventKit only on their first call. The reminder tools (`reminders`,
`set_reminder`, `set_timer`, `cancel_reminder`) keep reminders in the SQLite memory
database; building them fails when memory is not the SQLite store on a file.
The `look` tool takes one camera frame and has the vision model describe it (ADR
0026); the app attaches its `VisionService` when `[vision]` is enabled.

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
from lucy.core.reminders import ReminderStore
from lucy.core.toolsets import ToolGroup
from lucy.tools.calendar_changes import DeleteCalendarEventTool, EditCalendarEventTool
from lucy.tools.calendar_events import AddCalendarEventTool, CalendarTool
from lucy.tools.face_enrolment import EnrollFaceTool, ReplaceFaceTool
from lucy.tools.fetch_page import FetchPageTool
from lucy.tools.local_time import TimeTool
from lucy.tools.look import LookTool
from lucy.tools.netguard import Resolver, system_resolver
from lucy.tools.reminders import (
    CancelReminderTool,
    RemindersTool,
    SetReminderTool,
    SetTimerTool,
    build_store,
)
from lucy.tools.voice_enrolment import EnrollVoiceTool, ReplaceVoiceTool
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
    "calendar": lambda s, _transport, _resolver, clock: CalendarTool(
        s.tools.calendar, clock=clock, changes_enabled=_changes_enabled(s)
    ),
    "add_calendar_event": lambda s, _transport, _resolver, clock: AddCalendarEventTool(
        s.tools.calendar, clock=clock, changes_enabled=_changes_enabled(s)
    ),
    "edit_calendar_event": lambda s, _transport, _resolver, clock: EditCalendarEventTool(
        s.tools.calendar, clock=clock
    ),
    "delete_calendar_event": lambda s, _transport, _resolver, clock: DeleteCalendarEventTool(
        s.tools.calendar, clock=clock
    ),
    "reminders": lambda s, _transport, _resolver, clock: RemindersTool(
        s.tools.reminders, store=build_store(s, clock), clock=clock
    ),
    "set_reminder": lambda s, _transport, _resolver, clock: SetReminderTool(
        s.tools.reminders, store=build_store(s, clock), clock=clock
    ),
    "set_timer": lambda s, _transport, _resolver, clock: SetTimerTool(
        s.tools.reminders, store=build_store(s, clock), clock=clock
    ),
    "cancel_reminder": lambda s, _transport, _resolver, clock: CancelReminderTool(
        s.tools.reminders, store=build_store(s, clock), clock=clock
    ),
    "enroll_voice": lambda s, _transport, _resolver, _clock: EnrollVoiceTool(s),
    "replace_voice": lambda s, _transport, _resolver, _clock: ReplaceVoiceTool(s),
    "look": lambda s, _transport, _resolver, _clock: LookTool(s),
    "enroll_face": lambda s, _transport, _resolver, _clock: EnrollFaceTool(s),
    "replace_face": lambda s, _transport, _resolver, _clock: ReplaceFaceTool(s),
}


def _changes_enabled(settings: Settings) -> bool:
    """Whether an edit or delete calendar tool is offered, so calendar results say so."""
    return bool({"edit_calendar_event", "delete_calendar_event"} & set(settings.tools.enabled))


def build_tools(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    resolver: Resolver = system_resolver,
    clock: Clock = lambda: datetime.now(UTC),
    reminder_store: ReminderStore | None = None,
) -> list[Tool]:
    """The tools enabled in `settings`, in the order listed. Empty when none are.

    `transport` and `resolver` replace the network for tests. `clock` is "now" for
    the `time` tool (the app passes its own clock). With `reminder_store`, the
    reminder tools share it and do not close it (the app owns it); without, each
    opens its own.
    """
    tools: list[Tool] = []
    for name in settings.tools.enabled:
        factory = CATALOGUE.get(name)
        if factory is None:
            raise ValueError(f"unknown tool {name!r}; known tools: {', '.join(CATALOGUE)}")
        if reminder_store is not None and name in _REMINDER_CLASSES:
            tools.append(
                _REMINDER_CLASSES[name](
                    settings.tools.reminders, store=reminder_store, clock=clock, owns_store=False
                )
            )
            continue
        tools.append(factory(settings, transport, resolver, clock))
    return tools


_REMINDER_CLASSES: Final = {
    "reminders": RemindersTool,
    "set_reminder": SetReminderTool,
    "set_timer": SetTimerTool,
    "cancel_reminder": CancelReminderTool,
}


# Tool groups for dynamic tool loading (ADR 0025): each built-in tool's group, the
# one line the model is told about it, and the words (English and romanised Hindi)
# in a user message that offer it. `DEFAULT_CORE` are offered on every turn.
GROUPS: Final[tuple[ToolGroup, ...]] = (
    ToolGroup(
        "lookups",
        "Wikipedia, web search and reading a web page",
        ("wikipedia", "web_search", "fetch_page"),
        ("wikipedia", "search", "google", "look up", "web", "internet", "online", "news",
         "website", "page", "dhundh", "dhoondh", "khabar", "samachar"),
    ),
    ToolGroup(
        "time_weather",
        "the time anywhere and weather forecasts",
        ("time", "weather"),
        ("time", "weather", "forecast", "temperature", "rain", "raining", "sunny", "snow",
         "mausam", "baarish", "barish", "garmi", "thand", "samay", "kitne baje", "time zone"),
    ),
    ToolGroup(
        "calendar",
        "the user's calendar: what is on, free time, adding, changing or deleting events",
        ("calendar", "add_calendar_event", "edit_calendar_event", "delete_calendar_event"),
        ("calendar", "meeting", "meetings", "appointment", "appointments", "event", "events",
         "schedule", "scheduled", "agenda", "am i free", "are you free", "busy", "free on",
         "free at", "free tomorrow", "free today", "diary", "booked", "reschedule", "move my",
         "cancel my", "invite", "got on", "have on", "plans", "pencil in", "free hoon", "free hu",
         "free hain", "free rahunga", "free rahungi", "khali", "mulakat", "mulaqat", "milna hai",
         "program hai"),
    ),
    ToolGroup(
        "reminders",
        "reminders and timers: listing, setting and cancelling them",
        ("reminders", "set_reminder", "set_timer", "cancel_reminder"),
        ("remind", "reminder", "reminders", "timer", "timers", "alarm", "countdown",
         "yaad dila", "yaad dilana", "yaad dilao", "yaad kara", "yaad karana", "timer laga",
         "alarm laga", "minute baad", "minutes baad", "ghante baad", "nudge", "ping me", "wake me",
         "wake up", "don't let me forget", "utha dena", "utha do", "uthana", "jaga dena",
         "jaga do", "jagana"),
    ),
    ToolGroup(
        "voice",
        "voice enrolment: learning or re-recording a person's voice",
        ("enroll_voice", "replace_voice"),
        ("enrol", "enroll", "enrolment", "enrollment", "voice", "voices", "recognise", "recognize",
         "awaaz", "aawaz", "awaz", "pehchan", "pehchaan", "pehchano"),
    ),
    ToolGroup(
        "vision",
        "looking through the camera: what is in front of it, what the user is holding, "
        "who is there, and learning a person's face",
        ("look", "enroll_face", "replace_face"),
        # Bare "see" and "look" are left out, as are "see that", "see what", "photo" and
        # "picture": "let me see", "I see that", "look up", "picture this" are everyday
        # speech and would switch the group on constantly.
        ("camera", "webcam", "can you see", "do you see", "what do you see", "you see me",
         "see me", "see this", "look at", "take a look", "have a look",
         "looking at", "what am i holding", "what i'm holding", "what is this", "what's this",
         "what is that", "what's that", "how do i look", "what colour", "what color",
         "showing you", "in front of", "dekho", "dekh", "dekhna", "dekhke",
         "dekh ke", "dekh kar", "dikh", "dikhta", "dikhti", "dikhai", "dikha", "dikhao",
         "kya hai ye", "ye kya hai", "yeh kya hai", "kya hai yeh", "mere haath", "haath mein",
         "kaisa lag", "kaisi lag", "kaun sa rang", "kya pakda", "my face", "remember my face",
         "learn my face", "enrol my face", "enroll my face", "who is here",
         "who's here", "who is there", "who do you see", "recognise me", "recognize me",
         "chehra", "chehre", "kaun hai", "kaun hoon"),
    ),
)  # fmt: skip

DEFAULT_CORE: Final = ("wikipedia", "web_search", "fetch_page", "time", "weather")
