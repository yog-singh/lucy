"""Who a turn is with: the owner or a guest (ADR 0027, run A)."""

from __future__ import annotations

import asyncio
import io
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lucy.app import owner_enrolled, people_policy
from lucy.clients.voice_cli import TerminalUI
from lucy.core.bus import EventBus
from lucy.core.config import DEFAULT_GUEST_TOOLS, PeopleSettings, Settings, load_settings
from lucy.core.context import ContextBuilder
from lucy.core.events import (
    OUTBOUND_EVENTS,
    RECORD_EVENTS,
    AnyEvent,
    AssistantMessage,
    Seen,
    SeenFace,
    TimerFired,
    TurnEnded,
    TurnWith,
    UserMessage,
)
from lucy.core.gateway import LLMGateway
from lucy.core.history import events_to_messages
from lucy.core.orchestrator import Orchestrator
from lucy.core.presence import Presence, PresencePolicy, turn_with
from lucy.core.types import Message, Role
from lucy.core.workspace import Workspace
from lucy.vision.faces import FaceBook
from lucy.voice.session import TurnTimings
from tests.fakes import FakeLLM, InMemoryMemoryStore
from tests.test_app import make_settings
from tests.test_faces import ASHA, RAVI, STRANGER, face
from tests.test_glance import ONE_TURN, GlanceCamera, glancer
from tests.test_voice_session import Harness, RecordingUI, reply

T0 = datetime(2026, 11, 24, 18, 0, tzinfo=UTC)
POLICY = PresencePolicy("Yogesh", carry_minutes=5)


def msg(
    minute: float = 0,
    *,
    source: str = "voice",
    speaker: str | None = None,
    people: tuple[str, ...] = (),
    unknown: int = 0,
    glance: bool | None = None,
    session: str = "s",
) -> UserMessage:
    seen = None
    if people or unknown or glance:
        seen = Seen(people=tuple(SeenFace(name=n, score=0.9) for n in people), unknown=unknown)
    return UserMessage(
        session_id=session,
        text="hi",
        source=source,  # type: ignore[arg-type]
        speaker=speaker,
        seen=seen,
        ts=T0 + timedelta(minutes=minute),
    )


def decide(*earlier: AnyEvent, now: UserMessage) -> Presence:
    return POLICY.decide([*earlier, now], now)


def P(role: str, name: str | None, basis: str) -> Presence:
    return Presence(role=role, name=name, basis=basis)  # type: ignore[arg-type]


# --- The rules, in order ---------------------------------------------------------------


def test_a_typed_is_the_owner_whatever_else() -> None:
    assert decide(now=msg(source="text")) == P("owner", "Yogesh", "typed")
    assert decide(now=msg(source="correction")) == P("owner", "Yogesh", "typed")
    # Typed wins over tags (a typed message has none in practice).
    assert decide(now=msg(source="text", speaker="Asha", people=("Asha",))).basis == "typed"


def test_b_the_voice_tag_decides_before_the_glance() -> None:
    assert decide(now=msg(speaker="Yogesh")) == P("owner", "Yogesh", "voice")
    assert decide(now=msg(speaker="yogesh")) == P("owner", "yogesh", "voice")  # case ignored
    assert decide(now=msg(speaker="Asha")) == P("guest", "Asha", "voice")
    # The tag wins over the glance, both ways.
    assert decide(now=msg(speaker="Asha", people=("Yogesh",))) == P("guest", "Asha", "voice")
    assert decide(now=msg(speaker="Yogesh", unknown=2)) == P("owner", "Yogesh", "voice")


def test_c_the_glance_without_a_voice_tag() -> None:
    assert decide(now=msg(people=("Yogesh",))) == P("owner", "Yogesh", "face")
    assert decide(now=msg(people=("Asha",))) == P("guest", "Asha", "face")
    assert decide(now=msg(unknown=1)) == P("guest", None, "face")
    # Named only with exactly one recognised person and nobody unknown.
    assert decide(now=msg(people=("Asha", "Ravi"))) == P("guest", None, "face")
    assert decide(now=msg(people=("Asha",), unknown=1)) == P("guest", None, "face")


def test_c_the_owner_with_others_decides_nothing() -> None:
    for now in (msg(people=("Yogesh", "Asha")), msg(people=("Yogesh",), unknown=1)):
        assert decide(now=now) == P("guest", None, "none")
        assert decide(msg(-1, source="text"), now=now) == P("owner", "Yogesh", "carried")
    # An empty glance (nobody in view) decides nothing either.
    assert decide(msg(-1, speaker="Asha"), now=msg(glance=True)) == P("guest", "Asha", "carried")


def test_d_nothing_to_go_on_is_a_guest() -> None:
    assert decide(now=msg()) == P("guest", None, "none")


# --- Carrying over ---------------------------------------------------------------------


def test_carry_inside_and_outside_the_window() -> None:
    owner = msg(0, speaker="Yogesh")
    assert decide(owner, now=msg(5)) == P("owner", "Yogesh", "carried")  # exactly 5: kept
    assert decide(owner, now=msg(5.01)) == P("guest", None, "none")
    asha = msg(0, people=("Asha",))
    assert decide(asha, now=msg(2)) == P("guest", "Asha", "carried")


def test_the_latest_real_decision_is_carried() -> None:
    log = [msg(0, speaker="Yogesh"), msg(1, speaker="Asha")]
    assert decide(*log, now=msg(2)) == P("guest", "Asha", "carried")


def test_carried_decisions_do_not_chain_beyond_the_window() -> None:
    """Untagged turns every 2 minutes after one real decision: carried until 5 minutes
    after that decision, never renewed by the carried turns in between."""
    log: list[AnyEvent] = [msg(0, source="text")]
    results = []
    for minute in (2, 4, 6, 8):
        now = msg(minute)
        results.append(POLICY.decide([*log, now], now))
        log.append(now)
    assert [r.basis for r in results] == ["carried", "carried", "none", "none"]
    assert [r.role for r in results] == ["owner", "owner", "guest", "guest"]


def test_an_old_real_decision_stops_the_search() -> None:
    log = [msg(0, speaker="Yogesh"), msg(10, speaker="Asha")]
    assert decide(*log, now=msg(16)) == P("guest", None, "none")


def test_proactive_turns_and_other_sessions_are_not_decisions() -> None:
    timer = TimerFired(session_id="s", name="reminder", ts=T0 + timedelta(minutes=3))
    other = msg(3, source="text", session="other")
    assert decide(timer, other, now=msg(4)) == P("guest", None, "none")
    # A reminder spoken in between does not break a carry either.
    assert decide(msg(0, source="text"), timer, now=msg(4)).basis == "carried"


def test_a_yes_to_a_confirmation_follows_the_same_rules() -> None:
    """Nothing is special about an answer: an untagged "yes" carries over."""
    yes = msg(1).model_copy(update={"text": "yes"})
    assert decide(msg(0, speaker="Yogesh"), now=yes) == P("owner", "Yogesh", "carried")
    assert decide(now=yes) == P("guest", None, "none")


def test_typed_then_voice_in_one_session_carries_the_owner() -> None:
    """The known consequence: a typed message makes untagged voice turns the owner's."""
    assert decide(msg(0, source="text"), now=msg(3)) == P("owner", "Yogesh", "carried")


def test_event_timestamps_not_the_wall_clock() -> None:
    """Recomputed years later from the same log, the answer is the same."""
    old = datetime(2020, 1, 1, tzinfo=UTC)
    first = msg(0, speaker="Yogesh").model_copy(update={"ts": old})
    now = msg().model_copy(update={"ts": old + timedelta(minutes=2)})
    assert POLICY.decide([first, now], now) == P("owner", "Yogesh", "carried")


def test_events_after_the_message_are_ignored_and_the_message_may_be_absent() -> None:
    now = msg(1)
    later = msg(2, source="text")
    assert POLICY.decide([msg(0, speaker="Asha"), now, later], now).name == "Asha"
    assert POLICY.decide([msg(0, speaker="Asha")], now).name == "Asha"


def test_policy_arguments() -> None:
    with pytest.raises(ValueError):
        PresencePolicy(" ")
    with pytest.raises(ValueError):
        PresencePolicy("Yogesh", carry_minutes=-1)
    assert PresencePolicy("Yogesh", carry_minutes=0).decide(
        [msg(0, source="text"), now := msg(0)], now
    ) == P("owner", "Yogesh", "carried")


def test_label_and_record_round_trip() -> None:
    assert P("guest", "Asha", "face").label() == "guest: Asha"
    assert P("guest", None, "none").label() == "guest"
    assert P("owner", "Yogesh", "voice").label() == "owner"
    record = P("guest", "Asha", "voice").record("s", "t1", "m1")
    assert (record.role, record.name, record.basis, record.message_id) == (
        "guest",
        "Asha",
        "voice",
        "m1",
    )
    assert turn_with([record], "t1") == P("guest", "Asha", "voice")
    assert turn_with([record], "t2") is None


# --- The record ------------------------------------------------------------------------


def test_the_record_is_invisible_to_other_consumers() -> None:
    from lucy.memory.graph_store import CONTENT_EPISODE_TYPES

    record = P("guest", None, "none").record("s", "t", "m")
    assert TurnWith in RECORD_EVENTS and TurnWith not in OUTBOUND_EVENTS
    assert "turn_with" not in CONTENT_EPISODE_TYPES
    assert events_to_messages([record, UserMessage(session_id="s", text="hi")]) == [
        Message(role=Role.USER, content="hi")
    ]


class Rig:
    def __init__(self, people: PresencePolicy | None, earlier: list[AnyEvent] = ()) -> None:  # type: ignore[assignment]
        self.bus = EventBus()
        self.llm = FakeLLM([reply("One."), reply("Two."), reply("Three.")])
        self.memory = InMemoryMemoryStore()
        self.memory.episodes.extend(earlier)
        self.traces: list[Any] = []
        rig = self

        class Sink:
            async def record(self, trace: Any) -> None:
                rig.traces.append(trace)

        self.orchestrator = Orchestrator(
            bus=self.bus,
            gateway=LLMGateway({"chat": self.llm}),
            memory=self.memory,
            context=ContextBuilder("You are Lucy.", layout="stable_prefix", clock=None),
            trace=Sink(),  # type: ignore[arg-type]
            people=people,
        )

    async def turns(self, *messages: UserMessage) -> list[Any]:
        task = asyncio.create_task(self.orchestrator.run())
        published: list[Any] = []
        try:
            async with self.bus.subscribe() as events:
                for message in messages:
                    await self.bus.publish(message)
                    async with asyncio.timeout(2):
                        while not isinstance(event := await anext(events), TurnEnded):
                            published.append(event)
        finally:
            self.bus.close()
            await task
        return published


async def test_the_orchestrator_records_each_turn_before_its_message_unpublished() -> None:
    rig = Rig(POLICY)
    first, second = msg(0, speaker="Yogesh"), msg(1)
    published = await rig.turns(first, second)
    assert not any(isinstance(e, TurnWith) for e in published)
    records = [e for e in rig.memory.episodes if isinstance(e, TurnWith)]
    assert [(r.role, r.name, r.basis) for r in records] == [
        ("owner", "Yogesh", "voice"),
        ("owner", "Yogesh", "carried"),
    ]
    for record, message in zip(records, (first, second), strict=True):
        index = rig.memory.episodes.index(record)
        logged = rig.memory.episodes[index + 1]
        assert isinstance(logged, UserMessage) and logged.id == message.id == record.message_id
        assert logged.turn_id == record.turn_id is not None
    assert [t.presence for t in rig.traces] == [
        P("owner", "Yogesh", "voice"),
        P("owner", "Yogesh", "carried"),
    ]
    # Run A changes no prompt: the model saw only the conversation.
    assert [m.role for m in rig.llm.calls[1] if m.role is not Role.SYSTEM] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]


async def test_disabled_records_nothing_and_logs_as_before() -> None:
    def shape(events: list[AnyEvent]) -> list[tuple[str, str | None]]:
        return [(e.type, getattr(e, "text", None)) for e in events]

    logs = []
    for people in (None, POLICY):
        rig = Rig(people)
        await rig.turns(msg(0, speaker="Asha"), msg(1, source="text"))
        logs.append([e for e in rig.memory.episodes if not isinstance(e, TurnWith)])
        if people is None:
            assert not any(isinstance(e, TurnWith) for e in rig.memory.episodes)
            assert all(t.presence is None for t in rig.traces)
            assert await rig.orchestrator.turn_with(msg(2)) is None
        texts = [m.content for m in rig.llm.calls[1]]
        logs.append(texts)  # type: ignore[arg-type]
    off_log, off_prompt, on_log, on_prompt = logs
    assert shape(off_log) == shape(on_log)  # type: ignore[arg-type]
    assert off_prompt == on_prompt


async def test_turn_with_matches_what_the_turn_records() -> None:
    earlier: list[AnyEvent] = [msg(0, people=("Asha",))]
    rig = Rig(POLICY, earlier)
    coming = msg(1)
    assert await rig.orchestrator.turn_with(coming) == P("guest", "Asha", "carried")
    await rig.turns(coming)
    (record,) = [e for e in rig.memory.episodes if isinstance(e, TurnWith)]
    assert Presence.from_record(record) == P("guest", "Asha", "carried")
    assert not any(isinstance(e, TurnWith) for e in earlier)  # nothing stored by the query


async def test_old_logs_load_and_a_record_round_trips(tmp_path: Path) -> None:
    from lucy.memory.sqlite_store import SQLiteMemoryStore

    record = P("guest", "Asha", "face").record("s", "t", "m")
    async with SQLiteMemoryStore(tmp_path / "lucy.db") as store:
        await store.append_episode(UserMessage(session_id="s", text="old"))
        await store.append_episode(record)
        await store.append_episode(AssistantMessage(session_id="s", text="ok"))
        events = await store.recent_episodes("s", 10)
    assert [type(e) for e in events] == [UserMessage, TurnWith, AssistantMessage]
    assert events[1] == record


# --- Config ----------------------------------------------------------------------------


def test_people_settings_defaults_and_validation() -> None:
    assert PeopleSettings() == PeopleSettings(
        enabled=False, owner=None, carry_minutes=5, guest_tools=list(DEFAULT_GUEST_TOOLS)
    )
    assert DEFAULT_GUEST_TOOLS == (
        "wikipedia",
        "web_search",
        "fetch_page",
        "time",
        "weather",
        "set_timer",
        "look",
    )
    assert PeopleSettings(enabled=True, owner="  yogesh ").owner == "Yogesh"
    for bad, words in (
        ({"enabled": True}, "has no owner"),
        ({"owner": "my voice"}, "not a usable name"),
        ({"owner": "a/b"}, "not a usable name"),
        ({"guest_tools": ["calendar", "nope"]}, "unknown tool"),
        ({"guest_tools": ["time", "time"]}, "twice"),
        ({"carry_minutes": -1}, "greater than or equal"),
    ):
        with pytest.raises(ValidationError, match=words):
            PeopleSettings(**bad)


def test_the_example_documents_people_off(tmp_path: Path) -> None:
    from tests.test_config import EXAMPLE

    assert load_settings(EXAMPLE).people == PeopleSettings()
    text = EXAMPLE.read_text()
    block = text.split("# [people]", 1)[1].split("\n\n", 1)[0]
    uncommented = "\n".join(
        line[2:] for line in block.splitlines() if line.startswith("# ") and " = " in line
    )
    path = tmp_path / "people.toml"
    path.write_text('[llm.roles]\nchat = { adapter = "x" }\n[people]\n' + uncommented + "\n")
    people = load_settings(path).people
    assert people.enabled and people.owner == "Asha" and people.carry_minutes == 5
    assert people.guest_tools == list(DEFAULT_GUEST_TOOLS)


def _settings(tmp_path: Path, **people: Any) -> Settings:
    base = make_settings(tmp_path / "lucy.db")
    return base.model_copy(
        update={
            "people": PeopleSettings(**people),
            "workspace": base.workspace.model_copy(update={"path": str(tmp_path / "ws")}),
        }
    )


def test_policy_off_and_the_unenrolled_owner_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert people_policy(_settings(tmp_path)) is None
    settings = _settings(tmp_path, enabled=True, owner="Yogesh", carry_minutes=2)
    with caplog.at_level(logging.WARNING):
        policy = people_policy(settings)
    assert policy is not None and policy.owner == "Yogesh"
    assert "enrolled neither as a voice nor as a face" in caplog.text
    assert not (tmp_path / "ws").exists()  # the check only reads


def test_owner_enrolled_by_voice_or_face(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True, owner="Yogesh")
    assert not owner_enrolled(settings, "Yogesh")
    Workspace(tmp_path / "ws").dir("speakers", "yogesh")
    assert owner_enrolled(settings, "Yogesh")
    FaceBook(Workspace(tmp_path / "ws")).enroll("Asha", [ASHA])
    assert owner_enrolled(settings, "asha")
    assert not owner_enrolled(settings, "Ravi")


# --- Showing it ------------------------------------------------------------------------


class MarkingUI(RecordingUI):
    def __init__(self) -> None:
        super().__init__()
        self.guests: list[tuple[str, str | None, str]] = []

    def user_guest(self, text: str, speaker: str | None, label: str) -> None:
        self.guests.append((text, speaker, label))


async def test_a_guest_turn_is_marked_in_voice_and_timed(tmp_path: Path) -> None:
    g, _ = glancer(tmp_path, GlanceCamera(), [face(ASHA)])
    ui = MarkingUI()
    policy = PresencePolicy("Ravi")
    async with Harness(ONE_TURN, [reply("Hi.")], ["hi lucy"], glancer=g, people=policy, ui=ui) as h:
        await h.run()
    assert ui.guests == [("hi lucy", None, "guest: Asha")] and ui.users == []
    timing = h.session.timings[0]
    assert (timing.turn_with, timing.turn_with_name, timing.turn_with_basis) == (
        "guest",
        "Asha",
        "face",
    )
    assert timing.to_json()["turn_with"] == "guest"
    (record,) = h.events(TurnWith)
    assert record.name == "Asha"


async def test_an_owner_turn_is_not_marked(tmp_path: Path) -> None:
    g, _ = glancer(tmp_path, GlanceCamera(), [face(RAVI)])
    ui = MarkingUI()
    policy = PresencePolicy("Ravi")
    async with Harness(ONE_TURN, [reply("Hi.")], ["hi lucy"], glancer=g, people=policy, ui=ui) as h:
        await h.run()
    assert ui.guests == [] and ui.users == ["hi lucy"]
    assert h.session.timings[0].turn_with == "owner"


async def test_without_people_nothing_is_marked_or_timed(tmp_path: Path) -> None:
    g, _ = glancer(tmp_path, GlanceCamera(), [face(STRANGER)])
    ui = MarkingUI()
    async with Harness(ONE_TURN, [reply("Hi.")], ["hi lucy"], glancer=g, ui=ui) as h:
        await h.run()
    assert ui.guests == [] and ui.users == ["hi lucy"]
    assert h.session.timings[0].turn_with is None
    assert h.events(TurnWith) == []


def test_terminal_marker() -> None:
    out, err = io.StringIO(), io.StringIO()
    ui = TerminalUI(out, err, show_state=False)
    ui.user_guest("what time is it", None, "guest")
    ui.user_guest("hello", "Asha", "guest: Asha")
    assert out.getvalue() == ("you> what time is it  (guest)\nyou (Asha)> hello  (guest: Asha)\n")

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    tty = Tty()
    TerminalUI(tty, err, show_state=False).user_guest("hi", None, "guest")
    assert tty.getvalue() == "you> hi  \x1b[2m(guest)\x1b[0m\n"


def test_timings_default_has_no_presence() -> None:
    timing = TurnTimings(
        session_id="s", at="", speech_start=0, speech_end=1, endpoint=1, utterance_s=1
    )
    data = timing.to_json()
    assert data["turn_with"] is None and data["turn_with_basis"] is None
