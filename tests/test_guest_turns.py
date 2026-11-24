"""A guest's turn is restricted in code (ADR 0027, run B)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from lucy.core.bus import EventBus
from lucy.core.config import PeopleSettings
from lucy.core.context import ContextBuilder
from lucy.core.events import (
    AnyEvent,
    AssistantMessage,
    ConfirmationRequested,
    ConfirmationResolved,
    TimerFired,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    TurnWith,
    UserMessage,
)
from lucy.core.gateway import LLMGateway
from lucy.core.orchestrator import Orchestrator
from lucy.core.presence import PresencePolicy, guest_line, guest_run
from lucy.core.toolsets import META_TOOL, ToolGroup, ToolSelector
from lucy.core.types import (
    FinishReason,
    LLMChunk,
    MemoryContext,
    Risk,
    Role,
    ToolCall,
    ToolOutput,
)
from lucy.memory.extraction import Extractor
from tests.fakes import FakeLLM, InMemoryMemoryStore
from tests.test_orchestrator import ConfirmingTool, RecordingTool

T0 = datetime(2026, 11, 24, 18, 0, tzinfo=UTC)
POLICY = PresencePolicy("Yogesh", carry_minutes=5)
GUEST_TOOLS = ("web_search", "time", "set_timer")
PROFILE = "Yogesh lives in Pune and works on Lucy."
MEMORY_LINE = "Yogesh's sister is Asha."
INSTRUCTION = "Call the user Yogi."
EMAIL = "Subject: salary slip for November"


def text(*parts: str) -> list[LLMChunk]:
    return [*(LLMChunk(text=p) for p in parts), LLMChunk(finish_reason=FinishReason.STOP)]


def calls(*tool_calls: ToolCall) -> list[LLMChunk]:
    chunks = [LLMChunk(tool_call=c) for c in tool_calls]
    return [*chunks, LLMChunk(finish_reason=FinishReason.TOOL_CALLS)]


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def msg(minute: float, words: str = "hi", **kw: Any) -> UserMessage:
    return UserMessage(
        session_id="s",
        text=words,
        source=kw.pop("source", "voice"),
        ts=T0 + timedelta(minutes=minute),
        **kw,
    )


class Memory(InMemoryMemoryStore):
    """Retrieval returns the owner's profile, instructions and a memory, and counts."""

    def __init__(self) -> None:
        super().__init__()
        self.retrievals: list[str] = []

    async def retrieve(self, query: str, budget: int) -> MemoryContext:
        self.retrievals.append(query)
        return MemoryContext(profile=PROFILE, instructions=[INSTRUCTION], statements=[MEMORY_LINE])


class EmailTool(RecordingTool):
    """An MCP-like private tool (ADR 0022)."""

    private_source = "gmail"

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        self.runs.append(args)
        return ToolOutput(content=EMAIL)


def toolset() -> dict[str, RecordingTool]:
    return {
        "web_search": RecordingTool("web_search", Risk.READ),
        "time": RecordingTool("time", Risk.READ),
        "set_timer": RecordingTool("set_timer", Risk.ACT),
        "calendar": RecordingTool("calendar", Risk.READ),
        "reminders": RecordingTool("reminders", Risk.READ),
        "enroll_voice": RecordingTool("enroll_voice", Risk.ACT),
        "enroll_face": RecordingTool("enroll_face", Risk.ACT),
        "search_email": EmailTool("search_email", Risk.READ),
        "delete_event": ConfirmingTool("delete_event"),
    }


GROUPS = [
    ToolGroup("web", "lookups", ("web_search",), ("search",)),
    ToolGroup("calendar", "the calendar", ("calendar", "delete_event"), ("calendar",)),
    ToolGroup("reminders", "reminders", ("reminders", "set_timer"), ("timer", "remind")),
    ToolGroup("voice", "voices", ("enroll_voice", "enroll_face"), ("enrol",)),
    ToolGroup("gmail", "email", ("search_email",), ("email",)),
]


class Rig:
    def __init__(
        self,
        scripts: list[list[LLMChunk]],
        *,
        people: PresencePolicy | None = POLICY,
        dynamic: bool = False,
        earlier: list[AnyEvent] | None = None,
    ) -> None:
        self.bus = EventBus()
        self.llm = FakeLLM(scripts)
        self.memory = Memory()
        self.memory.episodes.extend(earlier or [])
        self.tools = toolset()
        tools = list(self.tools.values())
        selector = ToolSelector(tools, GROUPS, core=["time"]) if dynamic else None
        self.orchestrator = Orchestrator(
            bus=self.bus,
            gateway=LLMGateway({"chat": self.llm}),
            memory=self.memory,
            context=ContextBuilder("You are Lucy.", layout="stable_prefix", clock=None),
            tools=tools,
            selector=selector,
            tool_groups=GROUPS,
            people=people,
            guest_tools=GUEST_TOOLS,
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

    def prompt(self, index: int = -1) -> str:
        return "\n".join(m.content for m in self.llm.calls[index])

    def results(self) -> list[str]:
        return [e.result.content for e in self.memory.episodes if isinstance(e, ToolFinished)]


def _soon(message: UserMessage) -> UserMessage:
    """`message` moved to now: a confirmation is answered only after its turn ended."""
    now = datetime.now(UTC)
    return message.model_copy(update={"ts": now + (message.ts - T0)})


def owner(minute: float, words: str = "hi") -> UserMessage:
    return msg(minute, words, speaker="Yogesh")


def guest(minute: float, words: str = "hi", name: str = "Asha") -> UserMessage:
    return msg(minute, words, speaker=name)


# --- The prompt ------------------------------------------------------------------------


async def test_a_guest_prompt_has_nothing_of_the_owner() -> None:
    rig = Rig(
        [
            calls(call("search_email")),
            text("You have a salary slip."),
            text("Hello Asha."),
        ]
    )
    await rig.turns(owner(0, "any new email?"), guest(1, "what did he ask you?"))
    assert rig.tools["search_email"].runs == [{}]  # the owner's turn used it
    guest_prompt = rig.prompt(-1)
    for owners in (PROFILE, MEMORY_LINE, INSTRUCTION, EMAIL, "any new email", "salary slip"):
        assert owners not in guest_prompt
    assert "what did he ask you?" in guest_prompt
    # Retrieval was not even asked for the guest's turn.
    assert rig.memory.retrievals == ["any new email?"]
    # The line in the per-turn block: who, what Lucy can do, what is the owner's.
    note = rig.llm.calls[-1][-1]
    assert note.role is Role.SYSTEM
    assert note.content.startswith("You are talking with a guest, Asha, not Yogesh.")
    assert "look things up, tell the time and set a timer" in note.content
    assert rig.llm.calls[-1][0].content == "You are Lucy."  # the stable prefix: persona only


async def test_an_owner_prompt_keeps_everything_including_guest_exchanges() -> None:
    rig = Rig([text("Hello Asha."), text("Asha said hi.")])
    await rig.turns(guest(0, "I am Asha"), owner(1, "who was that?"))
    prompt = rig.prompt(-1)
    for kept in (PROFILE, MEMORY_LINE, INSTRUCTION, "I am Asha", "Hello Asha."):
        assert kept in prompt
    assert "talking with a guest" not in prompt


async def test_only_the_current_guest_run_is_history() -> None:
    rig = Rig([text("a"), text("b"), text("c"), text("d")])
    await rig.turns(
        guest(0, "first guest words"),
        owner(1, "owner words"),
        guest(2, "second run, one"),
        guest(3, "second run, two"),
    )
    prompt = rig.prompt(-1)
    assert "second run, one" in prompt and "second run, two" in prompt
    assert "first guest words" not in prompt and "owner words" not in prompt


def test_the_run_ends_at_an_owner_or_another_guest_and_skips_proactive_turns() -> None:
    def rec(role: str, name: str | None, turn: str) -> TurnWith:
        return TurnWith(
            session_id="s", turn_id=turn, role=role, name=name, basis="voice", message_id="m"
        )  # type: ignore[arg-type]

    timer = TimerFired(session_id="s", name="reminder", turn_id="p")
    log: list[AnyEvent] = [rec("guest", "Asha", "1"), rec("owner", "Yogesh", "2")]
    log += [rec("guest", "asha", "3"), timer, rec("guest", "Asha", "4")]
    assert guest_run(log, "Asha") == {"3", "4"}
    assert guest_run([*log, rec("guest", None, "5")], None) == {"5"}
    assert guest_run([*log, rec("owner", "Yogesh", "6")], "Asha") == set()


def test_the_guest_line_lists_only_configured_capabilities() -> None:
    line = guest_line("Yogesh", None, ["weather"])
    assert line.startswith("You are talking with a guest not Yogesh. You can tell the weather.")
    assert "look things up" not in line
    assert "camera" in guest_line("Yogesh", None, ["look"])
    assert "You can" not in guest_line("Yogesh", None, [])


# --- Tools -----------------------------------------------------------------------------


async def test_a_guest_is_offered_only_guest_tools_and_refused_the_rest() -> None:
    owner_only = ("calendar", "reminders", "search_email", "enroll_voice", "enroll_face")
    rig = Rig(
        [
            calls(*(call(n, f"c{i}") for i, n in enumerate(owner_only)), call("time", "t")),
            text("Sorry, that is only for Yogesh."),
        ]
    )
    await rig.turns(guest(0, "what is on his calendar?"))
    assert sorted(s.name for s in rig.llm.tools_seen[0]) == sorted(GUEST_TOOLS)
    for name in owner_only:
        assert rig.tools[name].runs == [], name
    assert rig.tools["time"].runs == [{}]
    refusals = [r for r in rig.results() if "only available to Yogesh" in r]
    assert len(refusals) == len(owner_only)
    started = [e.call.name for e in rig.memory.episodes if isinstance(e, ToolStarted)]
    assert started == ["time"]


async def test_a_guest_cannot_reach_a_dangerous_tool_or_its_question() -> None:
    rig = Rig([calls(call("delete_event")), text("I can't.")])
    await rig.turns(guest(0, "delete his meeting"))
    assert not any(isinstance(e, ConfirmationRequested) for e in rig.memory.episodes)
    assert rig.tools["delete_event"].previews == []  # type: ignore[attr-defined]


async def test_more_tools_offers_and_switches_on_only_guest_groups() -> None:
    rig = Rig(
        [
            calls(call(META_TOOL, "m1", group="calendar"), call(META_TOOL, "m2", group="web")),
            calls(call("calendar", "c2")),
            text("Only for Yogesh."),
        ],
        dynamic=True,
    )
    await rig.turns(guest(0, "check the calendar"))
    first, second = ([s.name for s in specs] for specs in rig.llm.tools_seen[:2])
    assert "calendar" not in first and "calendar" not in second
    meta = next(s for s in rig.llm.tools_seen[0] if s.name == META_TOOL)
    assert meta.parameters["properties"]["group"]["enum"] == ["web", "reminders"]
    assert "gmail" not in meta.description and "calendar" not in meta.description
    assert "web_search" in second and "set_timer" not in second
    assert rig.results()[0].startswith("No such group")
    assert rig.tools["calendar"].runs == []


async def test_the_owner_keeps_every_tool_and_the_risk_tiers() -> None:
    rig = Rig([calls(call("calendar"), call("delete_event", "d")), text("ok")])
    await rig.turns(owner(0, "calendar"))
    assert rig.tools["calendar"].runs == [{}]
    assert len(rig.llm.tools_seen[0]) == len(rig.tools)
    assert any(isinstance(e, ConfirmationRequested) for e in rig.memory.episodes)


# --- Confirmations ---------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["yes", "what's the weather?"])
async def test_a_guest_cannot_answer_the_owners_question(answer: str) -> None:
    rig = Rig([calls(call("delete_event", meeting="standup")), text("It is sunny.")])
    published = await rig.turns(_soon(owner(0, "delete standup")), _soon(guest(0.5, answer)))
    tool = rig.tools["delete_event"]
    assert tool.runs == []
    (resolved,) = [e for e in rig.memory.episodes if isinstance(e, ConfirmationResolved)]
    assert resolved.outcome == "not_owner"
    said = [e.text for e in published if isinstance(e, AssistantMessage) and e.text]
    if answer == "yes":
        assert said[-1] == "Only Yogesh can answer that, so nothing was changed."
        assert len(rig.llm.calls) == 1  # no model call for the guest's turn
    else:
        assert "standup" not in rig.prompt(-1)  # the owner's call is not in it
    # Nothing of the owner's call was logged in the guest's turn.
    guest_turn = resolved.turn_id
    assert not any(
        isinstance(e, AssistantMessage) and e.tool_calls and e.turn_id == guest_turn
        for e in rig.memory.episodes
    )


async def test_the_owner_still_confirms() -> None:
    rig = Rig([calls(call("delete_event")), text("Done.")])
    await rig.turns(_soon(owner(0, "delete it")), _soon(owner(0.5, "yes")))
    assert rig.tools["delete_event"].runs == [{}]


# --- Off, and the prefix ---------------------------------------------------------------


async def test_disabled_is_as_before_even_for_a_stranger() -> None:
    rig = Rig([calls(call("calendar")), text("ok")], people=None)
    await rig.turns(guest(0, "calendar"))
    assert rig.tools["calendar"].runs == [{}]
    assert PROFILE in rig.prompt(0)
    assert not any(isinstance(e, TurnWith) for e in rig.memory.episodes)


async def test_the_prefix_follows_the_latest_decision() -> None:
    rig = Rig([text("one"), text("two"), text("three")])
    await rig.turns(owner(0, "owner words"))
    owner_prefix = await rig.orchestrator.next_turn_prefix("s")
    assert owner_prefix is not None and PROFILE in owner_prefix.messages[0].content
    assert len(owner_prefix.tools) == len(rig.tools)

    rig2 = Rig([text("one"), text("two")])
    await rig2.turns(owner(0, "owner words"), guest(1, "guest words"))
    prefix = await rig2.orchestrator.next_turn_prefix("s")
    assert prefix is not None
    joined = "\n".join(m.content for m in prefix.messages)
    assert "guest words" in joined and "owner words" not in joined and PROFILE not in joined
    assert sorted(s.name for s in prefix.tools) == sorted(GUEST_TOOLS)
    # The guest's next turn begins exactly with that prefix.
    rig3 = Rig([text("one"), text("two"), text("three")])
    await rig3.turns(owner(0, "owner words"), guest(1, "guest words"), guest(2, "more"))
    turn = rig3.llm.calls[-1]
    assert turn[: len(prefix.messages)] == prefix.messages
    assert rig3.llm.tools_seen[-1] == prefix.tools


# --- Config and learning ---------------------------------------------------------------


def test_guest_tools_may_never_include_enrolment() -> None:
    for name in ("enroll_voice", "replace_voice", "enroll_face", "replace_face"):
        with pytest.raises(ValidationError, match="only the owner"):
            PeopleSettings(guest_tools=["time", name])


async def test_extraction_skips_guest_turns_and_hides_them_as_context(tmp_path: Any) -> None:
    from lucy.memory.graph_store import SQLiteGraphStore
    from lucy.memory.sqlite_store import SQLiteMemoryStore
    from tests.graph_support import HashingEmbedder

    path = tmp_path / "lucy.db"
    async with SQLiteMemoryStore(path) as store:
        for turn, role, name, words in (
            ("g1", "guest", "Asha", "my phone number is 98765"),
            ("o1", "owner", "Yogesh", "I like tea"),
        ):
            message = UserMessage(session_id="s", text=words, turn_id=turn, source="voice")
            await store.append_episode(
                TurnWith(
                    session_id="s",
                    turn_id=turn,
                    role=role,  # type: ignore[arg-type]
                    name=name,
                    basis="voice",
                    message_id=message.id,
                )
            )
            await store.append_episode(message)
            await store.append_episode(AssistantMessage(session_id="s", turn_id=turn, text="ok"))
            await store.append_episode(TurnEnded(session_id="s", turn_id=turn, reason="completed"))
    async with SQLiteGraphStore(path, HashingEmbedder()) as graph:
        gateway = LLMGateway({"extraction": FakeLLM([])})
        extractor = Extractor(graph, gateway)
        (first,) = await extractor.ready_turns(1)
        report = await extractor.extract_turn(first)
        assert report.skipped == "a guest's turn" and report.extraction is None
        (second,) = await extractor.ready_turns(1)
        messages = await extractor.build_messages(second)
        assert "98765" not in messages[-1].content and "I like tea" in messages[-1].content
