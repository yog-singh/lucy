"""Recording the configured tools and leaving out exchanges a newly configured tool
made stale (ADR 0025, amendment 2026-11-17)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from lucy.app import tool_groups
from lucy.core.bus import EventBus
from lucy.core.config import McpServerSettings, McpSettings, ToolsSettings
from lucy.core.context import ContextBuilder
from lucy.core.events import (
    OUTBOUND_EVENTS,
    AnyEvent,
    AssistantMessage,
    ConfirmationResolved,
    TimerFired,
    ToolFinished,
    ToolsConfigured,
    TurnEnded,
    UserMessage,
)
from lucy.core.gateway import LLMGateway
from lucy.core.history import events_to_messages
from lucy.core.orchestrator import Orchestrator
from lucy.core.toolchanges import ToolChanges
from lucy.core.toolsets import ToolGroup, ToolSelector
from lucy.core.types import Message, Role, ToolCall, ToolResult
from tests.fakes import FakeLLM, InMemoryMemoryStore
from tests.test_app import make_settings
from tests.test_toolsets import Stub, reply

VISION = ToolGroup("vision", "the camera", ("look", "enroll_face"), ("my face", "see me"))
WEB = ToolGroup("web", "lookups", ("web_search",), ("search",))
GROUPS = [WEB, VISION]
REFUSAL = "I cannot remember your face."


def record(*names: str) -> ToolsConfigured:
    return ToolsConfigured(session_id="s", names=names)


def user(text: str, turn: str | None = None) -> UserMessage:
    return UserMessage(session_id="s", text=text, turn_id=turn)


def said(text: str, turn: str | None = None, *tool_calls: ToolCall) -> AssistantMessage:
    return AssistantMessage(session_id="s", text=text, turn_id=turn, tool_calls=list(tool_calls))


def roles_and_texts(events: list[AnyEvent]) -> list[tuple[str, str]]:
    return [(m.role.value, m.content) for m in events_to_messages(events)]


NOW = ToolChanges(["web_search", "look", "enroll_face"], GROUPS)


# --- Recording -------------------------------------------------------------------------


def test_record_on_first_turn_then_only_when_the_set_changes() -> None:
    first = NOW.record([], "s", "t1")
    assert first is not None and first.names == ("web_search", "look", "enroll_face")
    assert first.turn_id == "t1"
    same_set = record("enroll_face", "look", "web_search")  # order does not matter
    assert NOW.record([user("hi"), same_set, said("hello")], "s", "t2") is None
    changed = NOW.record([record("web_search", "look"), user("hi")], "s", "t2")
    assert changed is not None and set(changed.names) == {"web_search", "look", "enroll_face"}
    # Only the latest record counts.
    assert NOW.record([record("look"), record(*NOW.names)], "s", "t") is None
    assert NOW.record([record(*NOW.names), record("look")], "s", "t") is not None


def test_no_record_without_tools_unless_one_was_recorded() -> None:
    none = ToolChanges([], GROUPS)
    assert none.record([user("hi")], "s", "t") is None
    removed = none.record([record("look")], "s", "t")
    assert removed is not None and removed.names == ()


def test_the_record_is_not_a_message_nor_content_to_extract() -> None:
    from lucy.memory.graph_store import CONTENT_EPISODE_TYPES

    assert "tools_configured" not in CONTENT_EPISODE_TYPES
    assert ToolsConfigured not in OUTBOUND_EVENTS
    assert events_to_messages([record("look"), user("hi")]) == [
        Message(role=Role.USER, content="hi")
    ]


# --- Staleness -------------------------------------------------------------------------


def stale_log(before: tuple[str, ...] = ("web_search", "look")) -> list[AnyEvent]:
    return [
        record(*before),
        user("Can you remember my face?", "t1"),
        said(REFUSAL, "t1"),
        TurnEnded(session_id="s", turn_id="t1", reason="completed"),
        user("What is the time?", "t2"),
        said("Ten.", "t2"),
        record(*NOW.names),
        user("Remember my face.", "t3"),
    ]


def test_a_refusal_before_the_tool_existed_is_left_out() -> None:
    kept, count = NOW.leave_out_stale(stale_log())
    assert count == 1
    assert roles_and_texts(kept) == [
        ("user", "What is the time?"),
        ("assistant", "Ten."),
        ("user", "Remember my face."),
    ]
    assert sum(isinstance(e, ToolsConfigured) for e in kept) == 2  # records stay


def test_a_keeps_an_exchange_whose_message_switches_no_group_on() -> None:
    log = stale_log()
    log[1] = user("Can you remember my name?", "t1")
    assert NOW.leave_out_stale(log)[1] == 0


def test_b_keeps_an_exchange_when_the_group_gained_no_tool() -> None:
    assert NOW.leave_out_stale(stale_log(before=NOW.names))[1] == 0
    # A new tool in another group does not make a vision exchange stale.
    assert NOW.leave_out_stale(stale_log(before=("look", "enroll_face")))[1] == 0


def test_c_keeps_an_exchange_whose_turn_called_a_tool() -> None:
    log = stale_log()
    look = ToolCall(id="c1", name="look", arguments={})
    log[2:2] = [
        said("", "t1", look),
        ToolFinished(session_id="s", turn_id="t1", result=ToolResult(call_id="c1", content="a")),
    ]
    assert NOW.leave_out_stale(log)[1] == 0
    resolved = stale_log()
    resolved.insert(
        2, ConfirmationResolved(session_id="s", turn_id="t1", request_id="r", outcome="declined")
    )
    assert NOW.leave_out_stale(resolved)[1] == 0


def test_legacy_history_without_records_is_untouched() -> None:
    log = [e for e in stale_log() if not isinstance(e, ToolsConfigured)]
    assert NOW.leave_out_stale(log) == (log, 0)
    # A record written after legacy history: the time before it counts as "now".
    log.insert(len(log) - 1, record(*NOW.names))
    assert NOW.leave_out_stale(log)[1] == 0


def test_the_latest_user_message_is_never_left_out() -> None:
    log = [record("web_search"), user("Remember my face.", "t1")]
    assert NOW.leave_out_stale(log) == (log, 0)
    # As the pre-fill sees it, with its placeholder last: the earlier message can go.
    placeholder = user("")
    log2 = [*log, said(REFUSAL, "t1"), placeholder]
    kept, count = NOW.leave_out_stale(log2)
    assert count == 1 and roles_and_texts(kept) == [("user", "")]


def test_the_span_ends_at_a_timer_so_a_reminder_stays() -> None:
    log = stale_log()
    log[4:4] = [
        TimerFired(session_id="s", turn_id="t9", name="reminder"),
        said("Time to stretch.", "t9"),
    ]
    kept, count = NOW.leave_out_stale(log)
    assert count == 1
    assert ("assistant", "Time to stretch.") in roles_and_texts(kept)


def test_mcp_server_group_from_app_settings(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "lucy.db")
    settings = settings.model_copy(
        update={
            "mcp": McpSettings(
                servers={"gmail": McpServerSettings(command="x", group_keywords=["inbox"])}
            )
        }
    )
    mail = Stub("search_email")
    mail.server = "gmail"  # type: ignore[attr-defined]
    groups = tool_groups(settings, [mail])
    changes = ToolChanges(["search_email"], groups)
    log: list[AnyEvent] = [
        record(),
        user("Anything new in my inbox?", "t1"),
        said("I cannot read your email.", "t1"),
        record("search_email"),
        user("Check my inbox.", "t2"),
    ]
    kept, count = changes.leave_out_stale(log)
    assert count == 1 and roles_and_texts(kept) == [("user", "Check my inbox.")]


# --- In the orchestrator ----------------------------------------------------------------


class Rig:
    def __init__(
        self, scripts: list[list[Any]], tl: list[Stub], *, dynamic: bool, earlier: list[AnyEvent]
    ) -> None:
        self.bus = EventBus()
        self.llm = FakeLLM(scripts)
        self.memory = InMemoryMemoryStore()
        self.memory.episodes.extend(earlier)
        sel = ToolSelector(tl, GROUPS, core=["web_search"]) if dynamic else None
        self.orchestrator = Orchestrator(
            bus=self.bus,
            gateway=LLMGateway({"chat": self.llm}),
            memory=self.memory,
            context=ContextBuilder("You are Lucy.", layout="stable_prefix", clock=None),
            tools=tl,
            selector=sel,
            tool_groups=GROUPS,
        )

    async def turns(self, *texts: str, prefix_before: int | None = None) -> Any:
        task = asyncio.create_task(self.orchestrator.run())
        prefix = None
        try:
            async with self.bus.subscribe(TurnEnded) as ends:
                for i, text in enumerate(texts):
                    if i == prefix_before:
                        prefix = await self.orchestrator.next_turn_prefix("s")
                    await self.bus.publish(UserMessage(session_id="s", text=text))
                    async with asyncio.timeout(2):
                        await anext(ends)
        finally:
            self.bus.close()
            await task
        return prefix


EARLIER: list[AnyEvent] = [
    record("web_search", "look"),
    user("Can you remember my face?", "t0"),
    said(REFUSAL, "t0"),
    TurnEnded(session_id="s", turn_id="t0", reason="completed"),
]


def user_texts(sent: list[Message]) -> list[str]:
    return [m.content for m in sent if m.role is Role.USER]


async def run_stale(dynamic: bool) -> Rig:
    tl = [Stub("web_search"), Stub("look"), Stub("enroll_face")]
    rig = Rig([reply("What's your name?"), reply("Done.")], tl, dynamic=dynamic, earlier=EARLIER)
    await rig.turns("Remember my face.", "Asha.")
    return rig


async def test_orchestrator_records_and_leaves_the_stale_exchange_out_dynamic_on_and_off() -> None:
    for dynamic in (True, False):
        rig = await run_stale(dynamic)
        first, second = rig.llm.calls
        assert user_texts(first) == ["Remember my face."]
        # The exchange after the change stays: it was answered with the tool there.
        assert user_texts(second) == ["Remember my face.", "Asha."]
        records = [e for e in rig.memory.episodes if isinstance(e, ToolsConfigured)]
        assert [r.names for r in records] == [
            ("web_search", "look"),
            ("web_search", "look", "enroll_face"),
        ]
        index = rig.memory.episodes.index(records[1])
        assert isinstance(rig.memory.episodes[index + 1], UserMessage)  # before the message
        # The log keeps the refusal.
        assert any(
            isinstance(e, AssistantMessage) and e.text == REFUSAL for e in rig.memory.episodes
        )


async def test_the_trace_counts_stale_exchanges() -> None:
    tl = [Stub("web_search"), Stub("look"), Stub("enroll_face")]
    rig = Rig([reply("Name?")], tl, dynamic=False, earlier=EARLIER)
    traces: list[Any] = []

    class Sink:
        async def record(self, trace: Any) -> None:
            traces.append(trace)

    rig.orchestrator._trace = Sink()  # type: ignore[assignment]
    await rig.turns("Remember my face.")
    (trace,) = traces
    stale = [s for s in trace.context if s.name == "stale_exchanges"]
    assert len(stale) == 1 and stale[0].dropped == 1


async def test_prefix_matches_the_next_turn_with_a_stale_exchange() -> None:
    """The pre-fill leaves out the same exchanges as the next turn, after a restart
    that added a tool (no record of the new set yet) and after a normal turn."""
    tl = [Stub("web_search"), Stub("look"), Stub("enroll_face")]
    rig = Rig([reply("Name?"), reply("Done.")], tl, dynamic=True, earlier=EARLIER)
    prefix = await rig.turns("Remember my face.", "Asha.", prefix_before=0)
    assert prefix is not None
    sent = rig.llm.calls[0]
    assert sent[: len(prefix.messages)] == prefix.messages
    assert sent[len(prefix.messages)] == Message(role=Role.USER, content="Remember my face.")
    assert prefix.tools == rig.llm.tools_seen[0]
    rig2 = Rig([reply("Name?"), reply("Done.")], tl, dynamic=True, earlier=EARLIER)
    prefix2 = await rig2.turns("Remember my face.", "Asha.", prefix_before=1)
    assert prefix2 is not None
    sent2 = rig2.llm.calls[1]
    assert sent2[: len(prefix2.messages)] == prefix2.messages
    assert sent2[len(prefix2.messages)] == Message(role=Role.USER, content="Asha.")


async def test_the_record_is_never_published_nor_a_message() -> None:
    tl = [Stub("web_search")]
    bus = EventBus()
    memory = InMemoryMemoryStore()
    llm = FakeLLM([reply("Hi."), reply("Again.")])
    orchestrator = Orchestrator(
        bus=bus,
        gateway=LLMGateway({"chat": llm}),
        memory=memory,
        context=ContextBuilder("p", layout="stable_prefix", clock=None),
        tools=tl,
        tool_groups=GROUPS,
    )
    task = asyncio.create_task(orchestrator.run())
    seen: list[Any] = []
    try:
        async with bus.subscribe(ToolsConfigured, TurnEnded) as events:
            for text in ("hi", "hi again"):
                await bus.publish(UserMessage(session_id="s", text=text))
                async with asyncio.timeout(2):
                    while not isinstance(event := await anext(events), TurnEnded):
                        seen.append(event)
    finally:
        bus.close()
        await task
    assert seen == []
    assert sum(isinstance(e, ToolsConfigured) for e in memory.episodes) == 1
    assert [m.role for m in llm.calls[1] if m.role is not Role.SYSTEM] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]


def test_tool_settings_do_not_matter_for_the_groups(tmp_path: Path) -> None:
    base = make_settings(tmp_path / "lucy.db")
    for dynamic in (True, False):
        s = base.model_copy(
            update={"tools": ToolsSettings(enabled=["time", "calendar"], dynamic=dynamic)}
        )
        assert any(g.name == "calendar" for g in tool_groups(s, [Stub("time"), Stub("calendar")]))
