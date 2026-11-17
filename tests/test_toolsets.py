"""Dynamic tool loading (ADR 0025): selection, stability, the pre-fill property,
confirmation of a tool not offered, settings and feature-off equivalence."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from lucy.app import build_selector, group_guidance, system_prompt
from lucy.core.bus import EventBus
from lucy.core.config import ToolsSettings
from lucy.core.context import ContextBuilder
from lucy.core.events import AssistantMessage, TurnEnded, UserMessage
from lucy.core.gateway import LLMGateway
from lucy.core.orchestrator import Orchestrator
from lucy.core.toolsets import META_TOOL, ToolGroup, ToolSelector, keyword_words
from lucy.core.types import FinishReason, LLMChunk, Risk, ToolCall, ToolOutput
from lucy.tools import GROUPS
from tests.fakes import FakeLLM, InMemoryMemoryStore
from tests.test_app import make_settings


def reply(text: str) -> list[LLMChunk]:
    return [LLMChunk(text=text), LLMChunk(finish_reason=FinishReason.STOP)]


class Stub:
    def __init__(self, name: str, risk: Risk = Risk.READ) -> None:
        self.name = name
        self.description = f"The {name} tool."
        self.parameters: dict[str, Any] = {"type": "object", "properties": {}}
        self.risk = risk
        self.runs: list[dict[str, Any]] = []

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        self.runs.append(args)
        return ToolOutput(content=f"{self.name} done")

    async def preview(self, args: dict[str, Any]) -> ToolOutput:
        return ToolOutput(content=f"Run {self.name}?")


GROUP_LIST = [
    ToolGroup("web", "lookups", ("web_search",), ("search",)),
    ToolGroup("calendar", "the calendar", ("calendar", "delete_event"), ("meeting", "calendar")),
    ToolGroup("reminders", "reminders", ("set_reminder",), ("remind", "yaad dila")),
]


def tools() -> list[Stub]:
    return [
        Stub("web_search"),
        Stub("calendar"),
        Stub("delete_event", Risk.DANGEROUS),
        Stub("set_reminder", Risk.ACT),
    ]


def selector(keep_turns: int = 3, **kw: Any) -> ToolSelector:
    return ToolSelector(tools(), GROUP_LIST, core=["web_search"], keep_turns=keep_turns, **kw)


def names(specs: Any) -> list[str]:
    return [s.name for s in specs]


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, arguments=arguments)


def calls(*tool_calls: ToolCall) -> list[LLMChunk]:
    return [
        *(LLMChunk(tool_call=c) for c in tool_calls),
        LLMChunk(finish_reason=FinishReason.TOOL_CALLS),
    ]


# --- Selection -------------------------------------------------------------------


def test_core_and_meta_tool_are_always_offered_in_a_fixed_order() -> None:
    s = selector()
    assert s.core == ["web_search"]
    assert names(s.specs(())) == ["web_search", META_TOOL]
    # Groups follow in configured order, whatever order they were switched on in.
    assert names(s.specs(["reminders", "calendar"])) == [
        "web_search", META_TOOL, "calendar", "delete_event", "set_reminder",
    ]  # fmt: skip


def test_keywords_match_whole_words_and_romanised_hindi_phrases() -> None:
    s = selector()
    assert s.matched("Any meeting tomorrow?") == {"calendar"}
    assert s.matched("kal subah 7 baje yaad   dila dena") == {"reminders"}
    assert s.matched("meetings") == set()  # whole words only
    assert s.matched("how are you") == set()
    assert selector(keywords=False).matched("any meeting?") == set()


def test_the_built_in_groups_match_common_requests() -> None:
    s = ToolSelector(
        [Stub(n) for g in GROUPS for n in g.tools],
        GROUPS,
        core=["wikipedia", "web_search", "fetch_page", "time", "weather"],
    )
    assert s.matched("What's on my calendar on Friday?") == {"calendar"}
    assert s.matched("remind me to call mum in 20 minutes") == {"reminders"}
    assert s.matched("mujhe kal yaad dilana") == {"reminders"}
    assert s.matched("Please enrol my voice") == {"voice"}
    assert s.matched("Who won the match yesterday?") == set()


def test_a_group_stays_on_for_keep_turns_after_its_last_use() -> None:
    s = selector(keep_turns=3)
    log: list[Any] = [UserMessage(session_id="s", text="any meeting today?")]
    assert s.groups_on(log) == {"calendar"}
    log.append(AssistantMessage(session_id="s", tool_calls=[call("calendar", when="today")]))
    for i, expected in enumerate([{"calendar"}, {"calendar"}, set()]):
        log.append(UserMessage(session_id="s", text=f"small talk {i}"))
        assert s.groups_on(log) == expected, i
    # A tool call, or a more_tools call, also counts as use.
    log.append(AssistantMessage(session_id="s", tool_calls=[call(META_TOOL, group="reminders")]))
    log.append(UserMessage(session_id="s", text="ok"))
    assert s.groups_on(log) == {"reminders"}


def test_from_calls_ignores_core_tools_and_unknown_groups() -> None:
    s = selector()
    found = s.from_calls([call("web_search"), call(META_TOOL, group="nope"), call("delete_event")])
    assert found == {"calendar"}


def test_unknown_core_tool_is_an_error() -> None:
    with pytest.raises(ValueError, match="not configured"):
        ToolSelector(tools(), GROUP_LIST, core=["nope"])


def test_tools_in_no_group_are_core_and_nothing_to_switch_means_inactive() -> None:
    s = ToolSelector([Stub("a"), Stub("b")], GROUP_LIST, core=[])
    assert s.core == ["a", "b"] and not s.active


async def test_meta_tool_result_names_the_group_tools() -> None:
    s = selector()
    out = await s.meta.run({"group": "calendar"})
    assert "calendar, delete_event" in out.content and not out.is_error
    assert (await s.meta.run({"group": "nope"})).is_error
    assert "calendar: the calendar" in s.meta.description


def test_keyword_words_drops_generic_parts() -> None:
    words = keyword_words("gmail", "search_email", "read_email_thread")
    assert words[:2] == ("gmail", "email")
    assert "mail" in words and "inbox" in words  # related words for an email server
    assert "search" not in words and "thread" not in words


def test_group_guidance_moves_sentences_naming_group_tools() -> None:
    groups = {g.name: g for g in GROUPS}
    core = ["wikipedia", "web_search", "fetch_page", "time", "weather", META_TOOL]
    moved = group_guidance(groups, core)
    assert "the calendar tool" in moved["calendar"]
    assert "the set_reminder tool" in moved["reminders"]
    prompt = system_prompt(core, confirming=True)
    assert "the calendar tool" not in prompt and "the weather tool" in prompt
    assert moved["calendar"] not in prompt


# --- Settings ----------------------------------------------------------------------


def test_tools_settings_defaults_and_validation() -> None:
    t = ToolsSettings()
    assert t.dynamic is True and t.core is None and t.keep_turns == 6
    assert "dynamic" not in ToolsSettings.known_tools()
    with pytest.raises(ValidationError, match="twice"):
        ToolsSettings(enabled=["time"], core=["time", "time"])
    with pytest.raises(ValidationError, match="not in"):
        ToolsSettings(enabled=["time"], core=["weather"])
    with pytest.raises(ValidationError):
        ToolsSettings(keep_turns=0)
    ToolsSettings(enabled=["time"], core=["time", "search_email"])  # MCP names allowed


def test_build_selector_is_none_when_off_or_everything_is_core(tmp_path: Any) -> None:
    base = make_settings(tmp_path / "lucy.db")
    on = base.model_copy(
        update={"tools": ToolsSettings(enabled=["time", "calendar"], dynamic=True)}
    )
    off = base.model_copy(
        update={"tools": ToolsSettings(enabled=["time", "calendar"], dynamic=False)}
    )
    mixed = [Stub("time"), Stub("calendar")]
    assert build_selector(off, mixed) is None
    assert build_selector(on, [Stub("time")]) is None
    sel = build_selector(on, mixed)
    assert sel is not None and sel.core == ["time"] and list(sel.groups) == ["calendar"]


# --- Orchestrator -------------------------------------------------------------------


class Rig:
    def __init__(self, scripts: list[list[LLMChunk]], sel: ToolSelector | None, tl: list[Stub]):
        self.bus = EventBus()
        self.llm = FakeLLM(scripts)
        self.memory = InMemoryMemoryStore()
        self.tools = tl
        self.orchestrator = Orchestrator(
            bus=self.bus,
            gateway=LLMGateway({"chat": self.llm}),
            memory=self.memory,
            context=ContextBuilder("You are Lucy.", layout="stable_prefix", clock=None),
            tools=tl,
            selector=sel,
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


async def test_prefill_uses_exactly_the_tools_the_next_turn_is_offered() -> None:
    """The tools equivalent of the history property in test_context/test_residency."""
    tl = tools()
    rig = Rig(
        [calls(call("calendar")), reply("You have one."), reply("Sure."), reply("Fine.")],
        ToolSelector(tl, GROUP_LIST, core=["web_search"], keep_turns=3),
        tl,
    )
    prefix = await rig.turns("any meeting today?", "thanks", prefix_before=1)
    assert prefix is not None
    assert prefix.tools == rig.llm.tools_seen[2]  # the next turn's first step
    assert "calendar" in names(prefix.tools)


async def test_groups_drop_at_the_prefill_after_keep_turns() -> None:
    tl = tools()
    sel = ToolSelector(tl, GROUP_LIST, core=["web_search"], keep_turns=2)
    rig = Rig([reply("a"), reply("b"), reply("c")], sel, tl)
    prefix = await rig.turns("any meeting?", "hm", "ok", prefix_before=2)
    seen = [names(t) for t in rig.llm.tools_seen]
    assert "calendar" in seen[0] and "calendar" in seen[1] and "calendar" not in seen[2]
    assert prefix is not None and names(prefix.tools) == seen[2]


async def test_more_tools_offers_the_group_from_the_next_step() -> None:
    tl = tools()
    rig = Rig(
        [calls(call(META_TOOL, group="reminders")), calls(call("set_reminder")), reply("Done.")],
        ToolSelector(tl, GROUP_LIST, core=["web_search"]),
        tl,
    )
    await rig.turns("wake me at six")
    seen = [names(t) for t in rig.llm.tools_seen]
    assert seen[0] == ["web_search", META_TOOL]
    assert seen[1] == seen[2] == ["web_search", META_TOOL, "set_reminder"]
    assert tl[3].runs == [{}]


async def test_a_configured_tool_not_offered_still_runs_when_called() -> None:
    tl = tools()
    rig = Rig(
        [calls(call("set_reminder")), reply("Done.")],
        ToolSelector(tl, GROUP_LIST, core=["web_search"]),
        tl,
    )
    await rig.turns("wake me at six")
    assert tl[3].runs == [{}]
    assert "set_reminder" in names(rig.llm.tools_seen[1])


async def test_confirmation_runs_the_logged_call_when_its_tool_is_no_longer_offered() -> None:
    tl = tools()
    sel = ToolSelector(tl, GROUP_LIST, core=["web_search"], keep_turns=1)
    rig = Rig([calls(call("delete_event", id="e1")), reply("Deleted.")], sel, tl)
    await rig.turns("delete the meeting", "yes")
    assert tl[2].runs == [{"id": "e1"}]
    # keep_turns=1 and "yes" names nothing, but the logged call is a use of the group.
    assert "delete_event" in names(rig.llm.tools_seen[-1])


async def test_feature_off_offers_every_tool_on_every_step() -> None:
    tl = tools()
    off = Rig([calls(call("calendar")), reply("x")], None, tl)
    await off.turns("hello")
    assert all(names(t) == [s.name for s in tl] for t in off.llm.tools_seen)
    # A selector with nothing to switch is the same as none.
    flat = Rig([reply("x")], ToolSelector([Stub("a")], GROUP_LIST, core=["a"]), [Stub("a")])
    await flat.turns("hello")
    assert names(flat.llm.tools_seen[0]) == ["a"]
