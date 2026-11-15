"""The private-data guard (ADR 0022): while a private source's result is in what the
model sees, a call that sends data out is asked about first, once, and runs only on
a plain yes."""

from typing import Any

from lucy.core.events import (
    AssistantMessage,
    ConfirmationRequested,
    ConfirmationResolved,
    ToolFinished,
    UserMessage,
)
from lucy.core.history import SHORTENED_MARKER, events_to_messages
from lucy.core.privacy import guard_question, needs_guard, visible_private_sources
from lucy.core.types import Risk, ToolCall, ToolOutput, ToolResult
from tests.fakes import InMemoryMemoryStore
from tests.test_orchestrator import ConfirmingTool, Harness, RecordingTool, calls, text

SECRET = "Your bank PIN reset code is 4412. " + "Lorem ipsum dolor sit amet. " * 40


class Tool(RecordingTool):
    """A recording tool with guard declarations."""

    def __init__(
        self,
        name: str,
        risk: Risk = Risk.READ,
        *,
        private_source: str | None = None,
        sends_out: bool = False,
        destination: str | None = None,
        output: str | None = None,
    ) -> None:
        super().__init__(name, risk)
        self.private_source = private_source
        self.sends_out = sends_out
        if destination is not None:
            self.destination = destination
        self.output = output

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        await super().run(args)
        return ToolOutput(content=self.output or f"{self.name} done")


def mail() -> Tool:
    return Tool("gmail_search", private_source="gmail", sends_out=True, output=SECRET)


def search() -> Tool:
    return Tool("web_search", sends_out=True, destination="your search service")


def call(call_id: str, name: str, **args: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


def requests(h: Harness) -> list[ConfirmationRequested]:
    return [e for e in h.memory.episodes if isinstance(e, ConfirmationRequested)]


def results(h: Harness) -> dict[str, ToolResult]:
    return {e.result.call_id: e.result for e in h.memory.episodes if isinstance(e, ToolFinished)}


QUESTION = (
    "Private information from gmail is in our conversation. web_search would send "
    'query "4412 bank" to your search service. Shall I go ahead?'
)


async def test_no_private_result_means_no_change() -> None:
    web, local = search(), Tool("time")
    script = [calls(call("c1", "web_search", query="kilimanjaro"), call("c2", "time")), text("ok")]
    async with Harness(script, tools=[web, local]) as h:
        await h.say("how tall is kilimanjaro")
        await h.until_turns_end()
    assert web.runs == [{"query": "kilimanjaro"}] and local.runs == [{}]
    assert requests(h) == []


async def test_private_result_then_outward_call_asks_once_and_runs_on_yes() -> None:
    gmail, web = mail(), search()
    script = [
        calls(call("c1", "gmail_search", q="bank")),
        calls(
            call("c2", "web_search", query="4412 bank"),
            call("c3", "web_search", query="other"),
        ),
        text("Searched."),
    ]
    async with Harness(script, tools=[gmail, web]) as h:
        await h.say("check my mail for the bank")
        await h.until_turns_end()
        assert web.runs == []
        [request] = requests(h)
        assert request.question == QUESTION and request.private_sources == ["gmail"]
        waiting = results(h)["c3"]
        assert waiting.is_error and "waiting for the user's confirmation" in waiting.content
        await h.say("Yes.")
        await h.until_turns_end()
    assert web.runs == [{"query": "4412 bank"}]
    assert [e.outcome for e in h.memory.episodes if isinstance(e, ConfirmationResolved)] == [
        "confirmed"
    ]
    assert len(requests(h)) == 1


async def test_anything_but_a_plain_yes_runs_nothing() -> None:
    gmail, web = mail(), search()
    script = [
        calls(call("c1", "gmail_search", q="bank")),
        calls(call("c2", "web_search", query="4412 bank")),
        text("Fine."),
    ]
    async with Harness(script, tools=[gmail, web]) as h:
        await h.say("check my mail")
        await h.until_turns_end()
        await h.say("ok sure")
        await h.until_turns_end()
    assert web.runs == []


async def test_same_private_server_read_tools_and_local_tools_are_not_asked() -> None:
    gmail = mail()
    read_thread = Tool("gmail_read", private_source="gmail", sends_out=True)
    local = Tool("time")
    script = [
        calls(call("c1", "gmail_search", q="bank")),
        calls(call("c2", "gmail_read", id="42"), call("c3", "time")),
        text("Done."),
    ]
    async with Harness(script, tools=[gmail, read_thread, local]) as h:
        await h.say("read the bank mail")
        await h.until_turns_end()
    assert read_thread.runs == [{"id": "42"}] and local.runs == [{}]
    assert requests(h) == []


async def test_same_server_non_read_tool_is_asked() -> None:
    gmail = mail()
    label = Tool("gmail_label", Risk.ACT, private_source="gmail", sends_out=True)
    script = [calls(call("c1", "gmail_search", q="x")), calls(call("c2", "gmail_label", id="1"))]
    async with Harness(script, tools=[gmail, label]) as h:
        await h.say("label it")
        await h.until_turns_end()
    assert label.runs == [] and len(requests(h)) == 1


async def test_dangerous_tool_question_gets_the_notice_and_rechecks_with_it() -> None:
    gmail = mail()
    danger = ConfirmingTool(question="Delete {title}?")
    danger.sends_out = True  # type: ignore[attr-defined]
    script = [
        calls(call("c1", "gmail_search", q="x")),
        calls(call("c2", "delete_event", title="Gym")),
        text("Deleted."),
    ]
    async with Harness(script, tools=[gmail, danger]) as h:
        await h.say("delete it")
        await h.until_turns_end()
        [request] = requests(h)
        assert request.question == (
            "Private information from gmail is in our conversation. Delete Gym?"
        )
        await h.say("yes")
        await h.until_turns_end()
    assert danger.runs == [{"title": "Gym"}]


async def test_marking_survives_shortening_while_the_result_is_visible() -> None:
    gmail, web = mail(), search()
    memory = InMemoryMemoryStore()
    # Turn 1: a private result. Turn 2: an outward call; the result is shortened by now.
    script = [
        calls(call("c1", "gmail_search", q="bank")),
        text("You have a code."),
        calls(call("c2", "web_search", query="4412")),
    ]
    async with Harness(script, tools=[gmail, web], memory=memory) as h:
        await h.say("any bank mail?")
        await h.until_turns_end()
        await h.say("search that code")
        await h.until_turns_end()
    shortened = [m.content for m in h.llm.calls[2] if m.tool_call_id == "c1"]
    assert shortened and shortened[0].endswith(SHORTENED_MARKER)
    assert web.runs == [] and len(requests(h)) == 1


async def test_marking_ends_once_the_result_leaves_the_window() -> None:
    gmail, web = mail(), search()
    script = [
        calls(call("c1", "gmail_search", q="bank")),
        text("You have a code."),
        text("Hello."),
        calls(call("c2", "web_search", query="weather")),
        text("Sunny."),
    ]
    async with Harness(script, tools=[gmail, web]) as h:
        h.orchestrator._history_limit = 4  # the private turn falls out of the window
        await h.say("any bank mail?")
        await h.until_turns_end()
        await h.say("hi")
        await h.until_turns_end()
        await h.say("search the weather")
        await h.until_turns_end()
    assert all(m.tool_call_id != "c1" for m in h.llm.calls[3])
    assert web.runs == [{"query": "weather"}] and requests(h) == []


def test_visible_sources_follow_the_rebuilt_history() -> None:
    tools = {"gmail_search": mail(), "web_search": search()}
    gmail_call = call("c1", "gmail_search", q="x")
    events = [
        UserMessage(session_id="s", text="a"),
        AssistantMessage(session_id="s", tool_calls=[gmail_call]),
        ToolFinished(session_id="s", result=ToolResult(call_id="c1", content=SECRET)),
        UserMessage(session_id="s", text="b"),
    ]
    assert visible_private_sources(events_to_messages(events), tools) == ["gmail"]  # shortened
    assert visible_private_sources(events_to_messages(events[2:]), tools) == []  # orphan dropped


def test_needs_guard_rules() -> None:
    other_private = Tool("drive_get", private_source="drive", sends_out=True)
    gmail_read = Tool("gmail_read", private_source="gmail", sends_out=True)
    assert not needs_guard(search(), [])
    assert needs_guard(search(), ["gmail"])
    assert not needs_guard(Tool("time"), ["gmail"])
    assert not needs_guard(gmail_read, ["gmail"])
    assert needs_guard(gmail_read, ["drive", "gmail"])  # drive data could reach gmail
    assert needs_guard(other_private, ["gmail"])


def test_guard_question_is_plain_and_bounded() -> None:
    tool = Tool("fetch_page", sends_out=True)
    question = guard_question(tool, {"url": 'https://x.example/?a="b"' + "c" * 100}, ["a", "b"])
    assert question.startswith("Private information from a and b is in our conversation.")
    assert '"b"' not in question and "outside this Mac" in question
    assert question.count("?") == 1 and len(question) < 220
