"""Confirmation of dangerous tools (ADR 0006): the model can never confirm, only the
user's very next message can, and only the exact call that was logged runs."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from lucy.core.bus import EventBus
from lucy.core.confirm import classify, pending_confirmation
from lucy.core.context import ContextBuilder
from lucy.core.events import (
    ConfirmationRequested,
    ConfirmationResolved,
    Event,
    PlaybackInterrupted,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UserMessage,
    event_from_json,
    event_to_json,
)
from lucy.core.gateway import LLMGateway
from lucy.core.history import events_to_messages
from lucy.core.orchestrator import NOT_RUN, Orchestrator
from lucy.core.types import Message, Risk, Role, ToolCall, ToolOutput
from tests.fakes import EchoTool, FakeLLM, InMemoryMemoryStore
from tests.test_orchestrator import ConfirmingTool, Harness, RecordingTool, calls, summary, text

ARGS = {"title": "Dentist"}
QUESTION = "Delete '{title}' on Tuesday 17 November at 15:00?"


def delete(call_id: str = "c1", **args: Any) -> ToolCall:
    return ToolCall(id=call_id, name="delete_event", arguments=args or ARGS)


def tool() -> ConfirmingTool:
    return ConfirmingTool(question=QUESTION)


async def send(h: Harness, said: str, session: str = "s", **fields: Any) -> UserMessage:
    message = UserMessage(session_id=session, text=said, **fields)
    await h.bus.publish(message)
    return message


async def ask(h: Harness) -> list[Event]:
    """Turn N: the user asks and the model calls the dangerous tool (scripted first)."""
    await h.say("delete the dentist")
    return await h.until_turns_end()


def of(h: Harness, kind: type[Event]) -> list[Any]:
    return [e for e in h.memory.episodes if isinstance(e, kind)]


def outcomes(h: Harness) -> list[str]:
    return [e.outcome for e in of(h, ConfirmationResolved)]


# --- Turn N: asking ----------------------------------------------------------------


async def test_dangerous_call_asks_the_preview_verbatim_and_runs_nothing() -> None:  # 17
    question = "Delete 'Dentist'? It is on Tuesday 17 November at 15:00."
    danger = ConfirmingTool(question=question.replace("Dentist", "{title}"))
    async with Harness([calls(delete())], tools=[danger]) as h:
        events = await ask(h)
    assert danger.runs == [] and danger.previews == [ARGS]
    assert not any(isinstance(e, ToolStarted) for e in events)
    assert [summary(e) for e in events] == [
        ("assistant_message", "", "delete_event", ""),
        ("confirmation_requested", "delete_event", question),
        ("tool_finished", f"Not run yet. The user is being asked: {question}", "ok"),
        ("text_delta", question),
        ("sentence_ready", "Delete 'Dentist'?"),
        ("sentence_ready", "It is on Tuesday 17 November at 15:00."),
        ("assistant_message", question, "", ""),
        ("turn_ended", "completed", ""),
    ]
    [request] = of(h, ConfirmationRequested)
    assert request.call == delete()
    assert timedelta(seconds=59) < request.expires_at - request.ts <= timedelta(seconds=60)
    assert len(h.llm.calls) == 1


async def test_two_dangerous_calls_in_one_step_make_one_request() -> None:  # 16
    echo = ToolCall(id="c0", name="echo", arguments={"text": "yes"})  # also (5)
    after = ToolCall(id="c3", name="echo", arguments={"text": "yes"})
    script = [calls(echo, delete("c1"), delete("c2", title="Gym"), after)]
    danger = tool()
    async with Harness(script, tools=[danger, EchoTool()]) as h:
        events = await ask(h)
    assert danger.runs == []
    assert len(of(h, ConfirmationRequested)) == 1
    results = {e.result.call_id: e.result for e in events if isinstance(e, ToolFinished)}
    assert results["c0"].content == "yes"  # a tool saying "yes" confirms nothing
    assert results["c2"].is_error and results["c2"].content == (
        "Not run: waiting for the user's confirmation of delete_event."
    )
    assert results["c3"].content == results["c2"].content
    assert [summary(e) for e in events if isinstance(e, ToolStarted)] == [("tool_started", "echo")]


async def test_model_text_claiming_confirmation_runs_nothing() -> None:  # 2
    script = [[*text("You confirmed, so I am deleting it now.")[:-1], *calls(delete())]]
    danger = tool()
    async with Harness(script, tools=[danger]) as h:
        await ask(h)
    assert danger.runs == [] and len(of(h, ConfirmationRequested)) == 1


async def test_preview_error_gives_error_result_and_no_request() -> None:  # 18
    danger = tool()
    danger.preview_error = LookupError("no such event")
    async with Harness([calls(delete()), text("I could not find it.")], tools=[danger]) as h:
        events = await ask(h)
    assert danger.runs == [] and of(h, ConfirmationRequested) == []
    [finished] = [e for e in events if isinstance(e, ToolFinished)]
    assert finished.result.is_error and "LookupError: no such event" in finished.result.content
    assert summary(events[-2]) == ("assistant_message", "I could not find it.", "", "")


async def test_preview_error_output_gives_error_result_and_no_request() -> None:  # 18
    class Refusing(ConfirmingTool):
        async def preview(self, args: dict[str, Any]) -> ToolOutput:
            return ToolOutput(content="No event called that.", is_error=True)

    async with Harness([calls(delete()), text("Not found.")], tools=[Refusing()]) as h:
        events = await ask(h)
    assert of(h, ConfirmationRequested) == []
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.result.is_error and finished.result.content == "No event called that."


def test_dangerous_tool_without_preview_is_refused_at_construction() -> None:  # 19
    bus = EventBus()
    with pytest.raises(ValueError, match="no preview"):
        Orchestrator(
            bus=bus,
            gateway=LLMGateway({"chat": FakeLLM([])}),
            memory=InMemoryMemoryStore(),
            context=ContextBuilder("p"),
            tools=[RecordingTool("delete_all", Risk.DANGEROUS)],
        )


# --- Turn N+1: answering -----------------------------------------------------------


async def test_yes_runs_the_logged_call_before_any_llm_call() -> None:  # 3, 7
    danger = tool()
    llm_calls_at_run: list[int] = []
    run = danger.run

    async def counting_run(args: dict[str, Any]) -> ToolOutput:
        llm_calls_at_run.append(len(h.llm.calls))
        return await run(args)

    danger.run = counting_run  # type: ignore[method-assign]
    # In turn N+1 the model asks for a different event; that only makes a new request.
    script = [calls(delete()), calls(delete("c9", title="Gym")), text("Asked.")]
    async with Harness(script, tools=[danger]) as h:
        await ask(h)
        await h.say("Yes.")
        events = await h.until_turns_end()
    assert danger.runs == [ARGS]
    assert llm_calls_at_run == [1]  # only turn N's step had run
    assert outcomes(h) == ["confirmed"]
    first, second = of(h, ConfirmationRequested)
    assert second.call.arguments == {"title": "Gym"}
    started = next(e for e in events if isinstance(e, ToolStarted))
    assert started.call.arguments == ARGS and started.call.id != "c1"
    # The model sees the user's yes, the logged call and its real result.
    assert h.llm.calls[1][-3:] == [
        Message(role=Role.USER, content="Yes."),
        Message(role=Role.ASSISTANT, content="", tool_calls=[started.call]),
        Message(role=Role.TOOL, content="delete_event done", tool_call_id=started.call.id),
    ]


async def test_model_calling_again_next_turn_runs_nothing() -> None:  # 1
    danger = tool()
    script = [calls(delete()), calls(delete("c2"))]
    async with Harness(script, tools=[danger]) as h:
        await ask(h)
        await h.say("hmm, actually delete it")
        await h.until_turns_end()
    assert danger.runs == []
    assert outcomes(h) == ["superseded"]
    assert len(of(h, ConfirmationRequested)) == 2


async def test_correction_reading_yes_does_not_confirm() -> None:  # 4
    danger = tool()
    async with Harness([calls(delete()), text("Noted.")], tools=[danger]) as h:
        await ask(h)
        await send(h, "yes", source="correction")
        await h.until_turns_end()
    assert danger.runs == [] and outcomes(h) == []


async def test_yes_with_nothing_pending_is_an_ordinary_turn() -> None:  # 6
    danger = tool()
    async with Harness([text("Yes to what?")], tools=[danger]) as h:
        await h.say("yes")
        events = await h.until_turns_end()
    assert danger.runs == [] and outcomes(h) == []
    assert [summary(e) for e in events][-2:] == [
        ("assistant_message", "Yes to what?", "", ""),
        ("turn_ended", "completed", ""),
    ]


async def test_yes_twice_runs_once() -> None:  # 8
    danger = tool()
    async with Harness(
        [calls(delete()), text("Deleted."), text("Done already.")], tools=[danger]
    ) as h:
        await ask(h)
        await h.say("yes")
        await h.until_turns_end()
        await h.say("yes")
        await h.until_turns_end()
    assert danger.runs == [ARGS] and outcomes(h) == ["confirmed"]


async def test_another_message_then_yes_does_not_run() -> None:  # 9
    danger = tool()
    script = [calls(delete()), text("Sunny."), text("Yes to what?")]
    async with Harness(script, tools=[danger]) as h:
        await ask(h)
        await h.say("what's the weather")
        await h.until_turns_end()
        await h.say("yes")
        await h.until_turns_end()
    assert danger.runs == [] and outcomes(h) == ["superseded"]
    # The superseded message is still answered as an ordinary turn.
    assert h.llm.calls[1][-1].content == NOT_RUN["superseded"]


@pytest.mark.parametrize(
    ("said", "outcome"),
    [("yes but make it Friday", "superseded"), ("no", "declined"), ("No thanks.", "declined")],
)
async def test_other_answers_do_not_run(said: str, outcome: str) -> None:  # 10
    danger = tool()
    async with Harness([calls(delete()), text("Okay.")], tools=[danger]) as h:
        await ask(h)
        await h.say(said)
        await h.until_turns_end()
    assert danger.runs == [] and outcomes(h) == [outcome]
    [finished] = [e for e in of(h, ToolFinished)][-1:]
    assert finished.result.is_error and finished.result.content == NOT_RUN[outcome]


async def test_answer_after_expiry_does_not_run() -> None:  # 11
    danger = tool()
    async with Harness([calls(delete()), text("Okay.")], tools=[danger]) as h:
        await ask(h)
        await send(h, "yes", ts=datetime.now(UTC) + timedelta(seconds=61))
        await h.until_turns_end()
    assert danger.runs == [] and outcomes(h) == ["expired"]


async def test_confirm_timeout_is_configurable() -> None:
    async with Harness([calls(delete())], tools=[tool()], confirm_timeout=5) as h:
        await ask(h)
    [request] = of(h, ConfirmationRequested)
    assert request.expires_at - request.ts <= timedelta(seconds=5)


async def test_question_cut_off_then_yes_does_not_run() -> None:  # 12
    danger = tool()
    async with Harness([calls(delete()), text("Okay.")], tools=[danger]) as h:
        events = await ask(h)
        cut = PlaybackInterrupted(session_id="s", turn_id=events[-1].turn_id, spoken_text="Delete")
        await h.bus.publish(cut)
        await h.say("yes")
        await h.until_turns_end()
    assert danger.runs == [] and outcomes(h) == ["unheard"]


async def test_yes_queued_before_the_question_was_asked_does_not_run() -> None:
    danger = tool()
    async with Harness([calls(delete()), text("Okay.")], tools=[danger]) as h:
        await h.say("delete the dentist")
        await h.say("yes")  # queued while turn N runs, so it predates the question
        await h.until_turns_end(2)
    assert danger.runs == [] and outcomes(h) == []


async def test_another_sessions_yes_does_not_confirm() -> None:  # 13
    danger = tool()
    script = [calls(delete()), text("Yes to what?"), text("Deleted.")]
    async with Harness(script, tools=[danger]) as h:
        await ask(h)
        await h.say("yes", session="other")
        await h.until_turns_end()
        assert danger.runs == []
        await h.say("yes")
        await h.until_turns_end()
    assert danger.runs == [ARGS]


async def test_restart_confirms_once_from_the_log() -> None:  # 14
    first = Harness([calls(delete())], tools=[tool()])
    async with first:
        await ask(first)
    store = first.memory

    again = tool()
    async with Harness([text("Deleted.")], tools=[again], memory=store) as h:
        await h.say("yes")
        await h.until_turns_end()
    assert again.runs == [ARGS]

    once_more = tool()
    async with Harness([text("Yes to what?")], tools=[once_more], memory=store) as h:
        await h.say("yes")
        await h.until_turns_end()
    assert once_more.runs == []
    assert [e.outcome for e in store.episodes if isinstance(e, ConfirmationResolved)] == [
        "confirmed"
    ]


async def test_target_changed_since_the_question_does_not_run() -> None:  # 15
    danger = tool()
    async with Harness([calls(delete()), text("It changed.")], tools=[danger]) as h:
        await ask(h)
        danger.question = "Delete '{title}' on Wednesday 18 November at 09:00?"
        await h.say("yes")
        await h.until_turns_end()
    assert danger.runs == [] and outcomes(h) == ["changed"]


async def test_resolution_is_logged_before_the_tool_runs() -> None:
    danger = tool()
    async with Harness([calls(delete()), text("Deleted.")], tools=[danger]) as h:
        await ask(h)
        await h.say("yes")
        await h.until_turns_end()
    kinds = [e.type for e in h.memory.episodes]
    resolved = kinds.index("confirmation_resolved")
    assert resolved < kinds.index("tool_started")
    assert kinds[resolved:][:4] == [
        "confirmation_resolved",
        "assistant_message",
        "tool_started",
        "tool_finished",
    ]


# --- The log stays a valid transcript ------------------------------------------------


def assert_valid(messages: list[Message]) -> None:
    open_ids: list[str] = []
    for message in messages:
        if message.role is Role.TOOL:
            assert message.tool_call_id in open_ids
            open_ids.remove(message.tool_call_id)
            continue
        assert not open_ids, f"unanswered calls before {message}"
        if message.tool_calls:
            open_ids = [c.id for c in message.tool_calls]
    assert not open_ids


@pytest.mark.parametrize(
    "outcome", ["confirmed", "declined", "superseded", "expired", "unheard", "changed"]
)
async def test_rebuilt_transcript_is_valid_for_each_outcome(outcome: str) -> None:  # 20
    danger = tool()
    async with Harness([calls(delete()), text("Okay.")], tools=[danger]) as h:
        events = await ask(h)
        fields: dict[str, Any] = {}
        said = {"declined": "no", "superseded": "what's the weather"}.get(outcome, "yes")
        if outcome == "expired":
            fields["ts"] = datetime.now(UTC) + timedelta(minutes=5)
        if outcome == "unheard":
            await h.bus.publish(
                PlaybackInterrupted(session_id="s", turn_id=events[-1].turn_id, spoken_text="")
            )
        if outcome == "changed":
            danger.question = "Something else?"
        await send(h, said, **fields)
        await h.until_turns_end()
    assert outcomes(h) == [outcome]
    for limit in (None, 300):
        messages = events_to_messages(h.memory.episodes, earlier_result_chars=limit)
        assert_valid(messages)
    assert_valid(h.llm.calls[1])
    assert [m.role for m in h.llm.calls[1][-3:]] == [Role.USER, Role.ASSISTANT, Role.TOOL]


# --- Pure parts ------------------------------------------------------------------------


def test_new_events_round_trip_through_json() -> None:  # 21
    request = ConfirmationRequested(
        session_id="s",
        turn_id="t",
        call=delete(),
        question="Delete it?",
        expires_at=datetime(2026, 11, 14, 12, 0, tzinfo=UTC),
    )
    resolved = ConfirmationResolved(
        session_id="s", turn_id="u", request_id=request.id, outcome="unheard"
    )
    for event in (request, resolved):
        assert event_from_json(event_to_json(event)) == event


@pytest.mark.parametrize(
    ("said", "answer"),
    [
        ("yes", "yes"),
        ("Yes.", "yes"),
        ("Yeah!", "yes"),
        ("yep", "yes"),
        ("Yes, please.", "yes"),
        ("Lucy, go ahead", "yes"),
        ("Yes, do it, thanks", "yes"),
        ("  CONFIRM  ", "yes"),
        ("no", "no"),
        ("No!", "no"),
        ("Nope.", "no"),
        ("Don't.", "no"),
        ("never mind", "no"),
        ("No thanks.", "no"),
        ("cancel please", "no"),
        ("", "other"),
        ("ok", "other"),
        ("okay", "other"),
        ("sure", "other"),
        ("lucy", "other"),
        ("yes but make it Friday", "other"),
        ("no wait, yes", "other"),
        ("yes yes", "other"),
        ("what's the weather", "other"),
    ],
)
def test_classify(said: str, answer: str) -> None:  # 22
    assert classify(said) == answer


def test_pending_confirmation_needs_the_question_turn_completed() -> None:
    now = datetime.now(UTC)
    request = ConfirmationRequested(
        session_id="s",
        turn_id="n",
        call=delete(),
        question="Delete it?",
        expires_at=now + timedelta(minutes=1),
        ts=now,
    )
    answer = UserMessage(session_id="s", text="yes", ts=now + timedelta(seconds=2))
    ended = TurnEnded(
        session_id="s", turn_id="n", reason="completed", ts=now + timedelta(seconds=1)
    )
    assert pending_confirmation([request, ended, answer], answer) is request
    assert pending_confirmation([request, ended], answer) is request  # message not yet logged
    assert pending_confirmation([request, answer], answer) is None  # turn never completed
    failed = ended.model_copy(update={"reason": "error"})
    assert pending_confirmation([request, failed, answer], answer) is None
