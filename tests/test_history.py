from lucy.core.events import (
    AssistantMessage,
    Interrupt,
    SentenceReady,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UserMessage,
)
from lucy.core.history import events_to_messages
from lucy.core.types import Message, Role, ToolCall, ToolResult

S = "s"
CALL = ToolCall(id="c1", name="echo", arguments={"text": "hi"})


def test_plain_conversation_ignores_other_events() -> None:
    events = [
        UserMessage(session_id=S, text="hello"),
        TextDelta(session_id=S, text="Hi"),
        SentenceReady(session_id=S, text="Hi there."),
        AssistantMessage(session_id=S, text="Hi there."),
        TurnEnded(session_id=S, reason="completed"),
        Interrupt(session_id=S),
        UserMessage(session_id=S, text="how are you?"),
    ]
    assert events_to_messages(events) == [
        Message(role=Role.USER, content="hello"),
        Message(role=Role.ASSISTANT, content="Hi there."),
        Message(role=Role.USER, content="how are you?"),
    ]


def test_tool_round_trip() -> None:
    events = [
        UserMessage(session_id=S, text="echo hi"),
        AssistantMessage(session_id=S, tool_calls=[CALL]),
        ToolStarted(session_id=S, call=CALL),
        ToolFinished(session_id=S, result=ToolResult(call_id="c1", content="hi")),
        AssistantMessage(session_id=S, text="It said hi."),
    ]
    assert events_to_messages(events) == [
        Message(role=Role.USER, content="echo hi"),
        Message(role=Role.ASSISTANT, tool_calls=[CALL]),
        Message(role=Role.TOOL, content="hi", tool_call_id="c1"),
        Message(role=Role.ASSISTANT, content="It said hi."),
    ]


def test_dangling_tool_call_is_dropped() -> None:
    other = ToolCall(id="c2", name="echo", arguments={"text": "x"})
    events = [
        UserMessage(session_id=S, text="do two things"),
        AssistantMessage(session_id=S, text="On it.", tool_calls=[CALL, other]),
        ToolStarted(session_id=S, call=CALL),
        ToolFinished(session_id=S, result=ToolResult(call_id="c1", content="hi")),
        TurnEnded(session_id=S, reason="interrupted"),
        AssistantMessage(session_id=S, tool_calls=[ToolCall(id="c3", name="echo")]),
        UserMessage(session_id=S, text="next"),
    ]
    assert events_to_messages(events) == [
        Message(role=Role.USER, content="do two things"),
        Message(role=Role.ASSISTANT, content="On it.", tool_calls=[CALL]),
        Message(role=Role.TOOL, content="hi", tool_call_id="c1"),
        Message(role=Role.USER, content="next"),
    ]


def test_orphan_tool_result_is_dropped() -> None:
    events = [
        ToolFinished(session_id=S, result=ToolResult(call_id="c0", content="old")),
        UserMessage(session_id=S, text="hi"),
    ]
    assert events_to_messages(events) == [Message(role=Role.USER, content="hi")]


def test_interrupted_message_keeps_delivered_text() -> None:
    events = [
        UserMessage(session_id=S, text="tell me a story"),
        AssistantMessage(session_id=S, text="Once upon a", interrupted=True),
        UserMessage(session_id=S, text="stop"),
    ]
    assert events_to_messages(events)[1] == Message(role=Role.ASSISTANT, content="Once upon a")
