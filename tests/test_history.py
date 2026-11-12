from lucy.core.events import (
    AssistantMessage,
    Interrupt,
    PlaybackInterrupted,
    SentenceReady,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UserMessage,
)
from lucy.core.history import SHORTENED_MARKER, events_to_messages, shorten_tool_result
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


ARTICLE = "Article: Piranesi\n" + " ".join(f"word{i}" for i in range(200))


def _lookup_turn(call_id: str, question: str, content: str, *, is_error: bool = False) -> list:
    call = ToolCall(id=call_id, name="wikipedia", arguments={"query": question})
    return [
        UserMessage(session_id=S, text=question),
        AssistantMessage(session_id=S, tool_calls=[call]),
        ToolFinished(
            session_id=S, result=ToolResult(call_id=call_id, content=content, is_error=is_error)
        ),
        AssistantMessage(session_id=S, text="Answer."),
    ]


def test_earlier_turns_tool_results_are_shortened() -> None:
    events = _lookup_turn("c1", "first", ARTICLE) + _lookup_turn("c2", "second", ARTICLE)
    messages = events_to_messages(events)
    tool_messages = [m for m in messages if m.role is Role.TOOL]
    earlier, latest = tool_messages
    assert latest.content == ARTICLE  # the latest turn's lookup stays whole
    assert earlier.content.startswith("Article: Piranesi\nword0 word1 ")
    assert earlier.content.endswith(" " + SHORTENED_MARKER)
    body = earlier.content.removeprefix("Article: Piranesi\n").removesuffix(" " + SHORTENED_MARKER)
    assert 250 < len(body) <= 300 and ARTICLE.startswith(f"Article: Piranesi\n{body} ")
    # Pairing stays valid: every tool message answers the assistant call just before it.
    for i, message in enumerate(messages):
        if message.role is Role.TOOL:
            previous = messages[i - 1]
            assert previous.tool_calls and previous.tool_calls[0].id == message.tool_call_id
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]


def test_follow_up_step_in_the_same_turn_sees_the_whole_result() -> None:
    # The latest turn has two lookups and no later user message: both stay whole.
    events = _lookup_turn("c1", "first", ARTICLE)
    events += [
        AssistantMessage(session_id=S, tool_calls=[ToolCall(id="c2", name="wikipedia")]),
        ToolFinished(session_id=S, result=ToolResult(call_id="c2", content=ARTICLE)),
    ]
    tool_messages = [m for m in events_to_messages(events) if m.role is Role.TOOL]
    assert [m.content for m in tool_messages] == [ARTICLE, ARTICLE]


def test_errors_and_short_results_stay_whole_and_the_limit_is_a_parameter() -> None:
    error = "Wikipedia search failed: " + "x " * 300
    events = (
        _lookup_turn("c1", "a", error, is_error=True)
        + _lookup_turn("c2", "b", "Article: Short\nOne line.")
        + _lookup_turn("c3", "c", ARTICLE)
        + [UserMessage(session_id=S, text="thanks")]
    )
    contents = [m.content for m in events_to_messages(events) if m.role is Role.TOOL]
    assert contents[0] == error
    assert contents[1] == "Article: Short\nOne line."
    assert contents[2].endswith(SHORTENED_MARKER) and len(contents[2]) < 400
    whole = [m.content for m in events_to_messages(events, earlier_result_chars=None)]
    assert ARTICLE in whole
    tiny = [m.content for m in events_to_messages(events, earlier_result_chars=20)]
    assert f"Article: Piranesi\nword0 word1 word2 {SHORTENED_MARKER}" in tiny


def test_shorten_tool_result_caps_a_long_first_line() -> None:
    one_line = "x" * 1000
    assert shorten_tool_result(one_line, 300) == "x" * 300 + " " + SHORTENED_MARKER
    assert shorten_tool_result("short", 300) == "short"


# --- Spoken replies (ADR 0016) ------------------------------------------------------


def _turn(turn: str, text: str, *, reply: str, interrupted: bool = False) -> list:
    ids = {"session_id": S, "turn_id": turn}
    return [
        UserMessage(**ids, text=text, source="voice"),
        AssistantMessage(**ids, text=reply, interrupted=interrupted),
        TurnEnded(**ids, reason="interrupted" if interrupted else "completed"),
    ]


def test_playback_interrupted_replaces_the_reply_with_what_was_spoken() -> None:
    events = [
        *_turn("t1", "tell me a story", reply="Once upon a time. There was a fox. The end."),
        PlaybackInterrupted(session_id=S, turn_id="t1", spoken_text="Once upon a time."),
        *_turn("t2", "go on", reply="Sure."),
    ]
    assert events_to_messages(events) == [
        Message(role=Role.USER, content="tell me a story"),
        Message(role=Role.ASSISTANT, content="Once upon a time."),
        Message(role=Role.USER, content="go on"),
        Message(role=Role.ASSISTANT, content="Sure."),
    ]


def test_playback_interrupted_before_the_final_message_is_logged_also_applies() -> None:
    ids = {"session_id": S, "turn_id": "t1"}
    events = [
        UserMessage(**ids, text="question"),
        PlaybackInterrupted(**ids, spoken_text="The first part."),
        AssistantMessage(**ids, text="The first part. And more.", interrupted=True),
        TurnEnded(**ids, reason="interrupted"),
    ]
    assert events_to_messages(events)[1] == Message(role=Role.ASSISTANT, content="The first part.")


def test_nothing_spoken_drops_the_reply() -> None:
    events = [
        *_turn("t1", "question", reply="An answer nobody heard."),
        PlaybackInterrupted(session_id=S, turn_id="t1", spoken_text=""),
        UserMessage(session_id=S, turn_id="t2", text="never mind"),
    ]
    assert events_to_messages(events) == [
        Message(role=Role.USER, content="question"),
        Message(role=Role.USER, content="never mind"),
    ]


def test_earlier_steps_and_tool_results_of_the_turn_are_unaffected() -> None:
    ids = {"session_id": S, "turn_id": "t1"}
    events = [
        UserMessage(**ids, text="look it up"),
        AssistantMessage(**ids, text="Let me check.", tool_calls=[CALL]),
        ToolFinished(**ids, result=ToolResult(call_id="c1", content="hi")),
        AssistantMessage(**ids, text="It says hi. That is all."),
        TurnEnded(**ids, reason="completed"),
        PlaybackInterrupted(**ids, spoken_text="It says hi."),
    ]
    assert events_to_messages(events) == [
        Message(role=Role.USER, content="look it up"),
        Message(role=Role.ASSISTANT, content="Let me check.", tool_calls=[CALL]),
        Message(role=Role.TOOL, content="hi", tool_call_id="c1"),
        Message(role=Role.ASSISTANT, content="It says hi."),
    ]


def test_nothing_spoken_keeps_an_answered_tool_call_of_the_final_message() -> None:
    ids = {"session_id": S, "turn_id": "t1"}
    events = [
        UserMessage(**ids, text="look it up"),
        AssistantMessage(**ids, text="Checking.", tool_calls=[CALL]),
        ToolFinished(**ids, result=ToolResult(call_id="c1", content="hi")),
        PlaybackInterrupted(**ids, spoken_text=""),
    ]
    assert events_to_messages(events)[1] == Message(
        role=Role.ASSISTANT, content="", tool_calls=[CALL]
    )


def test_the_last_playback_interrupted_of_a_turn_wins() -> None:
    events = [
        *_turn("t1", "question", reply="One. Two. Three."),
        PlaybackInterrupted(session_id=S, turn_id="t1", spoken_text="One."),
        PlaybackInterrupted(session_id=S, turn_id="t1", spoken_text="One. Two."),
    ]
    assert events_to_messages(events)[1].content == "One. Two."


def test_playback_interrupted_of_another_turn_or_without_a_turn_changes_nothing() -> None:
    events = [
        *_turn("t1", "question", reply="Full answer."),
        PlaybackInterrupted(session_id=S, turn_id="elsewhere", spoken_text="Other."),
        PlaybackInterrupted(session_id=S, spoken_text="No turn."),
    ]
    assert events_to_messages(events)[1].content == "Full answer."
