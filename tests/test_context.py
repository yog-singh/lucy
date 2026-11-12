import pytest
from pydantic import ValidationError

from lucy.core.context import (
    INSTRUCTIONS_LABEL,
    MEMORIES_LABEL,
    PROFILE_LABEL,
    ContextBudgets,
    ContextBuilder,
    SectionReport,
    estimate_tokens,
)
from lucy.core.types import MemoryContext, Message, Role, ToolCall


def words(text: str) -> int:
    """A predictable token counter for tests: one token per word."""
    return len(text.split())


def user(text: str) -> Message:
    return Message(role=Role.USER, content=text)


def assistant(text: str) -> Message:
    return Message(role=Role.ASSISTANT, content=text)


def by_name(report: list[SectionReport]) -> dict[str, SectionReport]:
    return {section.name: section for section in report}


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_budgets_must_sum_to_at_most_one() -> None:
    with pytest.raises(ValidationError):
        ContextBudgets(profile=0.5, instructions=0.5, memories=0.5, history=0.5)
    budgets = ContextBudgets(profile=0.25, instructions=0.25, memories=0.25, history=0.25)
    assert budgets.history == 0.25


def test_token_budgets_exclude_persona_and_reserve() -> None:
    builder = ContextBuilder("one two", count_tokens=words)
    budgets = builder.token_budgets(context_window=1002, reserve_output=0)
    assert budgets == {"profile": 100, "instructions": 100, "memories": 250, "history": 500}


def test_empty_sections_are_omitted() -> None:
    builder = ContextBuilder("You are Lucy.")
    built = builder.build(memory=MemoryContext(), history=[user("hi")], context_window=8000)
    assert built.messages == [Message(role=Role.SYSTEM, content="You are Lucy."), user("hi")]
    report = by_name(built.report)
    assert report["persona"].included == 1
    for name in ("profile", "instructions", "memories"):
        assert report[name] == SectionReport(name=name, tokens=0, included=0, dropped=0)
    assert report["history"].included == 1


def test_all_sections_are_labelled_in_order() -> None:
    builder = ContextBuilder("You are Lucy.")
    memory = MemoryContext(
        profile="Yogesh lives in Pune.",
        instructions=["Call me Yogi."],
        statements=["Likes filter coffee."],
    )
    built = builder.build(memory=memory, history=[user("hi")], context_window=8000)
    assert built.messages[0].content == (
        "You are Lucy.\n\n"
        f"{PROFILE_LABEL}\nYogesh lives in Pune.\n\n"
        f"{INSTRUCTIONS_LABEL}\n- Call me Yogi.\n\n"
        f"{MEMORIES_LABEL}\n- Likes filter coffee."
    )


def test_lines_are_included_in_order_until_budget() -> None:
    # Usable window 100 words: memories budget 25, instructions budget 10.
    builder = ContextBuilder("persona", count_tokens=words)
    label = words(MEMORIES_LABEL)
    assert label == 8
    statements = ["a b c d e f g h", "j k l m n o p", "q", "r"]  # lines cost 9, 8, 2, 2
    built = builder.build(
        memory=MemoryContext(statements=statements, instructions=["x " * 20]),
        history=[user("hi")],
        context_window=101,
        reserve_output=0,
    )
    report = by_name(built.report)
    # 8 + 9 = 17 fits, + 8 = 25 fits, + 2 = 27 does not: stop, drop the rest.
    assert report["memories"] == SectionReport(name="memories", tokens=25, included=2, dropped=2)
    assert "- q" not in built.messages[0].content and "- j k l" in built.messages[0].content
    # The single instruction is over budget, so the section is omitted entirely.
    assert report["instructions"] == SectionReport(
        name="instructions", tokens=0, included=0, dropped=1
    )
    assert INSTRUCTIONS_LABEL not in built.messages[0].content


def test_persona_is_never_truncated() -> None:
    persona = "word " * 500
    builder = ContextBuilder(persona, count_tokens=words)
    built = builder.build(memory=MemoryContext(), history=[user("hi")], context_window=100)
    assert built.messages[0].content == persona.strip()
    assert by_name(built.report)["persona"].tokens == 500


def test_long_profile_is_truncated_to_budget() -> None:
    builder = ContextBuilder("p", count_tokens=words)
    profile = " ".join(f"w{i}" for i in range(50))
    built = builder.build(
        memory=MemoryContext(profile=profile),
        history=[user("hi")],
        context_window=101,
        reserve_output=0,
    )
    section = by_name(built.report)["profile"]
    assert section.truncated and section.included == 1 and section.tokens <= 10
    assert built.messages[0].content.endswith("…")


def test_history_keeps_newest_that_fit_and_never_splits_tool_group() -> None:
    call = ToolCall(id="c1", name="t", arguments={})
    history = [
        user("old " * 10),  # 10
        assistant("old reply " * 5),  # 10
        user("check the tool"),  # 3
        Message(role=Role.ASSISTANT, content="looking", tool_calls=[call]),  # 1 + 1 + 1
        Message(role=Role.TOOL, content="r " * 20, tool_call_id="c1"),  # 20
        assistant("done"),  # 1
        user("thanks"),  # 1
    ]
    builder = ContextBuilder("p", count_tokens=words)
    # Usable 100, history budget 50 -> fits: 1 + 1 + (3 + 20) + 3 = 28; +10 = 38; +10 = 48.
    built = builder.build(
        memory=MemoryContext(), history=history, context_window=101, reserve_output=0
    )
    assert built.messages[1:] == history
    # Budget 26 (usable 52): the tool group (23) plus 2 fits; the user message before it
    # (3) would exceed it, so it and everything older are dropped.
    built = builder.build(
        memory=MemoryContext(), history=history, context_window=53, reserve_output=0
    )
    assert built.messages[1:] == history[3:]
    assert by_name(built.report)["history"] == SectionReport(
        name="history", tokens=25, included=4, dropped=3
    )
    # Budget 10: the tool group no longer fits and is dropped whole, never split.
    built = builder.build(
        memory=MemoryContext(), history=history, context_window=21, reserve_output=0
    )
    assert built.messages[1:] == history[5:]


def test_latest_user_message_is_always_kept() -> None:
    history = [user("earlier"), assistant("ok"), user("huge " * 1000)]
    builder = ContextBuilder("p", count_tokens=words)
    built = builder.build(memory=MemoryContext(), history=history, context_window=50)
    assert built.messages[1:] == [history[-1]]
    assert by_name(built.report)["history"] == SectionReport(
        name="history", tokens=1000, included=1, dropped=2
    )
