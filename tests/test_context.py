from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from lucy.core.context import (
    INSTRUCTIONS_LABEL,
    MEMORIES_LABEL,
    PROFILE_LABEL,
    ContextBudgets,
    ContextBuilder,
    SectionReport,
    date_line,
    estimate_tokens,
)
from lucy.core.types import MemoryContext, Message, Role, ToolCall


def system_builder(*args: Any, **kwargs: Any) -> ContextBuilder:
    """A builder with the "system" layout, which the section tests below describe."""
    return ContextBuilder(*args, layout="system", **kwargs)


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
    builder = system_builder("one two", count_tokens=words)
    budgets = builder.token_budgets(context_window=1002, reserve_output=0)
    assert budgets == {"profile": 100, "instructions": 100, "memories": 250, "history": 500}


def test_empty_sections_are_omitted() -> None:
    builder = system_builder("You are Lucy.")
    built = builder.build(memory=MemoryContext(), history=[user("hi")], context_window=8000)
    assert built.messages == [Message(role=Role.SYSTEM, content="You are Lucy."), user("hi")]
    report = by_name(built.report)
    assert report["persona"].included == 1
    for name in ("profile", "instructions", "memories"):
        assert report[name] == SectionReport(name=name, tokens=0, included=0, dropped=0)
    assert report["history"].included == 1


def test_all_sections_are_labelled_in_order() -> None:
    builder = system_builder("You are Lucy.")
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
    builder = system_builder("persona", count_tokens=words)
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
    builder = system_builder(persona, count_tokens=words)
    built = builder.build(memory=MemoryContext(), history=[user("hi")], context_window=100)
    assert built.messages[0].content == persona.strip()
    assert by_name(built.report)["persona"].tokens == 500


def test_long_profile_is_truncated_to_budget() -> None:
    builder = system_builder("p", count_tokens=words)
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
    builder = system_builder("p", count_tokens=words)
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
    builder = system_builder("p", count_tokens=words)
    built = builder.build(memory=MemoryContext(), history=history, context_window=50)
    assert built.messages[1:] == [history[-1]]
    assert by_name(built.report)["history"] == SectionReport(
        name="history", tokens=1000, included=1, dropped=2
    )


IST = timezone(timedelta(hours=5, minutes=30), "IST")


def test_date_line_names_the_time_zone_and_offset() -> None:
    assert date_line(datetime(2026, 11, 12, 11, 42, tzinfo=IST)) == (
        "Today is Thursday, 12 November 2026, 11:42 IST (UTC+05:30)."
    )
    est = timezone(timedelta(hours=-5), "EST")
    assert (
        date_line(datetime(2027, 1, 3, 7, 5, tzinfo=est))
        == "Today is Sunday, 3 January 2027, 07:05 EST (UTC-05:00)."
    )
    nepal = timezone(timedelta(hours=5, minutes=45))  # no abbreviation: offset only
    assert date_line(datetime(2026, 11, 12, 11, 42, tzinfo=nepal)).endswith("11:42 (UTC+05:45).")
    assert date_line(datetime(2026, 11, 12, 6, 12, tzinfo=UTC)).endswith("06:12 (UTC+00:00).")
    odd = timezone(timedelta(hours=-3), "-03")  # numeric "names" are not repeated
    assert date_line(datetime(2026, 11, 12, 3, 12, tzinfo=odd)).endswith("03:12 (UTC-03:00).")


def test_date_line_without_a_time_zone_says_local_time() -> None:
    assert date_line(datetime(2026, 11, 12, 11, 42)) == (
        "Today is Thursday, 12 November 2026, 11:42 (local time)."
    )


def test_clock_adds_the_current_date_after_the_persona_on_every_build() -> None:
    now = [datetime(2026, 11, 12, 23, 59, tzinfo=IST)]
    builder = system_builder("You are Lucy.", count_tokens=words, clock=lambda: now[0])
    first = builder.build(memory=MemoryContext(), history=[user("hi")], context_window=100)
    now[0] = datetime(2026, 11, 13, 0, 1, tzinfo=IST)
    second = builder.build(memory=MemoryContext(), history=[user("hi")], context_window=100)
    assert first.messages[0].content == (
        "You are Lucy.\n\nToday is Thursday, 12 November 2026, 23:59 IST (UTC+05:30)."
    )
    assert "Friday, 13 November 2026, 00:01" in second.messages[0].content
    # The date line is counted with the persona, in the budgets and the report.
    assert by_name(first.report)["persona"].tokens == words(first.messages[0].content)
    plain = system_builder("You are Lucy.", count_tokens=words)
    assert (
        plain.token_budgets(100, reserve_output=0)["history"]
        > builder.token_budgets(100, reserve_output=0)["history"]
    )
    assert (
        plain.build(memory=MemoryContext(), history=[user("hi")], context_window=100)
        .messages[0]
        .content
        == "You are Lucy."
    )


# --- The "stable_prefix" layout ------------------------------------------------------


def stable(clock=None, **kwargs) -> ContextBuilder:
    return ContextBuilder("You are Lucy.", layout="stable_prefix", clock=clock, **kwargs)


FULL_MEMORY = MemoryContext(
    profile="Yogesh lives in Pune.",
    instructions=["Call me Yogi."],
    statements=["Likes filter coffee."],
)


def test_default_layout_is_stable_prefix() -> None:
    assert ContextBuilder("p").layout == "stable_prefix"


def test_stable_prefix_puts_rarely_changing_sections_first() -> None:
    now = datetime(2026, 11, 12, 11, 42, tzinfo=IST)
    built = stable(clock=lambda: now).build(
        memory=FULL_MEMORY,
        history=[user("hi"), assistant("hello"), user("coffee?")],
        context_window=8000,
    )
    system, *rest = built.messages
    assert system == Message(
        role=Role.SYSTEM,
        content=(
            "You are Lucy.\n\n"
            f"{INSTRUCTIONS_LABEL}\n- Call me Yogi.\n\n"
            f"{PROFILE_LABEL}\nYogesh lives in Pune."
        ),
    )
    # The date and memories follow the latest user message, as a system message.
    assert rest == [
        user("hi"),
        assistant("hello"),
        user("coffee?"),
        Message(
            role=Role.SYSTEM,
            content=(
                "Today is Thursday, 12 November 2026, 11:42 IST (UTC+05:30).\n\n"
                f"{MEMORIES_LABEL}\n- Likes filter coffee."
            ),
        ),
    ]


def test_stable_prefix_without_per_turn_content_adds_nothing() -> None:
    built = stable().build(memory=MemoryContext(), history=[user("hi")], context_window=8000)
    assert built.messages == [Message(role=Role.SYSTEM, content="You are Lucy."), user("hi")]


def test_stable_prefix_does_not_change_the_history_it_was_given() -> None:
    history = [user("hi"), assistant("hello"), user("coffee?")]
    original = [m.model_copy() for m in history]
    stable().build(memory=FULL_MEMORY, history=history, context_window=8000)
    assert history == original


def test_stable_prefix_next_turn_extends_the_previous_prompt() -> None:
    """Across turns, the prompt is identical up to the previous turn's user message."""
    times = iter(
        [datetime(2026, 11, 12, 11, 42, tzinfo=IST), datetime(2026, 11, 12, 11, 47, tzinfo=IST)]
    )
    builder = stable(clock=lambda: next(times))
    first_history = [user("hi"), assistant("hello"), user("coffee?")]
    first = builder.build(memory=FULL_MEMORY, history=first_history, context_window=8000)
    second_memory = FULL_MEMORY.model_copy(update={"statements": ["Drinks it black."]})
    second = builder.build(
        memory=second_memory,
        history=[*first_history, assistant("Filter coffee, as usual?"), user("yes")],
        context_window=8000,
    )
    assert second.messages[:4] == first.messages[:4]  # system, hi, hello, coffee?
    assert (
        "11:47" in second.messages[-1].content and "Drinks it black." in second.messages[-1].content
    )
    assert second.messages[-2] == user("yes")


def test_stable_prefix_keeps_the_tool_steps_after_the_turn_block() -> None:
    call = ToolCall(id="c1", name="t", arguments={})
    history = [
        user("look it up"),
        Message(role=Role.ASSISTANT, content="", tool_calls=[call]),
        Message(role=Role.TOOL, content="result", tool_call_id="c1"),
    ]
    built = stable(clock=lambda: datetime(2026, 11, 12, 11, 42, tzinfo=IST)).build(
        memory=MemoryContext(), history=history, context_window=8000
    )
    roles = [m.role for m in built.messages]
    assert roles == [Role.SYSTEM, Role.USER, Role.SYSTEM, Role.ASSISTANT, Role.TOOL]


def test_stable_prefix_report_and_budgets_match_the_system_layout() -> None:
    now = datetime(2026, 11, 12, 11, 42, tzinfo=IST)
    history = [user("earlier " * 30), assistant("ok"), user("now")]
    kwargs = dict(count_tokens=words, clock=lambda: now)
    old = system_builder("You are Lucy.", **kwargs)
    new = ContextBuilder("You are Lucy.", layout="stable_prefix", **kwargs)
    assert old.token_budgets(200) == new.token_budgets(200)
    a = old.build(memory=FULL_MEMORY, history=history, context_window=120, reserve_output=0)
    b = new.build(memory=FULL_MEMORY, history=history, context_window=120, reserve_output=0)
    assert a.report == b.report
    # The latest user message is always kept, and still last among the history.
    assert [m for m in b.messages if m.role is Role.USER][-1] == user("now")


def test_stable_prefix_latest_user_message_kept_when_over_budget() -> None:
    history = [user("earlier"), assistant("ok"), user("huge " * 1000)]
    builder = ContextBuilder("p", count_tokens=words, layout="stable_prefix")
    built = builder.build(
        memory=MemoryContext(statements=["fact"]),
        history=history,
        context_window=50,
        reserve_output=0,
    )
    assert built.messages[1] == history[-1]
    assert built.messages[2].role is Role.SYSTEM and "- fact" in built.messages[2].content


def test_prefix_messages_end_before_the_latest_user_message() -> None:
    builder = stable(clock=lambda: datetime(2026, 11, 12, 11, 42, tzinfo=IST))
    history = [user("hi"), assistant("hello"), user("(placeholder)")]
    built = builder.build(memory=FULL_MEMORY, history=history, context_window=8000)
    assert builder.prefix_messages(built) == built.messages[:3]
    assert built.messages[3] == user("(placeholder)")


def test_prefix_messages_need_the_stable_prefix_layout() -> None:
    builder = system_builder("p")
    built = builder.build(memory=MemoryContext(), history=[user("hi")], context_window=8000)
    with pytest.raises(ValueError):
        builder.prefix_messages(built)
    with pytest.raises(ValueError):
        stable().prefix_messages(built.model_copy(update={"messages": built.messages[:1]}))
