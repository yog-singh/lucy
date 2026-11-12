"""The core profile writer (lucy.memory.profile)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from lucy.core.context import estimate_tokens
from lucy.core.gateway import LLMGateway, Preempted, Priority
from lucy.core.types import LLMChunk, Message, Role
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.models import USER_ENTITY_ID, Origin, StatementKind
from lucy.memory.profile import (
    INPUT_HASH_KEY,
    ProfilePolicy,
    ProfileWriter,
    plain_text,
    select_profile_input,
    truncate_to_tokens,
    unsupported_pronouns,
    unsupported_terms,
)
from tests.fakes import FakeLLM, Pause
from tests.graph_support import T0, Clock, HashingEmbedder, new

DAY = timedelta(days=1)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def graph(tmp_path: Path, clock: Clock) -> AsyncIterator[SQLiteGraphStore]:
    async with SQLiteGraphStore(tmp_path / "lucy.db", HashingEmbedder(), clock=clock) as g:
        yield g


def reply(text: str) -> list[LLMChunk]:
    """A scripted response streamed in two chunks."""
    middle = len(text) // 2
    return [LLMChunk(text=text[:middle]), LLMChunk(text=text[middle:])]


def writer(
    graph: SQLiteGraphStore, clock: Clock, responses: list[list[LLMChunk]], **policy: object
) -> tuple[ProfileWriter, FakeLLM]:
    llm = FakeLLM(responses)
    gateway = LLMGateway({"summary": llm})
    return ProfileWriter(gateway, graph, policy=ProfilePolicy(**policy), clock=clock), llm  # type: ignore[arg-type]


async def populate(graph: SQLiteGraphStore) -> dict[str, int]:
    """A small graph about Yogesh; returns statement ids by label."""
    await graph.rename_entity(USER_ENTITY_ID, "Yogesh", keep_old_as_alias=False)
    priya = await graph.create_entity("person", "Priya")
    rahul = await graph.create_entity("person", "Rahul")
    lucy = await graph.create_entity("project", "Lucy")
    storage = await graph.create_entity("decision", "Storage choice")
    ids = {}

    async def add(label: str, text: str, **kwargs: object) -> None:
        ids[label] = (await graph.add_statement(new(text, **kwargs))).id  # type: ignore[arg-type]

    await add("city", "Yogesh lives in Bangalore")
    await add("sister", "Yogesh's sister is Priya", object_id=priya.id)
    await add("priya_city", "Priya lives in Pune", subject_id=priya.id)
    await add("friend", "Yogesh's friend is Rahul", object_id=rahul.id)
    await add("builds", "Yogesh is building Lucy, a local assistant", object_id=lucy.id)
    await add("goal", "Yogesh wants to ship the voice loop by January", kind=StatementKind.GOAL)
    await add("pref", "Yogesh prefers short answers", kind=StatementKind.PREFERENCE)
    await add(
        "habit",
        "Yogesh goes to the gym on Mon/Wed/Fri",
        kind=StatementKind.HABIT,
        origin=Origin.INFERRED,
        confidence=0.7,
    )
    await add(
        "decision",
        "Chose SQLite over Neo4j for simplicity",
        subject_id=storage.id,
        kind=StatementKind.DECISION,
    )
    # Excluded:
    await add("instruction", "Don't read URLs aloud", kind=StatementKind.INSTRUCTION)
    await add(
        "uncertain",
        "Yogesh may dislike mornings",
        kind=StatementKind.PREFERENCE,
        origin=Origin.INFERRED,
        confidence=0.4,
    )
    await add("old_city", "Yogesh lives in Mumbai")
    await graph.invalidate(ids["old_city"])
    return ids


# --- Selection ------------------------------------------------------------------


async def test_selection_includes_the_strongest_current_statements(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    ids = await populate(graph)
    clock.advance(seconds=1)
    selected = await select_profile_input(graph, now=clock.now)
    titles = [title for title, _ in selected.sections]
    assert titles == [
        "Identity and life",
        "People",
        "Work and projects",
        "Goals",
        "Preferences",
        "Habits",
        "Recent decisions",
    ]
    chosen = set(selected.statement_ids)
    for label in ("city", "sister", "priya_city", "friend", "builds", "goal", "pref", "habit"):
        assert ids[label] in chosen, label
    assert ids["decision"] in chosen
    for label in ("instruction", "uncertain", "old_city"):
        assert ids[label] not in chosen, label
    assert selected.text.startswith("Name: Yogesh\n")
    assert "- Yogesh goes to the gym on Mon/Wed/Fri (inferred)" in selected.text
    assert "- Yogesh prefers short answers\n" in selected.text


async def test_key_people_are_the_most_mentioned(graph: SQLiteGraphStore, clock: Clock) -> None:
    people = [await graph.create_entity("person", f"Person{i}") for i in range(4)]
    for i, person in enumerate(people):
        for j in range(i + 1):  # Person3 is mentioned most
            await graph.add_statement(
                new(f"User knows Person{i} from place {j}", object_id=person.id)
            )
        await graph.add_statement(new(f"Person{i} plays chess", subject_id=person.id))
    selected = await select_profile_input(
        graph, policy=ProfilePolicy(key_people=2, statements_per_person=1), now=clock.now
    )
    [people_section] = [items for title, items in selected.sections if title == "People"]
    texts = " ".join(s.text for s in people_section)
    assert "Person3" in texts and "Person2" in texts
    assert "Person0" not in texts and "Person1" not in texts
    assert "Person3 plays chess" in texts


async def test_ranking_by_confidence_and_recency(graph: SQLiteGraphStore, clock: Clock) -> None:
    await graph.add_statement(new("User likes old things", kind=StatementKind.PREFERENCE,
                                  confidence=0.9, observed_at=T0 - 720 * DAY))  # fmt: skip
    await graph.add_statement(new("User likes new things", kind=StatementKind.PREFERENCE,
                                  confidence=0.8, observed_at=T0))  # fmt: skip
    await graph.add_statement(new("User likes weak things", kind=StatementKind.PREFERENCE,
                                  confidence=0.7, observed_at=T0))  # fmt: skip
    selected = await select_profile_input(graph, now=T0)
    [(_, items)] = selected.sections
    assert [s.text for s in items] == [
        "User likes new things",
        "User likes weak things",
        "User likes old things",
    ]
    assert selected.text.startswith("Name: the user\n")


async def test_input_is_capped_by_the_token_budget_round_robin(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    for i in range(30):
        await graph.add_statement(new(f"User fact number {i} is rather long text" * 2))
        await graph.add_statement(
            new(f"User prefers option {i} in situation {i}", kind=StatementKind.PREFERENCE)
        )
    selected = await select_profile_input(
        graph, policy=ProfilePolicy(input_token_budget=200), now=clock.now
    )
    assert estimate_tokens(selected.text) <= 200 + 10
    assert [title for title, _ in selected.sections] == ["Identity and life", "Preferences"]
    counts = [len(items) for _, items in selected.sections]
    assert abs(counts[0] - counts[1]) <= 1 and sum(counts) < 60


# --- Writing --------------------------------------------------------------------

GOOD = (
    "Yogesh lives in Bangalore. Yogesh's sister Priya lives in Pune, and Yogesh's friend is "
    "Rahul. Yogesh is building Lucy and wants to ship the voice loop by January. Yogesh "
    "prefers short answers."
)


async def test_rewrite_saves_a_plain_third_person_profile(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    ids = await populate(graph)
    w, llm = writer(graph, clock, [reply(GOOD)])
    result = await w.rewrite()
    assert result.status == "written" and result.attempts == 1 and result.flagged == []
    assert result.profile is not None and result.profile.content == GOOD
    assert (await graph.current_profile()) == result.profile
    assert ids["instruction"] not in result.statement_ids
    [messages] = llm.calls
    assert messages[0].role is Role.SYSTEM and "Yogesh" in messages[0].content
    assert "{name}" not in messages[0].content and "{max_words}" not in messages[0].content
    assert "Yogesh lives in Bangalore" in messages[1].content
    assert "Mumbai" not in messages[1].content and "URLs" not in messages[1].content
    assert llm.schemas_seen == [None]  # plain text, not JSON


async def test_markdown_is_removed() -> None:
    text = "## Profile\n\n- **Yogesh** lives in Bangalore\n- He prefers *short* answers.\n"
    assert plain_text(text) == "Profile. Yogesh lives in Bangalore. He prefers short answers."


async def test_unchanged_input_skips_the_model(graph: SQLiteGraphStore, clock: Clock) -> None:
    await populate(graph)
    w, llm = writer(graph, clock, [reply(GOOD), reply(GOOD), reply(GOOD)])
    first = await w.rewrite()
    clock.advance(days=3)  # time alone does not change the input
    second = await w.rewrite()
    assert second.status == "unchanged" and second.profile == first.profile
    assert len(llm.calls) == 1
    forced = await w.rewrite(force=True)
    assert forced.status == "written" and len(llm.calls) == 2
    await graph.add_statement(new("Yogesh plays the guitar"))
    changed = await w.rewrite()
    assert changed.status == "written" and len(llm.calls) == 3
    assert len(await graph.profile_versions()) == 3  # old versions kept


async def test_rewrites_when_the_saved_profile_is_gone(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    """E.g. "forget that" deleted profile versions without changing the selected input."""
    await populate(graph)
    w, llm = writer(graph, clock, [reply(GOOD), reply(GOOD)])
    await w.rewrite()
    await graph.save_profile("Some other version")  # the hashed version is no longer current
    assert (await w.rewrite()).status == "written"
    assert len(llm.calls) == 2


async def test_no_eligible_statements_means_no_call_and_nothing_saved(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    await graph.add_statement(new("Don't read URLs aloud", kind=StatementKind.INSTRUCTION))
    await graph.add_statement(new("User may like jazz", origin=Origin.INFERRED, confidence=0.3))
    w, llm = writer(graph, clock, [])
    result = await w.rewrite()
    assert result.status == "empty" and result.profile is None
    assert llm.calls == [] and await graph.current_profile() is None
    assert await graph.get_meta(INPUT_HASH_KEY) is None


async def test_too_long_is_retried_once_then_accepted(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    await populate(graph)
    long = "Yogesh lives in Bangalore. " * 40
    w, llm = writer(graph, clock, [reply(long), reply(GOOD)], max_tokens=100)
    result = await w.rewrite()
    assert result.attempts == 2 and not result.truncated
    assert result.profile is not None and result.profile.content == GOOD
    retry = llm.calls[1]
    assert retry[-2] == Message(role=Role.ASSISTANT, content=long.strip())
    assert "words long" in retry[-1].content


async def test_still_too_long_is_truncated_at_a_sentence_boundary(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    await populate(graph)
    long = " ".join(f"Yogesh lives in Bangalore and this is sentence {i}." for i in range(60))
    w, _ = writer(graph, clock, [reply(long), reply(long)], max_tokens=60)
    result = await w.rewrite()
    assert result.truncated and result.profile is not None
    content = result.profile.content
    assert estimate_tokens(content) <= 60 and content.endswith(".")
    assert long.startswith(content)


async def test_invented_names_are_flagged_and_retried(
    graph: SQLiteGraphStore, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    await populate(graph)
    invented = GOOD + " His wife Anita works at Google since 2019."
    w, llm = writer(graph, clock, [reply(invented), reply(GOOD)])
    result = await w.rewrite()
    assert result.attempts == 2 and result.flagged == []
    assert result.profile is not None and result.profile.content == GOOD
    assert "Anita, Google, 2019" in llm.calls[1][-1].content

    w, _ = writer(graph, clock, [reply(invented), reply(invented)])
    with caplog.at_level(logging.WARNING, logger="lucy.memory.profile"):
        result = await w.rewrite(force=True)
    assert result.flagged == ["Anita", "Google", "2019", "his"]
    assert result.profile is not None and result.profile.content == invented
    assert "Anita" in caplog.text


async def test_preempted_propagates_and_nothing_is_saved(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    await populate(graph)
    pause = Pause()
    summary = FakeLLM([[LLMChunk(text="Yogesh"), pause, LLMChunk(text=" lives")]])
    chat = FakeLLM([[LLMChunk(text="hi")]])
    gateway = LLMGateway({"summary": summary, "chat": chat})
    w = ProfileWriter(gateway, graph, clock=clock)
    task = asyncio.create_task(w.rewrite())
    await pause.reached.wait()
    async for _ in gateway.stream("chat", [], priority=Priority.FOREGROUND):
        pass
    with pytest.raises(Preempted):
        await task
    assert await graph.current_profile() is None
    assert await graph.get_meta(INPUT_HASH_KEY) is None


# --- Helpers --------------------------------------------------------------------


def test_unsupported_terms() -> None:
    sources = [
        "Yogesh goes to the gym on Mon/Wed/Fri at 7am",
        "Yogesh's sister is Priya",
        "Yogesh works at Acme Corp since 2021",
    ]
    text = (
        "Yogesh goes to the gym on Mondays, Wednesdays and Fridays at 7 AM. "
        "She is Priya's sister. Overall, he works at Acme. Additionally he met Rahul in 2020 "
        "at Google."
    )
    assert unsupported_terms(text, sources) == ["Rahul", "2020", "Google"]
    assert unsupported_terms("", sources) == []


async def test_a_gender_no_statement_gives_is_flagged_and_retried(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    await populate(graph)  # nothing says whether Yogesh is a man or a woman
    guessed = "Yogesh lives in Bangalore. He prefers short answers, and his sister is Priya."
    w, llm = writer(graph, clock, [reply(guessed), reply(GOOD)])
    result = await w.rewrite()
    assert result.attempts == 2 and result.flagged == []
    assert result.profile is not None and result.profile.content == GOOD
    assert '"he", "his"' in llm.calls[1][-1].content and "name" in llm.calls[1][-1].content

    w, _ = writer(graph, clock, [reply(guessed), reply(guessed)])
    assert (await w.rewrite(force=True)).flagged == ["he", "his"]


def test_unsupported_pronouns() -> None:
    sources = ["Sam lives in Leeds", "Sam's sister is Priya", "Priya works as a doctor"]
    # "sister" establishes a woman, so "she"/"her" pass; nothing establishes a man.
    text = "Sam lives in Leeds. His sister Priya is a doctor; she loves her work. He walks."
    assert unsupported_pronouns(text, sources) == ["his", "he"]
    assert unsupported_pronouns("Sam lives in Leeds.", sources) == []
    assert unsupported_pronouns("He lives in Leeds.", ["Sam's husband is Tom"]) == []
    assert unsupported_pronouns("He said his piece.", ["Sam said he would come"]) == []
    assert unsupported_pronouns("She and he.", ["Sam lives in Leeds"]) == ["she", "he"]
    assert unsupported_pronouns("The theme here.", []) == []  # whole words only


def test_truncate_to_tokens() -> None:
    text = "One two three. Four five six. Seven eight nine."
    assert truncate_to_tokens(text, 100) == text
    assert truncate_to_tokens(text, 8) == "One two three. Four five six."
    assert truncate_to_tokens("averyveryverylongword " * 10, 5).endswith("…")
