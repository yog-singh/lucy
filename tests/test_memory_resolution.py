"""Entity resolution: exact, alias, similar, model-confirmed, rejected, type mismatch, new."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lucy.core.gateway import LLMGateway
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.models import USER_ENTITY_ID
from lucy.memory.policy import MemoryPolicy
from lucy.memory.resolution import EntityResolver, MatchMethod, types_compatible
from tests.fakes import FakeLLM
from tests.graph_support import Clock, HashingEmbedder, new
from tests.memory_extraction_support import TEST_POLICY_VALUES, same

POLICY = MemoryPolicy(**TEST_POLICY_VALUES)


@pytest.fixture
async def graph(tmp_path: Path) -> AsyncIterator[SQLiteGraphStore]:
    async with SQLiteGraphStore(tmp_path / "lucy.db", HashingEmbedder(), clock=Clock()) as g:
        yield g


def resolver(graph: SQLiteGraphStore, *answers: bool) -> tuple[EntityResolver, FakeLLM]:
    llm = FakeLLM([same(a) for a in answers])
    return EntityResolver(graph, LLMGateway({"extraction": llm}), policy=POLICY), llm


def test_type_compatibility() -> None:
    assert types_compatible("person", "Person")
    assert types_compatible("city", "place")
    assert types_compatible("company", "organization")
    assert types_compatible("other", "place")
    assert not types_compatible("person", "place")


@pytest.mark.parametrize("mention", ["user", "I", "me", "My", "the user", "User"])
async def test_references_to_the_user(graph: SQLiteGraphStore, mention: str) -> None:
    r, llm = resolver(graph)
    result = await r.resolve(mention, "person")
    assert result.entity.id == USER_ENTITY_ID and result.method is MatchMethod.USER
    assert llm.calls == []


async def test_the_users_name_and_aliases_mean_the_user(graph: SQLiteGraphStore) -> None:
    await graph.rename_entity(USER_ENTITY_ID, "Arjun", keep_old_as_alias=False)
    await graph.add_alias(USER_ENTITY_ID, "Arj")
    r, _ = resolver(graph)
    assert (await r.resolve("arjun", "person")).entity.id == USER_ENTITY_ID
    assert (await r.resolve("Arj", "person")).method is MatchMethod.USER
    assert await r.is_user("ARJUN") and not await r.is_user("Meera")


async def test_exact_name_and_alias(graph: SQLiteGraphStore) -> None:
    meera = await graph.create_entity("person", "Meera", aliases=["my sister"])
    r, llm = resolver(graph)
    exact = await r.resolve("meera", "person")
    alias = await r.resolve("My Sister", "person")
    assert (exact.entity.id, exact.method) == (meera.id, MatchMethod.EXACT)
    assert (alias.entity.id, alias.method) == (meera.id, MatchMethod.ALIAS)
    assert llm.calls == []


async def test_exact_name_prefers_a_compatible_type(graph: SQLiteGraphStore) -> None:
    await graph.create_entity("person", "Jordan")
    country = await graph.create_entity("place", "Jordan")
    r, llm = resolver(graph)
    result = await r.resolve("Jordan", "country")
    assert result.entity.id == country.id and llm.calls == []


async def test_highly_similar_matches_without_asking_and_adds_an_alias(
    graph: SQLiteGraphStore,
) -> None:
    clinic = await graph.create_entity("organisation", "St Clement Hospital")
    r, llm = resolver(graph)
    # The hashing embedder strips a trailing "s": identical vectors, different text.
    result = await r.resolve("St Clements Hospital", "organisation")
    assert result.entity.id == clinic.id and result.method is MatchMethod.SIMILAR
    assert result.score is not None and result.score >= POLICY.entity_match_similarity
    assert llm.calls == []
    stored = await graph.get_entity(clinic.id)
    assert stored is not None and stored.aliases == ["St Clements Hospital"]


async def test_ambiguous_similarity_confirmed_by_the_model(graph: SQLiteGraphStore) -> None:
    meera = await graph.create_entity("person", "Meera")
    await graph.add_statement(new("Meera lives in Leeds", subject_id=meera.id))
    r, llm = resolver(graph, True)
    result = await r.resolve("Meera Rao", "person", context="Meera Rao is a vet.")
    assert result.entity.id == meera.id and result.method is MatchMethod.CONFIRMED
    prompt = llm.calls[0][-1].content
    assert 'Mention: "Meera Rao" (type: person)' in prompt
    assert 'Sentence it appears in: "Meera Rao is a vet."' in prompt
    assert 'Known entity: "Meera" (type: person)' in prompt
    assert "- Meera lives in Leeds" in prompt
    assert llm.calls[0][0].content.startswith("You decide whether a name")
    stored = await graph.get_entity(meera.id)
    assert stored is not None and stored.aliases == ["Meera Rao"]


async def test_ambiguous_similarity_rejected_by_the_model(graph: SQLiteGraphStore) -> None:
    meera = await graph.create_entity("person", "Meera")
    r, llm = resolver(graph, False)
    result = await r.resolve("Meera Rao", "person")
    assert result.method is MatchMethod.CREATED and result.entity.id != meera.id
    assert result.entity.name == "Meera Rao" and len(llm.calls) == 1
    assert len(await graph.find_entities("Meera")) == 1


async def test_a_type_mismatch_counts_against_a_match(graph: SQLiteGraphStore) -> None:
    person = await graph.create_entity("person", "Jordan")
    r, llm = resolver(graph, False)
    result = await r.resolve("Jordan", "place")
    # Identical name, but person vs place: not matched by name, and the
    # penalised similarity is only in the ambiguous band, so the model is asked.
    assert len(llm.calls) == 1
    assert result.method is MatchMethod.CREATED and result.entity.id != person.id
    assert result.entity.type == "place"


async def test_dissimilar_creates_an_entity_without_asking(graph: SQLiteGraphStore) -> None:
    await graph.create_entity("person", "Meera")
    r, llm = resolver(graph)
    result = await r.resolve("  Kiran  ", "person")
    assert result.method is MatchMethod.CREATED and result.entity.name == "Kiran"
    assert llm.calls == []


async def test_blank_name_is_rejected(graph: SQLiteGraphStore) -> None:
    r, _ = resolver(graph)
    with pytest.raises(ValueError):
        await r.resolve("  ", "person")


async def test_resolutions_are_cached_until_reset(graph: SQLiteGraphStore) -> None:
    meera = await graph.create_entity("person", "Meera")
    r, llm = resolver(graph, True, True)
    await r.resolve("Meera Rao", "person")
    await r.resolve("meera rao", "person")
    assert len(llm.calls) == 1
    r.reset()
    # After the alias was added, the name now matches exactly.
    result = await r.resolve("Meera Rao", "person")
    assert result.entity.id == meera.id and result.method is MatchMethod.ALIAS
    assert len(llm.calls) == 1
