"""Retrieval latency at personal scale. Run with `uv run pytest -m slow -s`."""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

import pytest

from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.models import Origin, StatementKind
from lucy.memory.retrieval import Retriever
from tests.graph_support import HashingEmbedder, new

pytestmark = pytest.mark.slow

STATEMENTS = 20_000
ENTITIES = 1_000
DIMENSIONS = 768  # nomic-embed-text
QUERIES = 50


def _ms(seconds: list[float]) -> str:
    ordered = sorted(seconds)
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    return (
        f"p50 {statistics.median(ordered) * 1000:.1f} ms, p95 {p95 * 1000:.1f} ms,"
        f" max {ordered[-1] * 1000:.1f} ms"
    )


async def test_retrieval_latency_at_personal_scale(tmp_path: Path) -> None:
    rng = random.Random(7)
    vocabulary = [f"word{i}" for i in range(3_000)]
    embedder = HashingEmbedder(dimensions=DIMENSIONS)
    path = tmp_path / "bench.db"
    kinds = [k for k in StatementKind if k is not StatementKind.INSTRUCTION]

    def sentence() -> str:
        return " ".join(rng.choices(vocabulary, k=rng.randint(6, 12)))

    async with SQLiteGraphStore(path, embedder) as graph:
        started = time.perf_counter()
        entity_ids = [1]
        for i in range(ENTITIES):
            entity = await graph.create_entity("thing", f"entity{i}", summary=sentence())
            entity_ids.append(entity.id)
        entities_s = time.perf_counter() - started

        started = time.perf_counter()
        batch = []
        for i in range(STATEMENTS):
            batch.append(
                (
                    new(
                        sentence(),
                        subject_id=rng.choice(entity_ids) if i % 3 else 1,
                        predicate=f"pred{rng.randint(0, 200)}",
                        kind=rng.choice(kinds),
                        origin=rng.choice(list(Origin)),
                        confidence=rng.random(),
                    ),
                    [],
                )
            )
            if len(batch) == 1_000:
                await graph.add_statements(batch)
                batch = []
        statements_s = time.perf_counter() - started
        for _ in range(50):
            await graph.add_statement(
                new(sentence(), kind=StatementKind.INSTRUCTION, confidence=1.0)
            )

        retriever = Retriever(
            graph,
            embedder,
            vector_min_similarity=0.1,
            vector_margin=None,
            keyword_min_similarity=None,
            max_statements=None,
        )
        retrieve_times, similar_times, contexts = [], [], []
        for i in range(QUERIES):
            query = f"what about entity{rng.randint(0, ENTITIES - 1)} and {sentence()}?"
            started = time.perf_counter()
            contexts.append(await retriever.retrieve(query, 1_000))
            retrieve_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            await graph.similar_statements(sentence(), subject_id=1 if i % 2 else None, k=10)
            similar_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        await graph.drop_vector_index()
        rebuilt = await graph.reembed_missing(batch_size=500)
        rebuild_s = time.perf_counter() - started

    size_mb = sum(p.stat().st_size for p in tmp_path.iterdir()) / 1e6
    print(
        f"\n{STATEMENTS} statements, {ENTITIES} entities, {DIMENSIONS} dims, {size_mb:.0f} MB"
        f"\n  insert entities one by one: {entities_s:.1f} s"
        f"\n  insert statements in batches of 1000: {statements_s:.1f} s"
        f"\n  retrieve (3 channels, fusion, packing): {_ms(retrieve_times)}"
        f"\n  similar_statements k=10: {_ms(similar_times)}"
        f"\n  drop + rebuild vector index ({rebuilt} vectors): {rebuild_s:.1f} s"
    )
    assert all(context.statements for context in contexts)
    assert all(len(context.instructions) == 50 for context in contexts)
    assert rebuilt == STATEMENTS + 50 + ENTITIES + 1
    assert statistics.median(retrieve_times) < 1.0
