"""Test doubles and helpers for the knowledge-graph tests."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from lucy.core.events import AnyEvent
from lucy.core.types import EmbedderCaps
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.models import NewStatement, Origin, StatementKind
from lucy.memory.sqlite_store import SQLiteMemoryStore

T0 = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)

#: `Retriever` thresholds that suit `HashingEmbedder` (bag-of-words cosine: unrelated
#: texts score 0, one shared word of a few about 0.3 to 0.5). The defaults were
#: measured for embeddinggemma-2, whose scores sit in a high, narrow band.
HASHING_RETRIEVAL: dict[str, Any] = {
    "vector_min_similarity": 0.3,
    "vector_margin": None,
    "keyword_min_similarity": None,
    "entity_min_similarity": 0.5,
    "max_statements": None,
}

_STOP = frozenset(
    "a an the is are was were be to of in on at and or for with my your i you me do does what"
    " who where when how about that this it his her their has have had s".split()
)


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


class HashingEmbedder:
    """A deterministic bag-of-words embedder.

    Each content word (lowercased, stopwords dropped, a trailing "s" stripped,
    then mapped through `synonyms`) adds 1 to a dimension chosen by hashing it;
    the vector is L2-normalised. So texts sharing words are near each other,
    and `synonyms` makes chosen different words land together (what a real
    model does and keyword search cannot). Text without content words maps to
    a fixed unit vector. `kind` does not change the vector but is recorded.

    `fail` makes every call raise; `gate`, if set, is awaited before
    returning (to hold an embedding call open), and `entered` is set once a
    call is waiting on it.
    """

    def __init__(
        self,
        dimensions: int = 256,
        model: str = "hash-bow",
        synonyms: dict[str, str] | None = None,
    ) -> None:
        self.capabilities = EmbedderCaps(dimensions=dimensions, model=model)
        self.synonyms = synonyms or {}
        self.calls: list[tuple[list[str], str]] = []
        self.fail = False
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    def words(self, text: str) -> list[str]:
        result = []
        for word in re.findall(r"\w+", text.lower()):
            if word in _STOP:
                continue
            if len(word) > 3 and word.endswith("s"):
                word = word[:-1]
            result.append(self.synonyms.get(word, word))
        return result

    def vector(self, text: str) -> list[float]:
        dims = self.capabilities.dimensions
        vec = [0.0] * dims
        for word in self.words(text):
            digest = hashlib.sha256(word.encode()).digest()
            vec[1 + int.from_bytes(digest[:4], "big") % (dims - 1)] += 1.0
        if not any(vec):
            vec[0] = 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    async def embed(
        self, texts: Sequence[str], *, kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        self.calls.append((list(texts), kind))
        if self.fail:
            raise RuntimeError("embedder is down")
        if self.gate is not None:
            self.entered.set()
            await self.gate.wait()
        return [self.vector(t) for t in texts]


def new(
    text: str,
    *,
    subject_id: int = 1,
    predicate: str | None = None,
    kind: StatementKind = StatementKind.FACT,
    origin: Origin = Origin.STATED,
    confidence: float = 0.9,
    **kwargs: Any,
) -> NewStatement:
    """A `NewStatement` with defaults; the predicate defaults to the text."""
    return NewStatement(
        subject_id=subject_id,
        predicate=predicate or text,
        text=text,
        kind=kind,
        origin=origin,
        confidence=confidence,
        **kwargs,
    )


async def log(memory: SQLiteMemoryStore, event: AnyEvent) -> int:
    """Append `event` to the episode log and return its row id."""
    await memory.append_episode(event)
    conn = await memory._connection()
    async with conn.execute("SELECT id FROM episodes WHERE event_id = ?", (event.id,)) as cur:
        row = await cur.fetchone()
    assert row is not None
    return int(row[0])


def query(path: Path, sql: str, params: Sequence[Any] = ()) -> list[Any]:
    """Run `sql` on a fresh plain connection (no sqlite-vec) and return the rows."""
    with sqlite3.connect(path) as conn:
        return conn.execute(sql, params).fetchall()


def fts_consistent(path: Path) -> None:
    """Assert both full-text indexes match their content tables exactly."""
    with sqlite3.connect(path) as conn:
        for table in ("statements_fts", "entities_fts"):
            conn.execute(f"INSERT INTO {table}({table}, rank) VALUES ('integrity-check', 1)")
        # Every row is findable by its own words, and nothing else is indexed.
        rows = conn.execute("SELECT id FROM statements").fetchall()
        indexed = conn.execute("SELECT count(*) FROM statements_fts_docsize").fetchone()[0]
        assert indexed == len(rows)
        rows = conn.execute("SELECT id FROM entities").fetchall()
        indexed = conn.execute("SELECT count(*) FROM entities_fts_docsize").fetchone()[0]
        assert indexed == len(rows)


async def vector_ids(graph: SQLiteGraphStore, table: Literal["statements", "entities"]) -> set[int]:
    """Ids with a vector in the graph's current space (empty if it has none)."""
    conn = await graph._connection()
    space = await graph._vectors.current(conn)
    if space is None:
        return set()
    name = space.statements if table == "statements" else space.entities
    async with conn.execute(f"SELECT rowid FROM {name}") as cur:
        return {row[0] for row in await cur.fetchall()}
