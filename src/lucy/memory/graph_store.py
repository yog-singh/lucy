"""The knowledge graph in SQLite: entities, statements, provenance, profile (docs/memory.md).

`SQLiteGraphStore` is the storage API the learning loop (extract, reconcile,
reflect, decay, rewrite profile) and retrieval are built on. It does no LLM
work and makes no reconciliation decisions; it stores and finds.

Conventions:

- It opens its own connection to the shared Lucy file through `open_database`
  (lazily, or on entering `async with`) and loads sqlite-vec into it.
- Every write method is one `BEGIN IMMEDIATE` transaction, so it is atomic
  (a crash or cancellation leaves it either fully done or not at all) and
  serialises correctly against other connections and processes. Calls on one
  store are serialised by a lock; concurrent coroutines are safe.
- Embedding happens *before* the write transaction starts, so a slow embedder
  never holds the database write lock. Embedder errors propagate and nothing is
  written. Documents are embedded with `kind="document"`.
- Entity ids passed in may refer to merged entities; they are resolved to the
  surviving entity (following `merged_into`) wherever an entity is looked up
  or referenced.
- A statement is *current* at time t if its `invalid_at` is unset or after t
  ("now" comes from the injected clock). A statement whose `valid_from` is in
  the future is current (it is known, and rendered with its date).
- Missing ids: `get_*` return None (or skip them in lists); mutations raise
  `GraphNotFoundError`. Constraint violations (unknown episode ids, bad
  references, inconsistent dates) raise `GraphStoreError` and change nothing.
- Timestamps are stored as fixed-width UTC ISO 8601 text, so text order is
  time order.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
from collections.abc import AsyncIterator, Callable, Collection, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Literal, Self

import aiosqlite

from lucy.core.events import AssistantMessage, ToolFinished, UserMessage, event_from_json
from lucy.core.jobs import utc_now
from lucy.core.ports import Embedder
from lucy.memory import fts
from lucy.memory.db import MemoryStoreError, open_database
from lucy.memory.models import (
    DEFAULT_USER_NAME,
    USER_ENTITY_ID,
    Entity,
    EpisodeRecord,
    ForgetResult,
    GraphStats,
    NewStatement,
    Origin,
    ProfileVersion,
    ResetResult,
    ScoredEntity,
    ScoredStatement,
    Statement,
    StatementKind,
    VectorStatus,
)
from lucy.memory.vector_index import VectorIndex, load_vec, to_us

log = logging.getLogger(__name__)

#: Persisted event types that carry conversational content worth extracting.
CONTENT_EPISODE_TYPES: Final[tuple[str, ...]] = (
    "user_message",
    "assistant_message",
    "tool_finished",
)
#: Texts shorter than this are not used to find other turns' traces when
#: forgetting (they would match unrelated traces); the forgotten turns' own
#: traces are always deleted.
MIN_TRACE_NEEDLE: Final = 12
#: Merge chains longer than this are treated as corrupt.
_MAX_CHAIN: Final = 64
#: `memory_meta` keys starting with this describe the graph's current contents
#: (decay state, profile input hash, summary fingerprints, rejected merges); they
#: are cleared by `reset_graph`. Other keys (e.g. when maintenance last ran) are kept.
DERIVED_META_PREFIX: Final = "derived."


class GraphStoreError(MemoryStoreError):
    """A graph operation was invalid (bad reference, inconsistent dates, conflict...)."""


class GraphNotFoundError(GraphStoreError, LookupError):
    """A statement, entity or episode that a mutation refers to does not exist."""


# --- Encoding -----------------------------------------------------------------


def _ts(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError(f"naive datetime {value!r}; graph times must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _opt_ts(value: datetime | None) -> str | None:
    return None if value is None else _ts(value)


def _dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _ids(ids: Iterable[int]) -> str:
    """A JSON array of unique ids, for `IN (SELECT value FROM json_each(?))`."""
    return json.dumps(sorted({int(i) for i in ids}))


def _kinds(kinds: Collection[StatementKind | str] | None) -> list[str] | None:
    return None if kinds is None else sorted({StatementKind(k).value for k in kinds})


_ENTITY_COLUMNS = "id, type, name, aliases, summary, merged_into, created_at, updated_at"


def _entity(row: Sequence[Any]) -> Entity:
    return Entity(
        id=row[0],
        type=row[1],
        name=row[2],
        aliases=json.loads(row[3]),
        summary=row[4],
        merged_into=row[5],
        created_at=datetime.fromisoformat(row[6]),
        updated_at=datetime.fromisoformat(row[7]),
    )


_STATEMENT_COLUMNS = (
    "s.id, s.subject_id, s.predicate, s.object_id, s.object_value, s.text, s.kind, s.origin,"
    " s.confidence, s.evidence_count, s.valid_from, s.invalid_at, s.superseded_by,"
    " s.recorded_at, s.last_observed_at,"
    " (SELECT json_group_array(episode_id) FROM (SELECT episode_id FROM statement_sources"
    " WHERE statement_id = s.id ORDER BY episode_id))"
)


def _statement(row: Sequence[Any]) -> Statement:
    return Statement(
        id=row[0],
        subject_id=row[1],
        predicate=row[2],
        object_id=row[3],
        object_value=row[4],
        text=row[5],
        kind=StatementKind(row[6]),
        origin=Origin(row[7]),
        confidence=row[8],
        evidence_count=row[9],
        valid_from=_dt(row[10]),
        invalid_at=_dt(row[11]),
        superseded_by=row[12],
        recorded_at=datetime.fromisoformat(row[13]),
        last_observed_at=datetime.fromisoformat(row[14]),
        sources=json.loads(row[15]),
    )


def _episode(row: Sequence[Any]) -> EpisodeRecord:
    return EpisodeRecord(id=row[0], event=event_from_json(row[1]), extracted_at=_dt(row[2]))


def entity_embedding_text(name: str, summary: str | None) -> str:
    """The text embedded for an entity: its name, plus its summary when it has one."""
    return name if not summary else f"{name}: {summary}"


def _clean_aliases(name: str, aliases: Iterable[str]) -> list[str]:
    """Strip, drop blanks, the name itself and case-insensitive duplicates; keep order."""
    seen = {fts.normalise(name)}
    result = []
    for alias in aliases:
        alias = alias.strip()
        key = fts.normalise(alias)
        if key and key not in seen:
            seen.add(key)
            result.append(alias)
    return result


def _episode_text(payload: str) -> str:
    event = event_from_json(payload)
    if isinstance(event, UserMessage | AssistantMessage):
        return event.text
    if isinstance(event, ToolFinished):
        return event.result.content
    return ""


def _prefix_range(prefix: str) -> tuple[str, str] | None:
    """Bounds `[low, high)` of the keys starting with `prefix` (None: every key)."""
    if not prefix:
        return None
    return prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1)


def _json_needle(text: str) -> str:
    """`text` as it appears inside a JSON string (traces are stored as JSON)."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


async def _all(conn: aiosqlite.Connection, sql: str, params: Sequence[Any] = ()) -> list[Any]:
    async with conn.execute(sql, params) as cursor:
        return list(await cursor.fetchall())


async def _one(conn: aiosqlite.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
    rows = await _all(conn, sql, params)
    return rows[0] if rows else None


async def _begin(conn: aiosqlite.Connection, sql: str) -> None:
    """Execute `sql` (a BEGIN); if cancelled meanwhile, end the transaction it opened.

    aiosqlite still runs a statement whose await was cancelled. Without this, a
    cancellation while BEGIN waits (e.g. for another connection's write lock)
    left the connection inside a transaction, holding the write lock until it
    was closed, and the next BEGIN on it failed.
    """
    begin = asyncio.ensure_future(conn.execute(sql))
    try:
        await asyncio.shield(begin)
    except asyncio.CancelledError:
        cleanup = asyncio.ensure_future(_undo_begin(conn, begin))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                pass  # cancelled again: still finish the cleanup
        raise


async def _undo_begin(conn: aiosqlite.Connection, begin: asyncio.Future[Any]) -> None:
    try:
        await begin
    except Exception:
        return  # BEGIN failed: no transaction was opened
    if conn.in_transaction:
        await conn.execute("ROLLBACK")


# --- The store ----------------------------------------------------------------


class SQLiteGraphStore:
    """The knowledge graph in the Lucy SQLite file.

    `embedder` embeds statement texts and entity names/summaries; its
    `capabilities.model` and `dimensions` select the vector space (see
    `lucy.memory.vector_index`). `clock` supplies "now" (UTC).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        embedder: Embedder,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.path = Path(path)
        self._embedder = embedder
        self._clock = clock
        caps = embedder.capabilities
        self._vectors = VectorIndex(caps.model, caps.dimensions)
        self._conn: aiosqlite.Connection | None = None
        self._open_lock = asyncio.Lock()
        self._lock = asyncio.Lock()

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # --- Connection -----------------------------------------------------------

    async def _connection(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        async with self._open_lock:
            if self._conn is None:
                conn = await open_database(self.path)
                try:
                    await load_vec(conn)
                except BaseException as exc:
                    await conn.close()
                    if isinstance(exc, sqlite3.Error | AttributeError):
                        raise GraphStoreError(f"cannot load sqlite-vec: {exc}") from exc
                    raise
                self._conn = conn
            return self._conn

    async def close(self) -> None:
        """Close the connection. The store reconnects if used again."""
        async with self._open_lock:
            if self._conn is not None:
                conn, self._conn = self._conn, None
                async with self._lock:
                    await conn.close()

    async def __aenter__(self) -> Self:
        await self._connection()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @asynccontextmanager
    async def _write(self, *, dry_run: bool = False) -> AsyncIterator[aiosqlite.Connection]:
        """One `BEGIN IMMEDIATE` transaction; committed on success unless `dry_run`."""
        conn = await self._connection()
        async with self._lock:
            await _begin(conn, "BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException as exc:
                if conn.in_transaction:
                    await conn.execute("ROLLBACK")
                if isinstance(exc, sqlite3.IntegrityError):
                    raise GraphStoreError(f"graph constraint violated: {exc}") from exc
                raise
            if dry_run:
                await conn.execute("ROLLBACK")
            else:
                try:
                    await conn.execute("COMMIT")
                except BaseException:
                    if conn.in_transaction:
                        await conn.execute("ROLLBACK")
                    raise

    @asynccontextmanager
    async def _read(self) -> AsyncIterator[aiosqlite.Connection]:
        """A read transaction: every query inside sees one consistent snapshot."""
        conn = await self._connection()
        async with self._lock:
            await _begin(conn, "BEGIN")
            try:
                yield conn
            finally:
                if conn.in_transaction:
                    await conn.execute("COMMIT")

    def _now(self) -> datetime:
        return self._clock().astimezone(UTC)

    async def _embed(
        self, texts: Sequence[str], kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors = await self._embedder.embed(list(texts), kind=kind)
        dims = self._vectors.dimensions
        if len(vectors) != len(texts) or any(len(v) != dims for v in vectors):
            raise GraphStoreError(
                f"embedder {self._vectors.model!r} returned {len(vectors)} vectors for"
                f" {len(texts)} texts, or vectors not of its declared {dims} dimensions"
            )
        return [[float(x) for x in v] for v in vectors]

    async def embed_query(self, text: str) -> list[float]:
        """Embed `text` as a search query with this store's embedder."""
        [vector] = await self._embed([text], "query")
        return vector

    # --- Entity internals -----------------------------------------------------

    async def _root(self, conn: aiosqlite.Connection, entity_id: int) -> int:
        """The surviving entity `entity_id` resolves to; raises if it does not exist."""
        current = entity_id
        for _ in range(_MAX_CHAIN):
            row = await _one(conn, "SELECT merged_into FROM entities WHERE id = ?", (current,))
            if row is None:
                raise GraphNotFoundError(f"entity {entity_id} does not exist")
            if row[0] is None:
                return current
            current = row[0]
        raise GraphStoreError(f"entity {entity_id} has a merge chain longer than {_MAX_CHAIN}")

    async def _entity_row(self, conn: aiosqlite.Connection, entity_id: int) -> Entity | None:
        row = await _one(conn, f"SELECT {_ENTITY_COLUMNS} FROM entities WHERE id = ?", (entity_id,))
        return None if row is None else _entity(row)

    async def _entities(self, conn: aiosqlite.Connection, ids: Iterable[int]) -> list[Entity]:
        rows = await _all(
            conn,
            f"SELECT {_ENTITY_COLUMNS} FROM entities"
            " WHERE id IN (SELECT value FROM json_each(?)) ORDER BY id",
            (_ids(ids),),
        )
        return [_entity(row) for row in rows]

    async def _roots(self, conn: aiosqlite.Connection, entities: Iterable[Entity]) -> list[Entity]:
        """The surviving entities for `entities`, deduplicated, in first-seen order."""
        seen: dict[int, Entity] = {}
        for entity in entities:
            root = entity
            if entity.merged_into is not None:
                found = await self._entity_row(conn, await self._root(conn, entity.id))
                assert found is not None
                root = found
            seen.setdefault(root.id, root)
        return list(seen.values())

    async def _put_entity_vector(
        self, conn: aiosqlite.Connection, entity_id: int, embedded: str, vector: list[float]
    ) -> bool:
        """Store `vector` for `entity_id` if `embedded` is still its embedding text.

        Otherwise (it changed concurrently, or the entity was merged away) drop its
        vectors so `reembed_missing` picks it up. Returns whether it was stored.
        """
        entity = await self._entity_row(conn, entity_id)
        await self._vectors.delete_entities(conn, [entity_id])
        if entity is None or entity.merged_into is not None:
            return False
        if entity_embedding_text(entity.name, entity.summary) != embedded:
            return False
        space = await self._vectors.ensure_current(conn, _ts(self._now()))
        await self._vectors.put_entities(conn, space, [(entity_id, vector)])
        return True

    # --- Entities -------------------------------------------------------------

    async def user_entity(self) -> Entity:
        """The entity representing the user (id `USER_ENTITY_ID`, type "person").

        It is created by the schema migration with the name "User", renamed
        when the real name is learned, and is never deleted or merged away.
        """
        entity = await self.get_entity(USER_ENTITY_ID)
        if entity is None:  # pragma: no cover - only if someone deleted it by hand
            raise GraphStoreError("the user entity is missing from the database")
        return entity

    async def create_entity(
        self,
        type: str,
        name: str,
        *,
        aliases: Iterable[str] = (),
        summary: str | None = None,
    ) -> Entity:
        """Create an entity and embed its name plus summary.

        `name` and `type` are stripped and must not be blank; `aliases` are
        cleaned (stripped, de-duplicated case-insensitively, without the name).
        Does not check for an existing entity of the same name: resolve first
        with `find_entities` / `similar_entities`.
        """
        name, type = name.strip(), type.strip()
        if not name or not type:
            raise ValueError("entity name and type must not be blank")
        clean = _clean_aliases(name, aliases)
        embedded = entity_embedding_text(name, summary)
        [vector] = await self._embed([embedded])
        now = _ts(self._now())
        async with self._write() as conn:
            row = await _one(
                conn,
                "INSERT INTO entities (type, name, aliases, summary, created_at, updated_at)"
                f" VALUES (?, ?, ?, ?, ?, ?) RETURNING {_ENTITY_COLUMNS}",
                (type, name, json.dumps(clean), summary, now, now),
            )
            await self._put_entity_vector(conn, row[0], embedded, vector)
        return _entity(row)

    async def get_entity(self, entity_id: int, *, resolve: bool = False) -> Entity | None:
        """The entity `entity_id`, or None. With `resolve`, the entity it was merged into."""
        async with self._read() as conn:
            if resolve:
                try:
                    entity_id = await self._root(conn, entity_id)
                except GraphNotFoundError:
                    return None
            return await self._entity_row(conn, entity_id)

    async def get_entities(self, entity_ids: Iterable[int]) -> list[Entity]:
        """The existing entities among `entity_ids` (not resolved), ascending by id."""
        async with self._read() as conn:
            return await self._entities(conn, entity_ids)

    async def list_entities(
        self,
        *,
        type: str | None = None,
        include_merged: bool = False,
        after_id: int | None = None,
        limit: int | None = None,
    ) -> list[Entity]:
        """Entities ascending by id, for housekeeping; page with `after_id`."""
        where, params = ["1"], []
        if type is not None:
            where.append("type = ?")
            params.append(type)
        if not include_merged:
            where.append("merged_into IS NULL")
        if after_id is not None:
            where.append("id > ?")
            params.append(after_id)
        params.append(-1 if limit is None else limit)
        async with self._read() as conn:
            rows = await _all(
                conn,
                f"SELECT {_ENTITY_COLUMNS} FROM entities WHERE {' AND '.join(where)}"
                " ORDER BY id LIMIT ?",
                params,
            )
        return [_entity(row) for row in rows]

    async def find_entities(self, name: str) -> list[Entity]:
        """Surviving entities whose name or an alias equals `name`, ascending by id.

        Comparison ignores case, Unicode width forms and repeated whitespace,
        but not punctuation or accents. Matches on merged entities resolve to
        the entity they were merged into. Several results mean the name is
        ambiguous ("Priya" the sister and "Priya" the colleague).
        """
        key = fts.normalise(name)
        if not key:
            return []
        query = fts.phrase_query(name)
        async with self._read() as conn:
            if query is None:
                rows = await _all(conn, f"SELECT {_ENTITY_COLUMNS} FROM entities ORDER BY id")
            else:
                rows = await _all(
                    conn,
                    f"SELECT {', '.join('e.' + c for c in _ENTITY_COLUMNS.split(', '))}"
                    " FROM entities_fts f JOIN entities e ON e.id = f.rowid"
                    " WHERE entities_fts MATCH ? ORDER BY e.id",
                    (query,),
                )
            matches = [e for e in map(_entity, rows) if key in {fts.normalise(n) for n in e.names}]
            return sorted(await self._roots(conn, matches), key=lambda e: e.id)

    async def entities_mentioned_in(self, text: str, *, candidates: int = 200) -> list[Entity]:
        """Surviving entities whose name or an alias occurs in `text` as whole words.

        Matching is case-insensitive like `find_entities`. Candidates come from
        the entity full-text index (the best `candidates` keyword matches) and
        are then checked exactly. Longest matched name first.
        """
        haystack = fts.normalise(text)
        query = fts.any_terms_query(text)
        if not haystack or query is None:
            return []
        async with self._read() as conn:
            rows = await _all(
                conn,
                f"SELECT {', '.join('e.' + c for c in _ENTITY_COLUMNS.split(', '))}"
                " FROM entities_fts f JOIN entities e ON e.id = f.rowid"
                " WHERE entities_fts MATCH ? ORDER BY bm25(entities_fts), e.id LIMIT ?",
                (query, candidates),
            )
            hits: list[tuple[int, Entity]] = []
            for entity in map(_entity, rows):
                lengths = [
                    len(key)
                    for key in map(fts.normalise, entity.names)
                    if key and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", haystack)
                ]
                if lengths:
                    hits.append((max(lengths), entity))
            hits.sort(key=lambda hit: (-hit[0], hit[1].id))
            return await self._roots(conn, [entity for _, entity in hits])

    async def similar_entities(
        self,
        text: str,
        k: int = 10,
        *,
        kind: Literal["document", "query"] = "document",
    ) -> list[ScoredEntity]:
        """The `k` surviving entities whose name+summary vectors are nearest to `text`.

        Scores are cosine similarities, best first. `text` is embedded with
        `kind` ("document" by default, as entity resolution compares a mention
        with entity descriptions symmetrically). Entities without a vector for
        the current embedder are not found until `reembed_missing` runs.
        """
        [vector] = await self._embed([text], kind)
        return await self.similar_entities_by_vector(vector, k)

    async def similar_entities_by_vector(
        self, vector: Sequence[float], k: int = 10
    ) -> list[ScoredEntity]:
        """Like `similar_entities` for an already embedded vector of the current model."""
        async with self._read() as conn:
            hits = await self._vectors.search_entities(conn, vector, k)
            by_id = {e.id: e for e in await self._entities(conn, (eid for eid, _ in hits))}
            result: dict[int, ScoredEntity] = {}
            for eid, score in hits:
                entity = by_id.get(eid)
                if entity is None:
                    continue
                [root] = await self._roots(conn, [entity])
                result.setdefault(root.id, ScoredEntity(entity=root, score=score))
        return list(result.values())

    async def add_alias(self, entity_id: int, alias: str) -> Entity:
        """Add `alias` to the (surviving) entity; a no-op if it is already a name or alias."""
        async with self._write() as conn:
            root = await self._root(conn, entity_id)
            entity = await self._entity_row(conn, root)
            assert entity is not None
            aliases = _clean_aliases(entity.name, [*entity.aliases, alias])
            if aliases == entity.aliases:
                return entity
            await conn.execute(
                "UPDATE entities SET aliases = ?, updated_at = ? WHERE id = ?",
                (json.dumps(aliases), _ts(self._now()), root),
            )
            result = await self._entity_row(conn, root)
        assert result is not None
        return result

    async def rename_entity(
        self, entity_id: int, name: str, *, keep_old_as_alias: bool = True
    ) -> Entity:
        """Give the (surviving) entity a new canonical name and re-embed it.

        The old name becomes an alias unless `keep_old_as_alias` is false; the
        new name is removed from the aliases.
        """
        name = name.strip()
        if not name:
            raise ValueError("entity name must not be blank")
        current = await self.get_entity(entity_id, resolve=True)
        if current is None:
            raise GraphNotFoundError(f"entity {entity_id} does not exist")
        embedded = entity_embedding_text(name, current.summary)
        [vector] = await self._embed([embedded])
        async with self._write() as conn:
            root = await self._root(conn, entity_id)
            entity = await self._entity_row(conn, root)
            assert entity is not None
            old = [entity.name] if keep_old_as_alias else []
            aliases = _clean_aliases(name, [*entity.aliases, *old])
            await conn.execute(
                "UPDATE entities SET name = ?, aliases = ?, updated_at = ? WHERE id = ?",
                (name, json.dumps(aliases), _ts(self._now()), root),
            )
            await self._put_entity_vector(conn, root, embedded, vector)
            result = await self._entity_row(conn, root)
        assert result is not None
        return result

    async def update_entity_summary(self, entity_id: int, summary: str | None) -> Entity:
        """Replace the (surviving) entity's summary and re-embed its name plus summary."""
        current = await self.get_entity(entity_id, resolve=True)
        if current is None:
            raise GraphNotFoundError(f"entity {entity_id} does not exist")
        summary = summary.strip() if summary and summary.strip() else None
        embedded = entity_embedding_text(current.name, summary)
        [vector] = await self._embed([embedded])
        async with self._write() as conn:
            root = await self._root(conn, entity_id)
            await conn.execute(
                "UPDATE entities SET summary = ?, updated_at = ? WHERE id = ?",
                (summary, _ts(self._now()), root),
            )
            await self._put_entity_vector(conn, root, embedded, vector)
            result = await self._entity_row(conn, root)
        assert result is not None
        return result

    async def merge_entities(self, source_id: int, target_id: int) -> Entity:
        """Merge `source_id` into `target_id` and return the surviving target.

        Both ids are first resolved to their surviving entities, so merging into
        (or from) an entity that was itself merged works. In one transaction:
        every statement with the source as subject or object is repointed to the
        target; the source's name and aliases become target aliases; the source
        gets `merged_into` = target, as does every entity previously merged into
        the source (chains stay one hop); the source's vectors are removed. No
        statement is lost and lookups by any old name find the target. Merging
        an entity with itself is a no-op. The user entity can absorb others but
        cannot be merged away (`GraphStoreError`).
        """
        async with self._write() as conn:
            source = await self._root(conn, source_id)
            target = await self._root(conn, target_id)
            if source != target:
                if source == USER_ENTITY_ID:
                    raise GraphStoreError(
                        "the user entity cannot be merged into another entity;"
                        " merge the other entity into it instead"
                    )
                src = await self._entity_row(conn, source)
                tgt = await self._entity_row(conn, target)
                assert src is not None and tgt is not None
                now = _ts(self._now())
                aliases = _clean_aliases(tgt.name, [*tgt.aliases, src.name, *src.aliases])
                await conn.execute(
                    "UPDATE entities SET aliases = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(aliases), now, target),
                )
                moved = [
                    row[0]
                    for row in await _all(
                        conn, "SELECT id FROM statements WHERE subject_id = ?", (source,)
                    )
                ]
                await conn.execute(
                    "UPDATE statements SET subject_id = ? WHERE subject_id = ?", (target, source)
                )
                await conn.execute(
                    "UPDATE statements SET object_id = ? WHERE object_id = ?", (target, source)
                )
                await self._vectors.set_subject(conn, moved, target)
                await conn.execute(
                    "UPDATE entities SET merged_into = ?, updated_at = ? WHERE merged_into = ?",
                    (target, now, source),
                )
                await conn.execute(
                    "UPDATE entities SET merged_into = ?, updated_at = ? WHERE id = ?",
                    (target, now, source),
                )
                await self._vectors.delete_entities(conn, [source])
            result = await self._entity_row(conn, target)
        assert result is not None
        return result

    # --- Statement internals --------------------------------------------------

    async def _statements(self, conn: aiosqlite.Connection, ids: Sequence[int]) -> list[Statement]:
        """The existing statements among `ids`, in the order of `ids`."""
        rows = await _all(
            conn,
            f"SELECT {_STATEMENT_COLUMNS} FROM statements s"
            " WHERE s.id IN (SELECT value FROM json_each(?))",
            (_ids(ids),),
        )
        by_id = {row[0]: _statement(row) for row in rows}
        return [by_id[i] for i in dict.fromkeys(ids) if i in by_id]

    async def _link_sources(
        self, conn: aiosqlite.Connection, statement_id: int, episode_ids: Iterable[int]
    ) -> int:
        """Link episodes as sources; returns how many links are new."""
        async with conn.execute(
            "INSERT OR IGNORE INTO statement_sources (statement_id, episode_id)"
            " SELECT ?, value FROM json_each(?)",
            (statement_id, _ids(episode_ids)),
        ) as cursor:
            return cursor.rowcount

    async def _insert_statement(
        self, conn: aiosqlite.Connection, new: NewStatement, now: datetime
    ) -> tuple[int, int]:
        """Insert `new` (entities resolved); returns (statement id, subject id)."""
        subject = await self._root(conn, new.subject_id)
        obj = None if new.object_id is None else await self._root(conn, new.object_id)
        row = await _one(
            conn,
            "INSERT INTO statements (subject_id, predicate, object_id, object_value, text, kind,"
            " origin, confidence, evidence_count, valid_from, invalid_at, recorded_at,"
            " last_observed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?) RETURNING id",
            (
                subject,
                new.predicate,
                obj,
                new.object_value,
                new.text,
                new.kind.value,
                new.origin.value,
                new.confidence,
                _opt_ts(new.valid_from),
                _opt_ts(new.invalid_at),
                _ts(now),
                _ts(new.observed_at or now),
            ),
        )
        return row[0], subject

    def _current_sql(self, alias: str = "s") -> str:
        return f"({alias}.invalid_at IS NULL OR {alias}.invalid_at > ?)"

    # --- Statements -----------------------------------------------------------

    async def add_statement(self, new: NewStatement, episode_ids: Iterable[int] = ()) -> Statement:
        """Add a statement learned from `episode_ids` and embed its text.

        `recorded_at` is now, `last_observed_at` is `new.observed_at` or now,
        `evidence_count` is 1. Subject and object resolve to surviving entities.
        Unknown entity or episode ids raise and nothing is stored.
        """
        [statement] = await self.add_statements([(new, list(episode_ids))])
        return statement

    async def add_statements(
        self, items: Sequence[tuple[NewStatement, Iterable[int]]]
    ) -> list[Statement]:
        """Add several statements in one transaction (one embedder call), in order.

        Same semantics as `add_statement`; all or nothing.
        """
        items = [(new, list(episodes)) for new, episodes in items]
        if not items:
            return []
        vectors = await self._embed([new.text for new, _ in items])
        now = self._now()
        async with self._write() as conn:
            space = await self._vectors.ensure_current(conn, _ts(now))
            ids: list[int] = []
            rows = []
            for (new, episodes), vector in zip(items, vectors, strict=True):
                sid, subject = await self._insert_statement(conn, new, now)
                await self._link_sources(conn, sid, episodes)
                ids.append(sid)
                rows.append((sid, vector, subject, new.kind.value, to_us(new.invalid_at)))
            await self._vectors.put_statements(conn, space, rows)
            return await self._statements(conn, ids)

    async def get_statement(self, statement_id: int) -> Statement | None:
        """The statement `statement_id` with its sources, or None."""
        statements = await self.get_statements([statement_id])
        return statements[0] if statements else None

    async def get_statements(self, statement_ids: Sequence[int]) -> list[Statement]:
        """The existing statements among `statement_ids`, in the given order."""
        async with self._read() as conn:
            return await self._statements(conn, list(statement_ids))

    async def list_statements(
        self,
        *,
        subject_id: int | None = None,
        kinds: Collection[StatementKind | str] | None = None,
        origin: Origin | str | None = None,
        current_only: bool = True,
        observed_before: datetime | None = None,
        min_confidence: float | None = None,
        confidence_below: float | None = None,
        after_id: int | None = None,
        limit: int | None = None,
    ) -> list[Statement]:
        """Statements matching every given filter, ascending by id (page with `after_id`).

        - `subject_id`: about this (resolved) entity as subject.
        - `kinds`: allow-list of kinds; `origin`: stated or inferred.
        - `current_only`: only statements current now (the default).
        - `observed_before`: `last_observed_at` strictly before it (decay).
        - `min_confidence` (inclusive) / `confidence_below` (exclusive).
        """
        kind_list = _kinds(kinds)
        if kind_list == []:
            return []
        where, params = ["1"], []
        async with self._read() as conn:
            if subject_id is not None:
                where.append("s.subject_id = ?")
                params.append(await self._root(conn, subject_id))
            if kind_list is not None:
                where.append(f"s.kind IN ({', '.join('?' for _ in kind_list)})")
                params.extend(kind_list)
            if origin is not None:
                where.append("s.origin = ?")
                params.append(Origin(origin).value)
            if current_only:
                where.append(self._current_sql())
                params.append(_ts(self._now()))
            if observed_before is not None:
                where.append("s.last_observed_at < ?")
                params.append(_ts(observed_before))
            if min_confidence is not None:
                where.append("s.confidence >= ?")
                params.append(min_confidence)
            if confidence_below is not None:
                where.append("s.confidence < ?")
                params.append(confidence_below)
            if after_id is not None:
                where.append("s.id > ?")
                params.append(after_id)
            params.append(-1 if limit is None else limit)
            rows = await _all(
                conn,
                f"SELECT {_STATEMENT_COLUMNS} FROM statements s WHERE {' AND '.join(where)}"
                " ORDER BY s.id LIMIT ?",
                params,
            )
        return [_statement(row) for row in rows]

    async def current_statements(
        self, subject_id: int, *, kinds: Collection[StatementKind | str] | None = None
    ) -> list[Statement]:
        """The current statements about `subject_id` (resolved), optionally of `kinds`."""
        return await self.list_statements(subject_id=subject_id, kinds=kinds)

    async def statements_superseded_by(self, statement_id: int) -> list[Statement]:
        """The statements `statement_id` replaced (their `superseded_by` is it), by id."""
        async with self._read() as conn:
            rows = await _all(
                conn,
                f"SELECT {_STATEMENT_COLUMNS} FROM statements s WHERE s.superseded_by = ?"
                " ORDER BY s.id",
                (statement_id,),
            )
        return [_statement(row) for row in rows]

    async def active_instructions(self) -> list[Statement]:
        """All current instruction statements, oldest first."""
        return await self.list_statements(kinds=[StatementKind.INSTRUCTION])

    async def neighbours(
        self,
        entity_id: int,
        *,
        kinds: Collection[StatementKind | str] | None = None,
        limit: int | None = None,
    ) -> list[Statement]:
        """Current statements with the (resolved) entity as subject or object.

        One hop of graph expansion. Ordered by confidence, then most recently
        observed, then id; optionally only `kinds`, at most `limit`.
        """
        kind_list = _kinds(kinds)
        if kind_list == []:
            return []
        async with self._read() as conn:
            root = await self._root(conn, entity_id)
            params: list[Any] = [root, root, _ts(self._now())]
            kind_sql = ""
            if kind_list is not None:
                kind_sql = f" AND s.kind IN ({', '.join('?' for _ in kind_list)})"
                params.extend(kind_list)
            params.append(-1 if limit is None else limit)
            rows = await _all(
                conn,
                f"SELECT {_STATEMENT_COLUMNS} FROM statements s"
                f" WHERE (s.subject_id = ? OR s.object_id = ?) AND {self._current_sql()}{kind_sql}"
                " ORDER BY s.confidence DESC, s.last_observed_at DESC, s.id LIMIT ?",
                params,
            )
        return [_statement(row) for row in rows]

    async def similar_statements(
        self,
        text: str,
        *,
        subject_id: int | None = None,
        k: int = 10,
        include_invalid: bool = False,
        kinds: Collection[StatementKind | str] | None = None,
        kind: Literal["document", "query"] = "document",
    ) -> list[ScoredStatement]:
        """The `k` statements whose text vectors are nearest to `text`, best first.

        This is what reconciliation uses to find what a candidate reinforces or
        contradicts. Filters (applied inside the vector search, so up to `k`
        matching results are returned): `subject_id` (resolved), `kinds`, and
        current statements only unless `include_invalid`. Scores are cosine
        similarities. `text` is embedded with `kind` ("document" by default:
        comparing a candidate statement with stored statements is symmetric).
        Statements without a vector for the current embedder are not found.
        """
        [vector] = await self._embed([text], kind)
        return await self.similar_statements_by_vector(
            vector, subject_id=subject_id, k=k, include_invalid=include_invalid, kinds=kinds
        )

    async def similar_statements_by_vector(
        self,
        vector: Sequence[float],
        *,
        subject_id: int | None = None,
        k: int = 10,
        include_invalid: bool = False,
        kinds: Collection[StatementKind | str] | None = None,
    ) -> list[ScoredStatement]:
        """Like `similar_statements` for an already embedded vector of the current model."""
        async with self._read() as conn:
            subject = None if subject_id is None else await self._root(conn, subject_id)
            hits = await self._vectors.search_statements(
                conn,
                vector,
                k,
                subject_id=subject,
                kinds=_kinds(kinds),
                current_at=None if include_invalid else self._now(),
            )
            statements = await self._statements(conn, [sid for sid, _ in hits])
        scores = dict(hits)
        return [ScoredStatement(statement=s, score=scores[s.id]) for s in statements]

    async def keyword_statements(
        self,
        text: str,
        *,
        k: int = 20,
        include_invalid: bool = False,
        kinds: Collection[StatementKind | str] | None = None,
    ) -> list[ScoredStatement]:
        """Full-text search of statement texts for any word of `text`, best first.

        `text` may be arbitrary user input: it is reduced to quoted word tokens
        (stopwords dropped), so FTS5 syntax in it is treated literally. Scores
        are -bm25 (higher is better). Empty or word-less text finds nothing.
        """
        query = fts.any_terms_query(text)
        kind_list = _kinds(kinds)
        if query is None or k <= 0 or kind_list == []:
            return []
        where = ["statements_fts MATCH ?"]
        params: list[Any] = [query]
        if not include_invalid:
            where.append(self._current_sql())
            params.append(_ts(self._now()))
        if kind_list is not None:
            where.append(f"s.kind IN ({', '.join('?' for _ in kind_list)})")
            params.extend(kind_list)
        params.append(k)
        async with self._read() as conn:
            rows = await _all(
                conn,
                f"SELECT {_STATEMENT_COLUMNS}, bm25(statements_fts) AS score"
                " FROM statements_fts JOIN statements s ON s.id = statements_fts.rowid"
                f" WHERE {' AND '.join(where)} ORDER BY score, s.id LIMIT ?",
                params,
            )
        return [ScoredStatement(statement=_statement(row), score=-row[-1]) for row in rows]

    async def reinforce(
        self,
        statement_id: int,
        episode_ids: Iterable[int] = (),
        *,
        confidence: float | None = None,
        observed_at: datetime | None = None,
    ) -> Statement:
        """Record another observation supporting a statement.

        Links `episode_ids` as additional sources, sets `confidence` (if given,
        0..1; the caller computes the new value), moves `last_observed_at` to
        `observed_at` (default now) if that is later, and increments
        `evidence_count` by one. Idempotent for retries: if `episode_ids` is
        non-empty and every one of them was already a source, `evidence_count`
        is not incremented again. Works on invalidated statements too (they
        were true in their time).
        """
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be within [0, 1], got {confidence}")
        episodes = list(episode_ids)
        observed = _ts(observed_at or self._now())
        async with self._write() as conn:
            if await _one(conn, "SELECT 1 FROM statements WHERE id = ?", (statement_id,)) is None:
                raise GraphNotFoundError(f"statement {statement_id} does not exist")
            new_links = await self._link_sources(conn, statement_id, episodes)
            bump = 1 if not episodes or new_links > 0 else 0
            await conn.execute(
                "UPDATE statements SET evidence_count = evidence_count + ?,"
                " confidence = coalesce(?, confidence),"
                " last_observed_at = max(last_observed_at, ?) WHERE id = ?",
                (bump, confidence, observed, statement_id),
            )
            [statement] = await self._statements(conn, [statement_id])
        return statement

    async def supersede(
        self, old_id: int, new: NewStatement, episode_ids: Iterable[int] = ()
    ) -> Statement:
        """Replace statement `old_id` with `new`, atomically; returns the new statement.

        Adds `new` (as `add_statement`) and closes the old one: its
        `superseded_by` becomes the new id and its `invalid_at` becomes
        `new.valid_from` if given, else now, or stays at an earlier
        `invalid_at` it already had. Raises `GraphStoreError` (changing
        nothing) if the old statement is already superseded, or if the end
        would fall before its `valid_from`.
        """
        episodes = list(episode_ids)
        [vector] = await self._embed([new.text])
        now = self._now()
        end = new.valid_from or now
        async with self._write() as conn:
            row = await _one(
                conn,
                "SELECT superseded_by, valid_from, invalid_at FROM statements WHERE id = ?",
                (old_id,),
            )
            if row is None:
                raise GraphNotFoundError(f"statement {old_id} does not exist")
            if row[0] is not None:
                raise GraphStoreError(f"statement {old_id} is already superseded by {row[0]}")
            if row[1] is not None and end < datetime.fromisoformat(row[1]):
                raise GraphStoreError(
                    f"cannot end statement {old_id} at {end.isoformat()}, before it became"
                    f" valid at {row[1]}"
                )
            if row[2] is not None:
                end = min(end, datetime.fromisoformat(row[2]))
            sid, subject = await self._insert_statement(conn, new, now)
            await self._link_sources(conn, sid, episodes)
            await conn.execute(
                "UPDATE statements SET invalid_at = ?, superseded_by = ? WHERE id = ?",
                (_ts(end), sid, old_id),
            )
            await self._vectors.set_invalid(conn, [old_id], end)
            space = await self._vectors.ensure_current(conn, _ts(now))
            await self._vectors.put_statements(
                conn, space, [(sid, vector, subject, new.kind.value, to_us(new.invalid_at))]
            )
            [statement] = await self._statements(conn, [sid])
        return statement

    async def invalidate(self, statement_id: int, at: datetime | None = None) -> Statement:
        """Mark a statement as no longer true from `at` (default now), without a successor.

        Overwrites any earlier `invalid_at`. Raises `GraphStoreError` if `at` is
        before the statement's `valid_from`.
        """
        end = datetime.fromisoformat(_ts(at or self._now()))  # rejects naive datetimes
        async with self._write() as conn:
            row = await _one(
                conn, "SELECT valid_from FROM statements WHERE id = ?", (statement_id,)
            )
            if row is None:
                raise GraphNotFoundError(f"statement {statement_id} does not exist")
            if row[0] is not None and end < datetime.fromisoformat(row[0]):
                raise GraphStoreError(
                    f"cannot end statement {statement_id} before it became valid at {row[0]}"
                )
            await conn.execute(
                "UPDATE statements SET invalid_at = ? WHERE id = ?", (_ts(end), statement_id)
            )
            await self._vectors.set_invalid(conn, [statement_id], end)
            [statement] = await self._statements(conn, [statement_id])
        return statement

    async def set_confidence(self, statement_id: int, value: float) -> None:
        """Set one statement's confidence (0..1), e.g. for decay. Nothing else changes."""
        if await self.set_confidences({statement_id: value}) == 0:
            raise GraphNotFoundError(f"statement {statement_id} does not exist")

    async def set_confidences(self, values: Mapping[int, float]) -> int:
        """Set several statements' confidences in one transaction; returns how many existed.

        Unknown ids are skipped. Raises `ValueError` (changing nothing) if any
        value is outside [0, 1].
        """
        for value in values.values():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence must be within [0, 1], got {value}")
        if not values:
            return 0
        async with self._write() as conn:
            async with conn.execute(
                "UPDATE statements SET confidence = j.value ->> 'c' FROM"
                " (SELECT value FROM json_each(?)) AS j WHERE statements.id = j.value ->> 'id'",
                (json.dumps([{"id": k, "c": v} for k, v in values.items()]),),
            ) as cursor:
                return cursor.rowcount

    async def set_predicate(self, statement_ids: Iterable[int], predicate: str) -> int:
        """Rename the predicate of statements (reflection merging near-duplicate predicates).

        Returns how many statements were changed. Text and vectors are unaffected.
        """
        if not predicate.strip():
            raise ValueError("predicate must not be blank")
        async with self._write() as conn:
            async with conn.execute(
                "UPDATE statements SET predicate = ? WHERE id IN (SELECT value FROM json_each(?))",
                (predicate, _ids(statement_ids)),
            ) as cursor:
                return cursor.rowcount

    async def end_statement(
        self, statement_id: int, at: datetime, *, superseded_by: int | None = None
    ) -> Statement:
        """End a statement at `at` because of another, existing statement.

        Unlike `invalidate`, an earlier `invalid_at` is kept (the end is the
        earlier of the two), and `superseded_by`, if given, records the
        statement that ended it (several statements may point at one). Raises
        `GraphNotFoundError` if either statement does not exist, and
        `GraphStoreError` (changing nothing) if `at` is before the statement's
        `valid_from`, if it is already superseded by a different statement, or
        if it would point at itself.
        """
        end = datetime.fromisoformat(_ts(at))  # rejects naive datetimes
        if superseded_by == statement_id:
            raise GraphStoreError(f"statement {statement_id} cannot supersede itself")
        async with self._write() as conn:
            row = await _one(
                conn,
                "SELECT valid_from, invalid_at, superseded_by FROM statements WHERE id = ?",
                (statement_id,),
            )
            if row is None:
                raise GraphNotFoundError(f"statement {statement_id} does not exist")
            if superseded_by is not None:
                if (
                    await _one(conn, "SELECT 1 FROM statements WHERE id = ?", (superseded_by,))
                    is None
                ):
                    raise GraphNotFoundError(f"statement {superseded_by} does not exist")
                if row[2] is not None and row[2] != superseded_by:
                    raise GraphStoreError(
                        f"statement {statement_id} is already superseded by {row[2]}"
                    )
            if row[0] is not None and end < datetime.fromisoformat(row[0]):
                raise GraphStoreError(
                    f"cannot end statement {statement_id} before it became valid at {row[0]}"
                )
            if row[1] is not None:
                end = min(end, datetime.fromisoformat(row[1]))
            await conn.execute(
                "UPDATE statements SET invalid_at = ?, superseded_by = coalesce(?, superseded_by)"
                " WHERE id = ?",
                (_ts(end), superseded_by, statement_id),
            )
            await self._vectors.set_invalid(conn, [statement_id], end)
            [statement] = await self._statements(conn, [statement_id])
        return statement

    async def set_statement_texts(self, texts: Mapping[int, str]) -> int:
        """Replace the text of statements (same meaning, new wording) and re-embed them.

        For example when the user's name is learned, "The user ..." becomes
        "Dana ...". Nothing else changes (not the predicate, dates, sources or
        confidence). Texts are embedded first, then written in one
        transaction; unknown ids are skipped. Returns how many statements
        changed. Blank texts raise `ValueError` and nothing is written.
        """
        items = {sid: text.strip() for sid, text in texts.items()}
        if any(not text for text in items.values()):
            raise ValueError("statement text must not be blank")
        if not items:
            return 0
        ids = list(items)
        vectors = await self._embed([items[sid] for sid in ids])
        changed = 0
        async with self._write() as conn:
            space = await self._vectors.ensure_current(conn, _ts(self._now()))
            rows = []
            for sid, vector in zip(ids, vectors, strict=True):
                current = await _one(
                    conn,
                    "SELECT subject_id, kind, invalid_at, text FROM statements WHERE id = ?",
                    (sid,),
                )
                if current is None or current[3] == items[sid]:
                    continue
                await conn.execute("UPDATE statements SET text = ? WHERE id = ?", (items[sid], sid))
                rows.append((sid, vector, current[0], current[1], to_us(_dt(current[2]))))
            # Vectors of other embedders are stale now; `reembed_missing` redoes them.
            await self._vectors.delete_statements(conn, [row[0] for row in rows])
            await self._vectors.put_statements(conn, space, rows)
            changed = len(rows)
        return changed

    # --- Episodes (for the learning loop) -------------------------------------

    async def unextracted_episodes(self, limit: int = 50) -> list[EpisodeRecord]:
        """Up to `limit` content episodes not yet extracted, oldest first.

        Only `CONTENT_EPISODE_TYPES` are returned; other persisted events never
        need extracting and are never marked.
        """
        if limit <= 0:
            return []
        types = json.dumps(CONTENT_EPISODE_TYPES)
        async with self._read() as conn:
            rows = await _all(
                conn,
                "SELECT id, payload, extracted_at FROM episodes WHERE extracted_at IS NULL"
                " AND type IN (SELECT value FROM json_each(?)) ORDER BY id LIMIT ?",
                (types, limit),
            )
        return [_episode(row) for row in rows]

    async def mark_extracted(self, episode_ids: Iterable[int], at: datetime | None = None) -> int:
        """Set `extracted_at` (default now) on episodes; returns how many exist.

        Re-marking an already extracted episode updates its time.
        """
        async with self._write() as conn:
            async with conn.execute(
                "UPDATE episodes SET extracted_at = ? WHERE id IN (SELECT value FROM json_each(?))",
                (_ts(at or self._now()), _ids(episode_ids)),
            ) as cursor:
                return cursor.rowcount

    async def get_episodes(self, episode_ids: Iterable[int]) -> list[EpisodeRecord]:
        """The existing episodes among `episode_ids`, oldest first (provenance)."""
        async with self._read() as conn:
            rows = await _all(
                conn,
                "SELECT id, payload, extracted_at FROM episodes"
                " WHERE id IN (SELECT value FROM json_each(?)) ORDER BY id",
                (_ids(episode_ids),),
            )
        return [_episode(row) for row in rows]

    async def episodes_mentioning(self, text: str, *, limit: int = 20) -> list[EpisodeRecord]:
        """Up to `limit` content episodes whose text contains `text`, oldest first.

        Case-insensitive for ASCII letters (SQLite `lower`). Blank `text` finds
        nothing. For "forget": finding turns that repeat forgotten information.
        """
        needle = text.strip().lower()
        if not needle or limit <= 0:
            return []
        found: list[EpisodeRecord] = []
        after = 0
        async with self._read() as conn:
            while len(found) < limit:
                # The payload is JSON: prefilter on it, then check the episode's own text.
                rows = await _all(
                    conn,
                    "SELECT id, payload, extracted_at FROM episodes WHERE id > ?"
                    " AND type IN (SELECT value FROM json_each(?))"
                    " AND instr(lower(payload), ?) > 0 ORDER BY id LIMIT 200",
                    (after, json.dumps(CONTENT_EPISODE_TYPES), _json_needle(needle)),
                )
                if not rows:
                    break
                after = rows[-1][0]
                found.extend(
                    _episode(row) for row in rows if needle in _episode_text(row[1]).lower()
                )
        return found[:limit]

    async def turn_episodes(
        self, turn_id: str, *, types: Sequence[str] | None = CONTENT_EPISODE_TYPES
    ) -> list[EpisodeRecord]:
        """The episodes of `turn_id` of the given `types` (None: all), oldest first."""
        sql = "SELECT id, payload, extracted_at FROM episodes WHERE turn_id = ?"
        params: list[Any] = [turn_id]
        if types is not None:
            sql += " AND type IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(list(types)))
        async with self._read() as conn:
            rows = await _all(conn, sql + " ORDER BY id", params)
        return [_episode(row) for row in rows]

    async def episodes_before(
        self, episode_id: int, n: int, *, types: Sequence[str] | None = CONTENT_EPISODE_TYPES
    ) -> list[EpisodeRecord]:
        """Up to `n` episodes of the given `types` (None: all) preceding `episode_id`
        in its session, oldest first: the surrounding conversation for extraction.

        Raises `GraphNotFoundError` if `episode_id` does not exist.
        """
        async with self._read() as conn:
            row = await _one(conn, "SELECT session_id FROM episodes WHERE id = ?", (episode_id,))
            if row is None:
                raise GraphNotFoundError(f"episode {episode_id} does not exist")
            if n <= 0:
                return []
            sql = "SELECT id, payload, extracted_at FROM episodes WHERE session_id = ? AND id < ?"
            params: list[Any] = [row[0], episode_id]
            if types is not None:
                sql += " AND type IN (SELECT value FROM json_each(?))"
                params.append(json.dumps(list(types)))
            rows = await _all(
                conn, f"SELECT * FROM ({sql} ORDER BY id DESC LIMIT ?) ORDER BY id", [*params, n]
            )
        return [_episode(row) for row in rows]

    async def episodes_between(
        self,
        start: datetime,
        end: datetime,
        *,
        types: Sequence[str] | None = ("user_message",),
        after_id: int | None = None,
        limit: int | None = None,
    ) -> list[EpisodeRecord]:
        """Episodes of the given `types` (None: all) whose event time is in `[start, end)`,
        in time order (ties by id); with `after_id`, only those with a larger id.

        For reflection, which looks at what the user said over a period. `limit`
        keeps the earliest. Every matching row is read, so keep the period bounded.
        """
        for bound in (start, end):
            _ts(bound)  # rejects naive datetimes
        # Event times are stored as ISO text whose fractional part may be absent,
        # so compare whole seconds in SQL and the exact times below.
        sql = (
            "SELECT id, payload, extracted_at FROM episodes"
            " WHERE substr(ts, 1, 19) >= ? AND substr(ts, 1, 19) <= ?"
        )
        params: list[Any] = [
            start.astimezone(UTC).isoformat(timespec="seconds")[:19],
            end.astimezone(UTC).isoformat(timespec="seconds")[:19],
        ]
        if types is not None:
            sql += " AND type IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(list(types)))
        if after_id is not None:
            sql += " AND id > ?"
            params.append(after_id)
        async with self._read() as conn:
            rows = await _all(conn, sql, params)
        records = [r for r in map(_episode, rows) if start <= r.event.ts < end]
        records.sort(key=lambda r: (r.event.ts, r.id))
        return records if limit is None else records[: max(limit, 0)]

    async def count_episodes_after(
        self, after_id: int, *, types: Sequence[str] | None = ("user_message",)
    ) -> int:
        """How many episodes of the given `types` (None: all) have an id above `after_id`."""
        sql = "SELECT count(*) FROM episodes WHERE id > ?"
        params: list[Any] = [after_id]
        if types is not None:
            sql += " AND type IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(list(types)))
        async with self._read() as conn:
            row = await _one(conn, sql, params)
        return int(row[0])

    async def statements_recorded_since(
        self,
        since: datetime,
        *,
        subject_id: int | None = None,
        kinds: Collection[StatementKind | str] | None = None,
        limit: int | None = None,
    ) -> list[Statement]:
        """Statements recorded at or after `since` (current or not), newest first.

        Optionally only about the (resolved) `subject_id`, of `kinds`, at most `limit`.
        """
        kind_list = _kinds(kinds)
        if kind_list == []:
            return []
        where = ["s.recorded_at >= ?"]
        params: list[Any] = [_ts(since)]
        async with self._read() as conn:
            if subject_id is not None:
                where.append("s.subject_id = ?")
                params.append(await self._root(conn, subject_id))
            if kind_list is not None:
                where.append(f"s.kind IN ({', '.join('?' for _ in kind_list)})")
                params.extend(kind_list)
            params.append(-1 if limit is None else limit)
            rows = await _all(
                conn,
                f"SELECT {_STATEMENT_COLUMNS} FROM statements s WHERE {' AND '.join(where)}"
                " ORDER BY s.recorded_at DESC, s.id DESC LIMIT ?",
                params,
            )
        return [_statement(row) for row in rows]

    # --- Profile --------------------------------------------------------------

    async def save_profile(self, content: str) -> ProfileVersion:
        """Store a new profile version; it becomes the current one. Old versions are kept."""
        async with self._write() as conn:
            row = await _one(
                conn,
                "INSERT INTO profile_versions (created_at, content) VALUES (?, ?)"
                " RETURNING id, created_at, content",
                (_ts(self._now()), content),
            )
        return ProfileVersion(id=row[0], created_at=datetime.fromisoformat(row[1]), content=row[2])

    async def current_profile(self) -> ProfileVersion | None:
        """The most recently saved profile version, or None."""
        versions = await self.profile_versions(limit=1)
        return versions[0] if versions else None

    async def profile_versions(self, limit: int | None = None) -> list[ProfileVersion]:
        """Profile versions, newest first."""
        async with self._read() as conn:
            rows = await _all(
                conn,
                "SELECT id, created_at, content FROM profile_versions ORDER BY id DESC LIMIT ?",
                (-1 if limit is None else limit,),
            )
        return [
            ProfileVersion(id=r[0], created_at=datetime.fromisoformat(r[1]), content=r[2])
            for r in rows
        ]

    # --- Metadata (memory maintenance state) -----------------------------------

    async def get_meta(self, key: str) -> str | None:
        """The value stored under `key` in `memory_meta`, or None."""
        return (await self.get_meta_many([key])).get(key)

    async def get_meta_many(self, keys: Iterable[str]) -> dict[str, str]:
        """The stored values of those of `keys` that exist."""
        async with self._read() as conn:
            rows = await _all(
                conn,
                "SELECT key, value FROM memory_meta WHERE key IN (SELECT value FROM json_each(?))",
                (json.dumps(sorted(set(keys))),),
            )
        return {row[0]: row[1] for row in rows}

    async def set_meta(self, key: str, value: str) -> None:
        """Store `value` under `key`, replacing any previous value."""
        await self.write_meta({key: value})

    async def delete_meta(self, key: str) -> bool:
        """Delete `key`; returns whether it existed."""
        async with self._write() as conn:
            async with conn.execute("DELETE FROM memory_meta WHERE key = ?", (key,)) as cursor:
                return cursor.rowcount > 0

    async def list_meta(
        self, prefix: str = "", *, after: str | None = None, limit: int | None = None
    ) -> list[tuple[str, str]]:
        """`(key, value)` pairs whose key starts with `prefix`, ascending by key.

        Page with `after` (the last key seen) and `limit`.
        """
        where, params = ["1"], []
        bounds = _prefix_range(prefix)
        if bounds is not None:
            where.append("key >= ? AND key < ?")
            params.extend(bounds)
        if after is not None:
            where.append("key > ?")
            params.append(after)
        params.append(-1 if limit is None else limit)
        async with self._read() as conn:
            rows = await _all(
                conn,
                f"SELECT key, value FROM memory_meta WHERE {' AND '.join(where)}"
                " ORDER BY key LIMIT ?",
                params,
            )
        return [(row[0], row[1]) for row in rows]

    async def _write_meta(
        self, conn: aiosqlite.Connection, values: Mapping[str, str | None]
    ) -> None:
        now = _ts(self._now())
        for key, value in values.items():
            if value is None:
                await conn.execute("DELETE FROM memory_meta WHERE key = ?", (key,))
            else:
                await conn.execute(
                    "INSERT INTO memory_meta (key, value, updated_at) VALUES (?, ?, ?)"
                    " ON CONFLICT (key) DO UPDATE SET value = excluded.value,"
                    " updated_at = excluded.updated_at",
                    (key, value, now),
                )

    async def write_meta(self, values: Mapping[str, str | None]) -> None:
        """Set (or, for a None value, delete) several keys in one transaction."""
        if values:
            async with self._write() as conn:
                await self._write_meta(conn, values)

    async def set_confidences_and_meta(
        self, confidences: Mapping[int, float], meta: Mapping[str, str | None]
    ) -> int:
        """`set_confidences` and `write_meta` in one transaction (all or nothing).

        For maintenance that must record what it changed together with the
        change (decay). Returns how many of the statements existed.
        """
        for value in confidences.values():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence must be within [0, 1], got {value}")
        changed = 0
        async with self._write() as conn:
            if confidences:
                async with conn.execute(
                    "UPDATE statements SET confidence = j.value ->> 'c' FROM"
                    " (SELECT value FROM json_each(?)) AS j WHERE statements.id = j.value ->> 'id'",
                    (json.dumps([{"id": k, "c": v} for k, v in confidences.items()]),),
                ) as cursor:
                    changed = cursor.rowcount
            await self._write_meta(conn, meta)
        return changed

    async def entity_statement_fingerprints(
        self, entity_ids: Iterable[int]
    ) -> dict[int, tuple[int, str]]:
        """For each existing entity among `entity_ids` (not resolved): the number of
        current statements with it as subject or object, and a fingerprint of that set.

        The fingerprint (count, sum and maximum of the statement ids) changes
        whenever such a statement is added, superseded, invalidated (once its
        `invalid_at` has passed), forgotten or moved by a merge; it ignores
        confidence and `last_observed_at`. One query for all ids, for cheap
        staleness checks in maintenance.
        """
        async with self._read() as conn:
            rows = await _all(
                conn,
                "WITH ids(id) AS (SELECT value FROM json_each(?1)),"
                " links(eid, sid) AS ("
                "  SELECT s.subject_id, s.id FROM statements s JOIN ids ON s.subject_id = ids.id"
                f"  WHERE {self._current_sql()}"
                "  UNION"
                "  SELECT s.object_id, s.id FROM statements s JOIN ids ON s.object_id = ids.id"
                f"  WHERE {self._current_sql()})"
                " SELECT e.id, count(l.sid), coalesce(sum(l.sid), 0), coalesce(max(l.sid), 0)"
                " FROM entities e JOIN ids ON e.id = ids.id LEFT JOIN links l ON l.eid = e.id"
                " GROUP BY e.id",
                (_ids(entity_ids), _ts(self._now()), _ts(self._now())),
            )
        return {row[0]: (row[1], f"{row[1]}:{row[2]}:{row[3]}") for row in rows}

    async def stats(self) -> GraphStats:
        """Counts of episodes, statements, entities and profile versions (one snapshot)."""
        now = _ts(self._now())
        async with self._read() as conn:
            episodes, unextracted = await _one(
                conn,
                "SELECT count(*), coalesce(sum(extracted_at IS NULL"
                " AND type IN (SELECT value FROM json_each(?))), 0) FROM episodes",
                (json.dumps(CONTENT_EPISODE_TYPES),),
            )
            by_kind = await _all(
                conn,
                f"SELECT s.kind, count(*) FROM statements s WHERE {self._current_sql()}"
                " GROUP BY s.kind",
                (now,),
            )
            (statements,) = await _one(conn, "SELECT count(*) FROM statements")
            entities, merged = await _one(
                conn,
                "SELECT coalesce(sum(merged_into IS NULL), 0),"
                " coalesce(sum(merged_into IS NOT NULL), 0) FROM entities",
            )
            (profiles,) = await _one(conn, "SELECT count(*) FROM profile_versions")
        current = sum(row[1] for row in by_kind)
        return GraphStats(
            episodes=episodes,
            episodes_unextracted=unextracted,
            statements_current=current,
            statements_invalidated=statements - current,
            current_by_kind={StatementKind(row[0]): row[1] for row in by_kind},
            entities=entities,
            entities_merged=merged,
            profile_versions=profiles,
        )

    # --- Vector index maintenance ---------------------------------------------

    async def vector_status(self) -> VectorStatus:
        """How many statements and (unmerged) entities lack a current-model vector."""
        async with self._read() as conn:
            statements = await self._vectors.missing_statements(conn, None)
            entities = await self._vectors.missing_entities(conn, None)
        return VectorStatus(
            model=self._vectors.model,
            dimensions=self._vectors.dimensions,
            statements_missing=len(statements),
            entities_missing=len(entities),
        )

    async def missing_vectors(self, limit: int | None = None) -> tuple[list[int], list[int]]:
        """Ids of statements, and of unmerged entities, lacking a current-model vector.

        Each list is ascending and holds at most `limit` ids.
        """
        async with self._read() as conn:
            return (
                await self._vectors.missing_statements(conn, limit),
                await self._vectors.missing_entities(conn, limit),
            )

    async def reembed_missing(self, limit: int | None = None, *, batch_size: int = 64) -> int:
        """Embed up to `limit` (None: all) items lacking a current-model vector.

        Works in batches of `batch_size`: read texts, embed outside any
        transaction, then store each batch in its own transaction (statements
        first, then entities). An item changed or deleted while its batch was
        being embedded is skipped and picked up next time. Safe to interrupt
        and to repeat; returns the number of vectors stored.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        stored = 0
        # Scan forward from the last ids seen, so a full re-embed is linear, not quadratic.
        after_statement = after_entity = 0
        while limit is None or stored < limit:
            size = batch_size if limit is None else min(batch_size, limit - stored)
            async with self._read() as conn:
                sids = await self._vectors.missing_statements(conn, size, after_id=after_statement)
                statements = await self._statements(conn, sids)
                eids = await self._vectors.missing_entities(
                    conn, size - len(statements), after_id=after_entity
                )
                entities = await self._entities(conn, eids)
            after_statement = max(sids, default=after_statement)
            after_entity = max(eids, default=after_entity)
            texts = [s.text for s in statements] + [
                entity_embedding_text(e.name, e.summary) for e in entities
            ]
            if not texts:
                break
            vectors = await self._embed(texts)
            written = 0
            async with self._write() as conn:
                space = await self._vectors.ensure_current(conn, _ts(self._now()))
                rows = []
                for statement, vector in zip(statements, vectors, strict=False):
                    current = await _one(
                        conn,
                        "SELECT subject_id, kind, invalid_at, text FROM statements WHERE id = ?",
                        (statement.id,),
                    )
                    if current is None or current[3] != statement.text:
                        continue
                    if await self._vectors.has_statement(conn, space, statement.id):
                        continue
                    rows.append(
                        (statement.id, vector, current[0], current[1], to_us(_dt(current[2])))
                    )
                await self._vectors.put_statements(conn, space, rows)
                written += len(rows)
                for entity, vector in zip(entities, vectors[len(statements) :], strict=True):
                    if await self._vectors.has_entity(conn, space, entity.id):
                        continue
                    embedded = entity_embedding_text(entity.name, entity.summary)
                    if await self._put_entity_vector(conn, entity.id, embedded, vector):
                        written += 1
            stored += written  # items skipped here are behind the cursor; the next call retries
        return stored

    async def drop_vector_index(self) -> int:
        """Drop every vector, of every model; returns the number of spaces dropped.

        The base tables are untouched; `reembed_missing` rebuilds the index.
        """
        async with self._write() as conn:
            return await self._vectors.drop_all(conn)

    async def rebuild_vector_index(self, *, batch_size: int = 64) -> int:
        """Drop the vector index and re-embed everything with the current embedder.

        Returns the number of vectors stored. If interrupted, finish with
        `reembed_missing`.
        """
        await self.drop_vector_index()
        return await self.reembed_missing(batch_size=batch_size)

    async def rebuild_fts(self) -> None:
        """Rebuild both full-text indexes from the base tables (they are kept in sync by
        triggers; this is for repair)."""
        async with self._write() as conn:
            await conn.execute("INSERT INTO statements_fts(statements_fts) VALUES ('rebuild')")
            await conn.execute("INSERT INTO entities_fts(entities_fts) VALUES ('rebuild')")

    # --- Forget and reset -----------------------------------------------------

    async def forget_statements(
        self, statement_ids: Iterable[int], *, dry_run: bool = False
    ) -> ForgetResult:
        """Forget statements and everything they were learned from ("forget that").

        In one transaction (a crash leaves everything as it was):

        1. The *turns* of the requested statements' source episodes are found,
           and every episode of those turns is deleted (all event types: the
           assistant's reply and tool results in the same turn usually repeat
           the information). An episode without a turn id is deleted alone.
        2. The requested statements are deleted, and so is every other
           statement left with no source episode. Statements that still have a
           source outside the deleted episodes survive (losing the deleted
           links). Statements that never had sources are unaffected.
        3. Supersession stays consistent: a surviving statement that was
           superseded by a deleted one now points at the first surviving
           successor down the chain, keeping its `invalid_at`; if there is
           none, it is reopened (`invalid_at` and `superseded_by` cleared), as if
           the forgotten correction had never been made, which is also what a
           rebuild from the remaining log would produce.
        4. The traces of the affected turns are deleted, and so is every other
           trace containing a deleted statement's text or a deleted episode's
           text (at least `MIN_TRACE_NEEDLE` characters): traces copy memory and
           conversation into prompts.
        5. Entities that were subject or object of a deleted statement and are
           now referenced by no statement are deleted (with entities merged into
           them); surviving ones have their summary cleared, since it may have
           been generated from forgotten statements. The user entity is kept.
        6. Profile versions created at or after the earliest deleted statement's
           `recorded_at` are deleted; the profile should be regenerated.
        7. Vectors and full-text rows of everything deleted are removed.

        Unknown ids are ignored. With `dry_run`, nothing is changed and the
        result says what would be deleted (for a confirmation prompt).
        """
        requested = _ids(statement_ids)
        result = ForgetResult()
        async with self._write(dry_run=dry_run) as conn:
            req = [
                r[0]
                for r in await _all(
                    conn,
                    "SELECT id FROM statements WHERE id IN (SELECT value FROM json_each(?))",
                    (requested,),
                )
            ]
            if not req:
                return result
            source_eps = await _all(
                conn,
                "SELECT DISTINCT episode_id FROM statement_sources"
                " WHERE statement_id IN (SELECT value FROM json_each(?))",
                (_ids(req),),
            )
            turns = [
                r[0]
                for r in await _all(
                    conn,
                    "SELECT DISTINCT turn_id FROM episodes WHERE turn_id IS NOT NULL"
                    " AND id IN (SELECT value FROM json_each(?)) ORDER BY turn_id",
                    (_ids(r[0] for r in source_eps),),
                )
            ]
            episode_rows = await _all(
                conn,
                "SELECT id, payload FROM episodes WHERE id IN (SELECT value FROM json_each(?))"
                " OR turn_id IN (SELECT value FROM json_each(?)) ORDER BY id",
                (_ids(r[0] for r in source_eps), json.dumps(turns)),
            )
            episodes = [r[0] for r in episode_rows]
            orphans = [
                r[0]
                for r in await _all(
                    conn,
                    "SELECT DISTINCT ss.statement_id FROM statement_sources ss"
                    " WHERE ss.episode_id IN (SELECT value FROM json_each(?1))"
                    " AND NOT EXISTS (SELECT 1 FROM statement_sources o"
                    "   WHERE o.statement_id = ss.statement_id"
                    "   AND o.episode_id NOT IN (SELECT value FROM json_each(?1)))",
                    (_ids(episodes),),
                )
            ]
            deleted = sorted(set(req) | set(orphans))
            deleted_json = _ids(deleted)
            info = await _all(
                conn,
                "SELECT id, subject_id, object_id, text, recorded_at, superseded_by"
                " FROM statements WHERE id IN (SELECT value FROM json_each(?))",
                (deleted_json,),
            )
            successor = {r[0]: r[5] for r in info}
            gone = set(deleted)

            # 3. Supersession consistency.
            reopened, repointed = [], []
            for sid, sup in await _all(
                conn,
                "SELECT id, superseded_by FROM statements WHERE superseded_by IN"
                " (SELECT value FROM json_each(?1)) AND id NOT IN (SELECT value FROM json_each(?1))"
                " ORDER BY id",
                (deleted_json,),
            ):
                nxt, hops = sup, 0
                while nxt is not None and nxt in gone and hops <= len(gone):
                    nxt, hops = successor.get(nxt), hops + 1
                if nxt is not None and nxt not in gone:
                    await conn.execute(
                        "UPDATE statements SET superseded_by = ? WHERE id = ?", (nxt, sid)
                    )
                    repointed.append(sid)
                else:
                    await conn.execute(
                        "UPDATE statements SET superseded_by = NULL, invalid_at = NULL"
                        " WHERE id = ?",
                        (sid,),
                    )
                    reopened.append(sid)
            await self._vectors.set_invalid(conn, reopened, None)

            # 2 and 7. Statements, their sources, vectors (FTS rows go by trigger).
            await conn.execute(
                "DELETE FROM statement_sources"
                " WHERE statement_id IN (SELECT value FROM json_each(?))"
                " OR episode_id IN (SELECT value FROM json_each(?))",
                (deleted_json, _ids(episodes)),
            )
            await conn.execute(
                "DELETE FROM statements WHERE id IN (SELECT value FROM json_each(?))",
                (deleted_json,),
            )
            await self._vectors.delete_statements(conn, deleted)

            # 1. Episodes.
            await conn.execute(
                "DELETE FROM episodes WHERE id IN (SELECT value FROM json_each(?))",
                (_ids(episodes),),
            )

            # 4. Traces.
            needles = {r[3] for r in info} | {_episode_text(r[1]) for r in episode_rows}
            traces = set()
            for row in await _all(
                conn,
                "DELETE FROM turn_traces WHERE turn_id IN (SELECT value FROM json_each(?))"
                " RETURNING turn_id",
                (json.dumps(turns),),
            ):
                traces.add(row[0])
            for needle in sorted(n for n in needles if len(n.strip()) >= MIN_TRACE_NEEDLE):
                for row in await _all(
                    conn,
                    "DELETE FROM turn_traces WHERE instr(payload, ?) > 0 RETURNING turn_id",
                    (_json_needle(needle),),
                ):
                    traces.add(row[0])

            # 5. Entities.
            touched = sorted(
                ({r[1] for r in info} | {r[2] for r in info if r[2] is not None}) - {USER_ENTITY_ID}
            )
            removed_entities: list[int] = []
            cleared: list[int] = []
            for eid in touched:
                used = await _one(
                    conn,
                    "SELECT 1 FROM statements WHERE subject_id = ?1 OR object_id = ?1 LIMIT 1",
                    (eid,),
                )
                if used is None:
                    group = [
                        r[0]
                        for r in await _all(
                            conn,
                            "WITH RECURSIVE g(id) AS (SELECT ?1 UNION SELECT e.id FROM entities e"
                            " JOIN g ON e.merged_into = g.id) SELECT id FROM g",
                            (eid,),
                        )
                    ]
                    await conn.execute(
                        "DELETE FROM entities WHERE id IN (SELECT value FROM json_each(?))",
                        (_ids(group),),
                    )
                    await self._vectors.delete_entities(conn, group)
                    removed_entities.extend(group)
                else:
                    async with conn.execute(
                        "UPDATE entities SET summary = NULL, updated_at = ?"
                        " WHERE id = ? AND summary IS NOT NULL",
                        (_ts(self._now()), eid),
                    ) as cursor:
                        if cursor.rowcount:
                            cleared.append(eid)
                            await self._vectors.delete_entities(conn, [eid])

            # 6. Profile versions possibly generated from the forgotten statements.
            earliest = min(r[4] for r in info)
            async with conn.execute(
                "DELETE FROM profile_versions WHERE created_at >= ?", (earliest,)
            ) as cursor:
                profiles = cursor.rowcount

            result = ForgetResult(
                statement_ids=deleted,
                episode_ids=episodes,
                trace_turn_ids=sorted(traces),
                entity_ids=sorted(removed_entities),
                cleared_summary_entity_ids=cleared,
                reopened_statement_ids=reopened,
                repointed_statement_ids=repointed,
                profile_versions_deleted=profiles,
            )
        return result

    async def reset_graph(self) -> ResetResult:
        """Clear everything derived from the episode log, so it can be re-extracted.

        In one transaction: deletes all statements and their sources, all
        entities except the user entity (which keeps its id and is reset to its
        initial state: type "person", name "User", no aliases or summary), all
        profile versions, the derived metadata (`memory_meta` keys starting with
        `DERIVED_META_PREFIX`) and the whole vector index; rebuilds the full-text
        indexes; and sets `extracted_at` to NULL on every episode. Episodes,
        traces and jobs are untouched.
        """
        now = _ts(self._now())
        async with self._write() as conn:
            statements = (await _one(conn, "SELECT count(*) FROM statements"))[0]
            await conn.execute("DELETE FROM statement_sources")
            await conn.execute("DELETE FROM statements")
            async with conn.execute("DELETE FROM entities WHERE id <> ?", (USER_ENTITY_ID,)) as c:
                entities = c.rowcount
            await conn.execute(
                "INSERT INTO entities (id, type, name, aliases, created_at, updated_at)"
                " VALUES (?, 'person', ?, '[]', ?, ?)"
                " ON CONFLICT (id) DO UPDATE SET type = 'person', name = excluded.name,"
                " aliases = '[]', summary = NULL, merged_into = NULL,"
                " updated_at = excluded.updated_at",
                (USER_ENTITY_ID, DEFAULT_USER_NAME, now, now),
            )
            async with conn.execute("DELETE FROM profile_versions") as c:
                profiles = c.rowcount
            low, high = _prefix_range(DERIVED_META_PREFIX) or ("", "")
            await conn.execute("DELETE FROM memory_meta WHERE key >= ? AND key < ?", (low, high))
            await self._vectors.drop_all(conn)
            async with conn.execute(
                "UPDATE episodes SET extracted_at = NULL WHERE extracted_at IS NOT NULL"
            ) as c:
                episodes = c.rowcount
            await conn.execute("INSERT INTO statements_fts(statements_fts) VALUES ('rebuild')")
            await conn.execute("INSERT INTO entities_fts(entities_fts) VALUES ('rebuild')")
        return ResetResult(
            statements_deleted=statements,
            entities_deleted=entities,
            profile_versions_deleted=profiles,
            episodes_reset=episodes,
        )
