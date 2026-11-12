"""The `GraphStore` interface: what the learning loop, retrieval and the tools need
from a knowledge-graph store (docs/memory.md, ADR 0010).

`lucy.memory.graph_store.SQLiteGraphStore` is the only implementation today. The
protocol exists so another store (e.g. Postgres) can be swapped in later: it
lists every method the rest of Lucy calls, and the docstrings state the
semantics a second implementation must honour. `tests/test_graph_ports.py`
checks that `SQLiteGraphStore` matches it signature for signature.

Shared rules (see `graph_store` for the SQLite details):

- Every write is atomic: on any error (including cancellation) nothing of it
  is stored. Embedding happens before writing; embedder errors propagate and
  nothing is written. Documents are embedded with `kind="document"`.
- Entity ids passed in may refer to merged entities; they resolve to the
  surviving entity wherever an entity is looked up or referenced. The user is
  entity `USER_ENTITY_ID`, never deleted or merged away.
- A statement is *current* at time t if its `invalid_at` is unset or after t;
  "now" comes from the store's clock.
- Missing ids: getters return None or skip them; mutations raise
  `GraphNotFoundError`. Invalid references or dates raise `GraphStoreError` and
  change nothing. Both errors, and `DERIVED_META_PREFIX`, are defined here so
  code written against the interface never imports an implementation.
- Vectors belong to one embedder (model and dimensions). Items without a
  vector for the current embedder are invisible to vector search until
  `reembed_missing` runs.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Final, Literal, Protocol, runtime_checkable

from lucy.core.ports import Embedder
from lucy.memory.models import (
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

#: `memory_meta` keys starting with this describe the graph's current contents
#: (decay state, profile input hash, summary fingerprints, rejected merges); they
#: are cleared by `reset_graph`. Other keys (e.g. when maintenance last ran) are kept.
#: Part of the interface because callers choose their keys by it and every store's
#: `reset_graph` must honour it.
DERIVED_META_PREFIX: Final = "derived."


class MemoryStoreError(RuntimeError):
    """The memory store cannot do what was asked (duplicate event, bad schema, ...)."""


class GraphStoreError(MemoryStoreError):
    """A graph operation was invalid (bad reference, inconsistent dates, conflict...)."""


class GraphNotFoundError(GraphStoreError, LookupError):
    """A statement, entity or episode that a mutation refers to does not exist."""


@runtime_checkable
class GraphStore(Protocol):
    """Entities, statements, provenance, profile versions and maintenance metadata."""

    @property
    def embedder(self) -> Embedder:
        """The embedder whose vector space the store searches."""
        ...

    async def close(self) -> None:
        """Release connections. The store may reconnect if used again."""
        ...

    # --- Entities -------------------------------------------------------------

    async def user_entity(self) -> Entity:
        """The entity representing the user (id `USER_ENTITY_ID`, type "person")."""
        ...

    async def create_entity(
        self,
        type: str,
        name: str,
        *,
        aliases: Iterable[str] = (),
        summary: str | None = None,
    ) -> Entity:
        """Create and embed an entity. Blank name or type: `ValueError`. Aliases are
        cleaned (stripped, de-duplicated case-insensitively, never the name). Does
        not look for an existing entity of that name."""
        ...

    async def get_entity(self, entity_id: int, *, resolve: bool = False) -> Entity | None:
        """The entity, or None. With `resolve`, the entity it was merged into."""
        ...

    async def get_entities(self, entity_ids: Iterable[int]) -> list[Entity]:
        """The existing entities among `entity_ids` (not resolved), ascending by id."""
        ...

    async def list_entities(
        self,
        *,
        type: str | None = None,
        include_merged: bool = False,
        after_id: int | None = None,
        limit: int | None = None,
    ) -> list[Entity]:
        """Entities ascending by id (surviving only unless `include_merged`); page with
        `after_id`."""
        ...

    async def find_entities(self, name: str) -> list[Entity]:
        """Surviving entities whose name or an alias equals `name`, ignoring case and
        repeated whitespace (not punctuation or accents), ascending by id. Matches
        on merged entities resolve to the survivor."""
        ...

    async def entities_mentioned_in(self, text: str, *, candidates: int = 200) -> list[Entity]:
        """Surviving entities whose name or an alias occurs in `text` as whole words
        (case-insensitive), longest matched name first."""
        ...

    async def similar_entities(
        self,
        text: str,
        k: int = 10,
        *,
        kind: Literal["document", "query"] = "document",
    ) -> list[ScoredEntity]:
        """The `k` surviving entities whose name+summary vectors are nearest to `text`
        (embedded with `kind`), best first; scores are cosine similarities."""
        ...

    async def similar_entities_by_vector(
        self, vector: Sequence[float], k: int = 10
    ) -> list[ScoredEntity]:
        """Like `similar_entities` for an already embedded vector."""
        ...

    async def add_alias(self, entity_id: int, alias: str) -> Entity:
        """Add `alias` to the surviving entity; a no-op if it already is a name."""
        ...

    async def rename_entity(
        self, entity_id: int, name: str, *, keep_old_as_alias: bool = True
    ) -> Entity:
        """New canonical name (re-embedded); the old one becomes an alias unless told not."""
        ...

    async def update_entity_summary(self, entity_id: int, summary: str | None) -> Entity:
        """Replace the summary (blank means none) and re-embed name plus summary."""
        ...

    async def merge_entities(self, source_id: int, target_id: int) -> Entity:
        """Merge source into target atomically: statements repointed, names become
        aliases, earlier merges into the source follow. The user entity cannot be
        merged away (`GraphStoreError`). Returns the survivor."""
        ...

    # --- Statements -----------------------------------------------------------

    async def add_statement(self, new: NewStatement, episode_ids: Iterable[int] = ()) -> Statement:
        """Add and embed a statement learned from `episode_ids`. `recorded_at` is now,
        `last_observed_at` is `new.observed_at` or now, `evidence_count` is 1."""
        ...

    async def get_statement(self, statement_id: int) -> Statement | None:
        """The statement with its sources, or None."""
        ...

    async def get_statements(self, statement_ids: Sequence[int]) -> list[Statement]:
        """The existing statements among `statement_ids`, in the given order."""
        ...

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
        """Statements matching every given filter, ascending by id. `observed_before`
        is exclusive on `last_observed_at`; `min_confidence` inclusive,
        `confidence_below` exclusive."""
        ...

    async def current_statements(
        self, subject_id: int, *, kinds: Collection[StatementKind | str] | None = None
    ) -> list[Statement]:
        """The current statements with `subject_id` (resolved) as subject."""
        ...

    async def statements_superseded_by(self, statement_id: int) -> list[Statement]:
        """The statements `statement_id` replaced (their `superseded_by` is it), by id."""
        ...

    async def active_instructions(self) -> list[Statement]:
        """All current instruction statements, oldest first."""
        ...

    async def neighbours(
        self,
        entity_id: int,
        *,
        kinds: Collection[StatementKind | str] | None = None,
        limit: int | None = None,
    ) -> list[Statement]:
        """Current statements with the entity as subject or object (one hop), by
        confidence, then most recently observed, then id."""
        ...

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
        """Up to `k` statements nearest to `text` by vector (cosine), best first, with
        the filters applied inside the search; current only unless `include_invalid`."""
        ...

    async def similar_statements_by_vector(
        self,
        vector: Sequence[float],
        *,
        subject_id: int | None = None,
        k: int = 10,
        include_invalid: bool = False,
        kinds: Collection[StatementKind | str] | None = None,
    ) -> list[ScoredStatement]:
        """Like `similar_statements` for an already embedded vector."""
        ...

    async def keyword_statements(
        self,
        text: str,
        *,
        k: int = 20,
        include_invalid: bool = False,
        kinds: Collection[StatementKind | str] | None = None,
    ) -> list[ScoredStatement]:
        """Full-text search for any word of `text` (arbitrary input is safe), best
        first; higher scores are better."""
        ...

    async def reinforce(
        self,
        statement_id: int,
        episode_ids: Iterable[int] = (),
        *,
        confidence: float | None = None,
        observed_at: datetime | None = None,
    ) -> Statement:
        """Another observation: link sources, set `confidence` if given, move
        `last_observed_at` forward, count evidence once. Idempotent when every
        given episode was already a source."""
        ...

    async def supersede(
        self, old_id: int, new: NewStatement, episode_ids: Iterable[int] = ()
    ) -> Statement:
        """Add `new` and close `old_id` atomically: `superseded_by` = the new id and
        `invalid_at` = `new.valid_from` or now (or an earlier end it had). Raises
        `GraphStoreError` if it is already superseded or the end precedes its start."""
        ...

    async def invalidate(self, statement_id: int, at: datetime | None = None) -> Statement:
        """End a statement at `at` (default now) without a successor: `invalid_at` is
        set (overwriting an earlier one), `superseded_by` is untouched. An end
        before its `valid_from` raises `GraphStoreError` and changes nothing. For a
        goal that was achieved or dropped, which nothing replaces."""
        ...

    async def end_statement(
        self, statement_id: int, at: datetime, *, superseded_by: int | None = None
    ) -> Statement:
        """End a statement at `at` because of another, existing statement (e.g. one new
        statement ending several old ones). Unlike `invalidate`, an earlier end is
        kept (the earlier of the two wins); `superseded_by`, if given, records the
        statement that ended it. Missing statements: `GraphNotFoundError`; an end
        before its `valid_from`, an existing different successor, or pointing at
        itself: `GraphStoreError`, nothing changed."""
        ...

    async def set_statement_texts(self, texts: Mapping[int, str]) -> int:
        """Reword statements (same meaning, new text, e.g. "The user" replaced by the
        user's name) and re-embed them; nothing else changes. Unknown ids are
        skipped; blank texts raise `ValueError` and nothing is written. Returns how
        many changed."""
        ...

    async def statements_recorded_since(
        self,
        since: datetime,
        *,
        subject_id: int | None = None,
        kinds: Collection[StatementKind | str] | None = None,
        limit: int | None = None,
    ) -> list[Statement]:
        """Statements whose `recorded_at` (when Lucy learned them, not when they became
        true) is at or after `since`, current or not, newest first; optionally only
        about the (resolved) `subject_id`, of `kinds`, at most `limit`."""
        ...

    async def set_confidences_and_meta(
        self, confidences: Mapping[int, float], meta: Mapping[str, str | None]
    ) -> int:
        """Set confidences and write metadata in one transaction; returns how many of
        the statements existed. Values outside [0, 1]: `ValueError`, nothing written."""
        ...

    # --- Episodes ---------------------------------------------------------------

    async def unextracted_episodes(self, limit: int = 50) -> list[EpisodeRecord]:
        """Up to `limit` content episodes (user, assistant, tool result) not yet
        extracted, oldest first."""
        ...

    async def mark_extracted(self, episode_ids: Iterable[int], at: datetime | None = None) -> int:
        """Set `extracted_at` (default now); returns how many episodes exist."""
        ...

    async def get_episodes(self, episode_ids: Iterable[int]) -> list[EpisodeRecord]:
        """The existing episodes among `episode_ids`, oldest first."""
        ...

    async def turn_episodes(
        self, turn_id: str, *, types: Sequence[str] | None = ...
    ) -> list[EpisodeRecord]:
        """The episodes of `turn_id` of `types` (default: content types; None: all)."""
        ...

    async def episodes_before(
        self, episode_id: int, n: int, *, types: Sequence[str] | None = ...
    ) -> list[EpisodeRecord]:
        """Up to `n` episodes of `types` preceding `episode_id` in its session, oldest
        first. Unknown `episode_id`: `GraphNotFoundError`."""
        ...

    async def episodes_between(
        self,
        start: datetime,
        end: datetime,
        *,
        types: Sequence[str] | None = ...,
        after_id: int | None = None,
        limit: int | None = None,
    ) -> list[EpisodeRecord]:
        """Episodes of `types` (default: user messages; None: all) whose *event* time
        (`event.ts`, not the time they were logged) lies in `[start, end)`, in event
        time order (ties by id); with `after_id`, only ids above it; `limit` keeps the
        earliest. Naive datetimes: `ValueError`. Reads every match, so bound the period."""
        ...

    async def count_episodes_after(
        self, after_id: int, *, types: Sequence[str] | None = ...
    ) -> int:
        """How many episodes of `types` (default: user messages; None: all) have an id
        above `after_id`. Ids grow in logging order, so this counts what was logged
        since an episode was seen."""
        ...

    async def episodes_mentioning(self, text: str, *, limit: int = 20) -> list[EpisodeRecord]:
        """Up to `limit` content episodes whose text contains `text` (case-insensitive),
        oldest first. Blank `text` finds nothing."""
        ...

    # --- Profile ----------------------------------------------------------------

    async def save_profile(self, content: str) -> ProfileVersion:
        """Store a new profile version; it becomes current. Old versions are kept."""
        ...

    async def current_profile(self) -> ProfileVersion | None:
        """The most recently saved profile version, or None."""
        ...

    async def profile_versions(self, limit: int | None = None) -> list[ProfileVersion]:
        """Profile versions, newest first."""
        ...

    # --- Metadata -----------------------------------------------------------------

    async def get_meta(self, key: str) -> str | None:
        """The value stored under `key`, or None."""
        ...

    async def get_meta_many(self, keys: Iterable[str]) -> dict[str, str]:
        """The stored values of those of `keys` that exist."""
        ...

    async def set_meta(self, key: str, value: str) -> None:
        """Store `value` under `key`, replacing any previous value."""
        ...

    async def delete_meta(self, key: str) -> bool:
        """Delete `key`; returns whether it existed."""
        ...

    async def list_meta(
        self, prefix: str = "", *, after: str | None = None, limit: int | None = None
    ) -> list[tuple[str, str]]:
        """`(key, value)` pairs whose key starts with `prefix`, ascending by key."""
        ...

    async def write_meta(self, values: Mapping[str, str | None]) -> None:
        """Set (None: delete) several keys in one transaction."""
        ...

    async def entity_statement_fingerprints(
        self, entity_ids: Iterable[int]
    ) -> dict[int, tuple[int, str]]:
        """Per existing entity: the number of its current statements (as subject or
        object) and a fingerprint that changes whenever that set changes (but not
        on confidence or observation-time changes)."""
        ...

    # --- Overview, vectors, forget and reset -----------------------------------------

    async def stats(self) -> GraphStats:
        """Counts of episodes, statements, entities and profile versions, from one
        consistent snapshot."""
        ...

    async def vector_status(self) -> VectorStatus:
        """How many statements and surviving entities lack a current-embedder vector."""
        ...

    async def reembed_missing(self, limit: int | None = None, *, batch_size: int = 64) -> int:
        """Embed up to `limit` items lacking a current-embedder vector; safe to
        interrupt and repeat. Returns the number of vectors stored."""
        ...

    async def forget_statements(
        self, statement_ids: Iterable[int], *, dry_run: bool = False
    ) -> ForgetResult:
        """Delete statements and every episode of the turns they came from, with the
        consequences listed in `ForgetResult` (orphaned statements, supersession
        repair, traces, unreferenced entities, recent profile versions), in one
        transaction. With `dry_run`, change nothing and report what would go."""
        ...

    async def reset_graph(self) -> ResetResult:
        """Clear everything derived from the episode log (statements, entities but the
        user, profile versions, derived metadata, vectors) and mark every episode
        unextracted. Episodes, traces and jobs are kept."""
        ...
