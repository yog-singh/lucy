"""The knowledge graph's data model (docs/memory.md, ADR 0011).

Entities are the things statements are about; statements are typed, temporal
edges from a subject entity to an object entity or a literal value, each with a
natural-language `text` that is the retrieval surface, and links to the
episodes they were learned from.

All datetimes are timezone-aware and normalised to UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from lucy.core.events import AnyEvent

#: The id of the entity representing the user. It is created by migration 4,
#: is never deleted, merged away or renumbered, and survives `reset_graph`.
USER_ENTITY_ID: Final = 1
#: The user entity's name until the real name is learned (`rename_entity`).
DEFAULT_USER_NAME: Final = "User"


class StatementKind(StrEnum):
    """What sort of belief a statement is (docs/memory.md "Kinds")."""

    FACT = "fact"
    PREFERENCE = "preference"
    HABIT = "habit"
    DECISION = "decision"
    GOAL = "goal"
    INSTRUCTION = "instruction"


class Origin(StrEnum):
    """Whether the user said it or Lucy concluded it."""

    STATED = "stated"
    INFERRED = "inferred"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _utc(value: datetime | None) -> datetime | None:
    return None if value is None else value.astimezone(UTC)


class Entity(_Model):
    """Anything statements can be about: a person, place, project, decision, topic...

    `type` is free text (e.g. "person", "place", "decision"). `aliases` never
    contain `name` (compared case-insensitively). `merged_into` is set when the
    entity was merged into another; such an entity has no statements of its own
    and resolves to the surviving entity in every lookup.
    """

    id: int
    type: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    summary: str | None = None
    merged_into: int | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @property
    def names(self) -> list[str]:
        """The canonical name followed by the aliases."""
        return [self.name, *self.aliases]


class NewStatement(_Model):
    """Input for creating a statement (`SQLiteGraphStore.add_statement`).

    At most one of `object_id` (an entity) and `object_value` (a literal) may be
    set; both may be absent when the predicate says it all ("is vegetarian").
    `observed_at` is when the evidence was observed (usually the source
    episode's time) and becomes `last_observed_at`; it defaults to now.
    `invalid_at`, if known at creation ("lived in Pune until 2020"), must not be
    before `valid_from`.
    """

    subject_id: int
    predicate: str = Field(min_length=1)
    object_id: int | None = None
    object_value: str | None = None
    text: str = Field(min_length=1)
    kind: StatementKind
    origin: Origin
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: AwareDatetime | None = None
    invalid_at: AwareDatetime | None = None
    observed_at: AwareDatetime | None = None

    @field_validator("valid_from", "invalid_at", "observed_at")
    @classmethod
    def _to_utc(cls, value: datetime | None) -> datetime | None:
        return _utc(value)

    @field_validator("text", "predicate")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.object_id is not None and self.object_value is not None:
            raise ValueError("set at most one of object_id and object_value")
        if (
            self.valid_from is not None
            and self.invalid_at is not None
            and self.invalid_at < self.valid_from
        ):
            raise ValueError("invalid_at is before valid_from")
        return self


class Statement(_Model):
    """One thing Lucy believes, with provenance (ADR 0011).

    `valid_from`/`invalid_at` are when it was true in the world (either may be
    unknown); `recorded_at` is when Lucy learned it and `last_observed_at` when
    evidence for it was last seen. `superseded_by` points to the statement that
    replaced it (and implies `invalid_at`). `sources` are the ids of the
    episodes it was learned from, ascending.
    """

    id: int
    subject_id: int
    predicate: str
    object_id: int | None = None
    object_value: str | None = None
    text: str
    kind: StatementKind
    origin: Origin
    confidence: float
    evidence_count: int
    valid_from: AwareDatetime | None = None
    invalid_at: AwareDatetime | None = None
    superseded_by: int | None = None
    recorded_at: AwareDatetime
    last_observed_at: AwareDatetime
    sources: list[int] = Field(default_factory=list)

    def is_current(self, at: datetime) -> bool:
        """True unless the statement stopped being true at or before `at`."""
        return self.invalid_at is None or self.invalid_at > at


class ScoredStatement(_Model):
    """A search hit. `score` is cosine similarity (vector search) or -bm25 (keyword)."""

    statement: Statement
    score: float


class ScoredEntity(_Model):
    """A search hit; `score` is cosine similarity in [-1, 1]."""

    entity: Entity
    score: float


class ProfileVersion(_Model):
    """One generated version of the core profile."""

    id: int
    created_at: AwareDatetime
    content: str


class EpisodeRecord(_Model):
    """A persisted event with its row id (the id statement sources refer to)."""

    id: int
    event: AnyEvent
    extracted_at: AwareDatetime | None = None


class ForgetResult(_Model):
    """What `forget_statements` deleted or changed (all id lists ascending).

    - `statement_ids`: every deleted statement: the requested ones plus those
      left without any source episode.
    - `episode_ids`: deleted episodes (every episode of each affected turn).
    - `trace_turn_ids`: turns whose stored trace was deleted.
    - `entity_ids`: entities deleted because nothing referred to them any more.
    - `cleared_summary_entity_ids`: surviving entities whose summary was cleared
      because it may have been generated from forgotten statements.
    - `reopened_statement_ids`: statements that had been superseded by a deleted
      statement with no surviving successor; they are current again.
    - `repointed_statement_ids`: statements superseded by a deleted statement
      whose `superseded_by` now points at the deleted one's surviving successor.
    - `profile_versions_deleted`: profile versions created since the earliest
      deleted statement was recorded; the profile should be regenerated.
    """

    statement_ids: list[int] = Field(default_factory=list)
    episode_ids: list[int] = Field(default_factory=list)
    trace_turn_ids: list[str] = Field(default_factory=list)
    entity_ids: list[int] = Field(default_factory=list)
    cleared_summary_entity_ids: list[int] = Field(default_factory=list)
    reopened_statement_ids: list[int] = Field(default_factory=list)
    repointed_statement_ids: list[int] = Field(default_factory=list)
    profile_versions_deleted: int = 0


class VectorStatus(_Model):
    """How many items lack a vector for the current embedder's model and dimensions."""

    model: str
    dimensions: int
    statements_missing: int
    entities_missing: int


class ResetResult(_Model):
    """What `reset_graph` cleared."""

    statements_deleted: int
    entities_deleted: int
    profile_versions_deleted: int
    episodes_reset: int


class GraphStats(_Model):
    """Counts for an overview of memory (`lucy memory status`).

    - `episodes`: every persisted event; `episodes_unextracted`: content
      episodes (user, assistant, tool result) not yet processed by extraction.
    - `statements_current` / `statements_invalidated`: statements current now,
      and those whose `invalid_at` has passed; `current_by_kind`: the current
      ones per kind (kinds with none are absent).
    - `entities`: surviving entities (the user included); `entities_merged`:
      entities merged into another.
    """

    episodes: int
    episodes_unextracted: int
    statements_current: int
    statements_invalidated: int
    current_by_kind: dict[StatementKind, int] = Field(default_factory=dict)
    entities: int
    entities_merged: int
    profile_versions: int
