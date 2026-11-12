"""Tunable numbers of the learning loop's extract and reconcile steps (docs/memory.md).

Every value here is a heuristic to be tuned against the memory eval set. The
similarity thresholds are cosine similarities and depend on the embedding
model: the defaults were measured with `embeddinggemma-2` (768 dimensions,
"passage: " document prefix), whose scores are compressed into a narrow, high
band. Measured on pairs of statements about one person (hand-written, and
produced by the extraction models in the live evaluation of 2026-11-12):

- same meaning, reworded: 0.83 to 0.98 (0.89 when one says "The user" and
  the other the user's name)
- contradictions ("lives in Bristol" / "lives in Manchester"): 0.85 to 0.96
- related, different facts about the same person: 0.79 to 0.96 (two options
  of one decision: 0.95)
- unrelated facts about the same person: 0.66 to 0.83

and on entity names (a bare mention against a stored name):

- the same entity ("Meera" / "Meera Rao", "NYC" / "New York"): 0.82 to 1.0
- different entities ("Priya" / "Priyanka", "Leeds" / "Leeds United"): 0.79 to 0.93

A different embedder needs these measured again.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lucy.memory.models import Origin


class MemoryPolicy(BaseModel):
    """Confidence rules and similarity thresholds for extraction and reconciliation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stated_confidence: float = Field(0.9, ge=0.0, le=1.0)
    """Starting confidence of a statement the user said outright."""
    inferred_confidence: float = Field(0.4, ge=0.0, le=1.0)
    """Starting confidence of a statement Lucy concluded."""
    reinforcement_step: float = Field(0.25, gt=0.0, le=1.0)
    """Each reinforcement moves confidence this fraction of the way to 1.0."""
    stated_floor: float = Field(0.9, ge=0.0, le=1.0)
    """A stated statement's confidence never drops below this on reinforcement."""

    statement_similarity_floor: float = Field(0.84, ge=-1.0, le=1.0)
    """Below this, an existing statement is not similar enough to a candidate to
    be reinforced or contradicted by it: if nothing reaches it, the candidate is
    added without asking the model. Just above the unrelated band (up to 0.83)
    and below the weakest contradiction seen (0.85): a missed contradiction
    leaves two conflicting current beliefs, while a needless model call only
    costs time."""
    near_identical_similarity: float = Field(0.97, ge=-1.0, le=1.0)
    """At or above this (with the same object), two statement texts say the same
    thing: a candidate near-identical to a statement that already has the same
    source episodes was stored by an earlier run of the same turn, and the
    model's "add" of a near-identical statement is turned into a reinforcement.
    Above different-but-similar pairs seen (0.96: "considered the Kobo" /
    "chose the Kobo")."""

    entity_match_similarity: float = Field(0.95, ge=-1.0, le=1.0)
    """At or above this (after any type penalty), a mention is taken to be the
    existing entity without asking. Above the most similar different-entity
    pair measured (0.93)."""
    entity_ambiguous_similarity: float = Field(0.85, ge=-1.0, le=1.0)
    """Between this and `entity_match_similarity`, the model is asked whether the
    mention is the existing entity; below it, a new entity is created."""
    entity_type_mismatch_penalty: float = Field(0.1, ge=0.0, le=2.0)
    """Subtracted from the similarity when the mention's type differs from the
    entity's (a person is not a place)."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.entity_ambiguous_similarity > self.entity_match_similarity:
            raise ValueError("entity_ambiguous_similarity must not exceed entity_match_similarity")
        if self.statement_similarity_floor > self.near_identical_similarity:
            raise ValueError("statement_similarity_floor must not exceed near_identical_similarity")
        return self

    def initial_confidence(self, origin: Origin | str) -> float:
        """The confidence a new statement of `origin` starts with."""
        if Origin(origin) is Origin.STATED:
            return self.stated_confidence
        return self.inferred_confidence

    def reinforced(self, confidence: float, origin: Origin | str) -> float:
        """The confidence after one more supporting observation of a statement of `origin`."""
        value = confidence + (1.0 - confidence) * self.reinforcement_step
        if Origin(origin) is Origin.STATED:
            value = max(value, self.stated_floor)
        return min(max(value, 0.0), 1.0)
