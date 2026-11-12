"""Entity resolution for extraction (docs/memory.md "1. Extract").

A mention ("Meera", a person) is resolved to an entity in three steps:

1. exact name or alias match (`find_entities`), preferring an entity of a
   compatible type;
2. embedding similarity (`similar_entities`), with a penalty when the types
   differ: at or above `MemoryPolicy.entity_match_similarity` it is a match;
3. between `entity_ambiguous_similarity` and that, the model is asked a yes/no
   question (prompt `entity_match`).

Otherwise a new entity is created. Similarity alone below the upper threshold
never matches. References to the user ("user", "I", "me", the user's name)
resolve to the user entity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import BaseModel

from lucy.core.gateway import LLMGateway, Priority
from lucy.core.types import Message, Role
from lucy.memory import fts
from lucy.memory.models import USER_ENTITY_ID, Entity
from lucy.memory.policy import MemoryPolicy
from lucy.memory.ports import GraphStore
from lucy.prompts import load_prompt

log = logging.getLogger(__name__)

#: Mentions that always mean the user (compared after `fts.normalise`).
USER_REFERENCES: Final = frozenset(
    {"user", "the user", "i", "me", "my", "myself", "mine", "the speaker"}
)

#: Entity types treated as the same type when resolving.
_TYPE_SYNONYMS: Final = {
    "people": "person",
    "human": "person",
    "pet": "animal",
    "dog": "animal",
    "cat": "animal",
    "location": "place",
    "city": "place",
    "town": "place",
    "country": "place",
    "organization": "organisation",
    "company": "organisation",
    "employer": "organisation",
    "org": "organisation",
    "business": "organisation",
    "school": "organisation",
    "thing": "object",
    "item": "object",
    "product": "object",
    "device": "object",
}
#: Types that say nothing, and so are compatible with any type.
_GENERIC_TYPES: Final = frozenset({"", "other", "unknown", "entity"})

#: Ambiguous candidates the model is asked about per mention, best first.
MAX_LLM_CHECKS: Final = 2
#: Similar entities fetched per mention.
SIMILAR_K: Final = 5


def normalise_type(type: str) -> str:
    """A canonical spelling of an entity type for comparison."""
    key = " ".join(type.lower().split())
    return _TYPE_SYNONYMS.get(key, key)


def types_compatible(a: str, b: str) -> bool:
    """Whether entity types `a` and `b` may describe the same entity."""
    a, b = normalise_type(a), normalise_type(b)
    return a == b or a in _GENERIC_TYPES or b in _GENERIC_TYPES


class MatchMethod(StrEnum):
    """How a mention was resolved."""

    USER = "user"
    EXACT = "exact"
    ALIAS = "alias"
    SIMILAR = "similar"
    CONFIRMED = "confirmed"
    CREATED = "created"


@dataclass(frozen=True)
class Resolution:
    """The entity a mention resolved to, how, and with what similarity (if any)."""

    entity: Entity
    method: MatchMethod
    score: float | None = None


class EntityMatchAnswer(BaseModel):
    """The model's answer to "is this mention that entity?"."""

    same: bool


class EntityResolver:
    """Resolves mentions to entities, creating them when nothing matches.

    Resolutions are cached by (name, type) until `reset` (the extractor resets
    it per turn), so one turn resolves a repeated mention once.
    """

    def __init__(
        self,
        graph: GraphStore,
        gateway: LLMGateway,
        *,
        policy: MemoryPolicy | None = None,
        role: str = "extraction",
        priority: Priority = Priority.BACKGROUND,
    ) -> None:
        self._graph = graph
        self._gateway = gateway
        self.policy = policy or MemoryPolicy()
        self.role = role
        self.priority = priority
        self._prompt = load_prompt("entity_match")
        self._cache: dict[tuple[str, str], Resolution] = {}
        self._created: dict[int, Entity] = {}

    def reset(self) -> None:
        """Forget cached resolutions and the entities created so far."""
        self._cache.clear()
        self._created.clear()

    @property
    def created(self) -> list[Entity]:
        """The entities this resolver created since the last `reset`, oldest first."""
        return list(self._created.values())

    async def is_user(self, name: str) -> bool:
        """Whether `name` refers to the user."""
        key = fts.normalise(name)
        if key in USER_REFERENCES:
            return True
        user = await self._graph.user_entity()
        return key in {fts.normalise(n) for n in user.names}

    async def resolve(self, name: str, type: str, *, context: str = "") -> Resolution:
        """The entity `name` (of `type`) refers to; creates one if none matches.

        `context` (the sentence the mention came from) is shown to the model in
        the ambiguous case. Raises `ValueError` if `name` is blank.
        """
        name, type = " ".join(name.split()), " ".join(type.split()) or "other"
        if not name:
            raise ValueError("cannot resolve a blank name")
        cache_key = (fts.normalise(name), normalise_type(type))
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = await self._resolve(name, type, context)
        log.debug(
            "resolved %r (%s) to entity %d by %s", name, type, result.entity.id, result.method
        )
        self._cache[cache_key] = result
        return result

    async def _resolve(self, name: str, type: str, context: str) -> Resolution:
        key = fts.normalise(name)
        if key in USER_REFERENCES:
            return Resolution(await self._graph.user_entity(), MatchMethod.USER)

        exact = [e for e in await self._graph.find_entities(name) if types_compatible(type, e.type)]
        if exact:
            if len(exact) > 1:
                log.info("%r matches %d entities by name; using the oldest", name, len(exact))
            entity = exact[0]
            if entity.id == USER_ENTITY_ID:
                return Resolution(entity, MatchMethod.USER)
            method = MatchMethod.EXACT if fts.normalise(entity.name) == key else MatchMethod.ALIAS
            return Resolution(entity, method)

        policy = self.policy
        scored = []
        for hit in await self._graph.similar_entities(name, k=SIMILAR_K):
            score = hit.score
            if not types_compatible(type, hit.entity.type):
                score -= policy.entity_type_mismatch_penalty
            scored.append((score, hit.entity))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))

        if scored and scored[0][0] >= policy.entity_match_similarity:
            score, entity = scored[0]
            return Resolution(await self._alias(entity, name), MatchMethod.SIMILAR, score)

        ambiguous = [
            (score, entity)
            for score, entity in scored
            if score >= policy.entity_ambiguous_similarity
        ][:MAX_LLM_CHECKS]
        for score, entity in ambiguous:
            if await self._confirm(name, type, context, entity):
                return Resolution(await self._alias(entity, name), MatchMethod.CONFIRMED, score)
            log.info("model says %r (%s) is not entity %d %r", name, type, entity.id, entity.name)

        entity = await self._graph.create_entity(type, name)
        self._created[entity.id] = entity
        return Resolution(entity, MatchMethod.CREATED, scored[0][0] if scored else None)

    async def _alias(self, entity: Entity, name: str) -> Entity:
        if fts.normalise(name) in {fts.normalise(n) for n in entity.names}:
            return entity
        return await self._graph.add_alias(entity.id, name)

    async def _confirm(self, name: str, type: str, context: str, entity: Entity) -> bool:
        """Ask the model whether the mention `name` is `entity`."""
        facts = await self._graph.neighbours(entity.id, limit=5)
        lines = [
            f'Mention: "{name}" (type: {type})',
            *([f'Sentence it appears in: "{context}"'] if context else []),
            "",
            f'Known entity: "{entity.name}" (type: {entity.type})',
        ]
        if entity.aliases:
            lines.append("Also called: " + ", ".join(f'"{a}"' for a in entity.aliases))
        if entity.summary:
            lines.append(f"Summary: {entity.summary}")
        if facts:
            lines.append("Known facts:")
            lines.extend(f"- {s.text}" for s in facts)
        lines += ["", "Is the mention the known entity?"]
        messages = [
            Message(role=Role.SYSTEM, content=self._prompt),
            Message(role=Role.USER, content="\n".join(lines)),
        ]
        answer = await self._gateway.generate(
            self.role, messages, EntityMatchAnswer, priority=self.priority
        )
        return answer.same
