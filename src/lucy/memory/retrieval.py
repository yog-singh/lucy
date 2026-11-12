"""Per-turn memory retrieval (docs/memory.md "Retrieval").

`Retriever.retrieve` returns the current profile and all active instructions
(always), plus the statements most relevant to the user's message, ranked and
packed into a token budget. Relevance combines three channels over current,
non-instruction statements:

(a) vector similarity between the query (embedded with `kind="query"`) and
    statement texts;
(b) full-text keyword match on statement texts;
(c) entities named in the query (by name or alias) or similar to it (by vector
    over name plus summary), expanded one hop to their current statements.

The three ranked lists are fused with reciprocal rank fusion, then weighted by
confidence and recency (`last_observed_at`); a stated statement always outranks
an inferred one with the same subject and predicate. At most `max_statements`
are returned.

Gating (what keeps unrelated statements out). Measured with `embeddinggemma-2`
("query: " / "passage: " prefixes) on a test set of 61 statements about an
invented user and 44 queries (scratch `retrieval_eval`, 2026-11-12): unrelated
statements score 0.53 to 0.76 against a query and relevant ones 0.61 to 0.85;
the best match of a personal question scored 0.67 to 0.85, of a general-knowledge
question or small talk 0.62 to 0.74. So:

- vector matches must reach `vector_min_similarity` (absolute) and lie within
  `vector_margin` of the best match (relative);
- keyword matches must also be at least `keyword_min_similarity` similar by
  vector (a shared common word such as "works" or "new" is not relevance), and
  the keyword channel searches content words only (no fallback to stopwords);
- entities found by vector similarity must reach `entity_min_similarity`: entity
  vectors (name plus summary) score 0.58 to 0.72 against unrelated queries, so a
  low floor expanded five unrelated people on every turn (the "pets" question
  ranked a statement about the user's brother first). Entities named in the
  query are always expanded.

With these defaults: recall@5 0.86 (0.71 before), MRR 0.93 (0.71), 2.2
irrelevant statements per personal question (39), and 6 of 11 general or
small-talk queries retrieve nothing (0 before; each got about 41).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from lucy.core.context import estimate_tokens
from lucy.core.jobs import utc_now
from lucy.core.ports import Embedder
from lucy.core.types import MemoryContext
from lucy.memory import fts
from lucy.memory.models import USER_ENTITY_ID, Entity, Origin, Statement, StatementKind
from lucy.memory.ports import GraphStore

log = logging.getLogger(__name__)

#: Kinds searched per query; instructions are always included separately.
SEARCHED_KINDS: Final = tuple(k for k in StatementKind if k is not StatementKind.INSTRUCTION)

#: Reciprocal rank fusion constant: a hit at rank r in one list scores 1 / (RRF_K + r).
#: Larger values flatten the difference between top and lower ranks.
RRF_K: Final = 60
#: Share of the score that depends on confidence: the factor is
#: (1 - CONFIDENCE_WEIGHT) + CONFIDENCE_WEIGHT * confidence.
CONFIDENCE_WEIGHT: Final = 0.5
#: Share of the score that depends on recency: the factor is
#: (1 - RECENCY_WEIGHT) + RECENCY_WEIGHT * 0.5 ** (age / RECENCY_HALF_LIFE),
#: where age is the time since `last_observed_at`.
RECENCY_WEIGHT: Final = 0.3
RECENCY_HALF_LIFE: Final = timedelta(days=90)
#: Inferred statements below this confidence are marked as unconfirmed.
UNCERTAIN_BELOW: Final = 0.6
#: Candidates taken from the vector and keyword channels.
VECTOR_K: Final = 40
#: Vector hits less similar than this (cosine) are dropped: KNN always returns the
#: k nearest, however unrelated. Specific to the embedding model (see the module
#: docstring for the measurement); overridable in `[retrieval]`.
VECTOR_MIN_SIMILARITY: Final = 0.68
#: Vector hits more than this below the best hit are dropped (None: no relative floor).
VECTOR_MARGIN: Final = 0.06
KEYWORD_K: Final = 40
#: Keyword hits need at least this vector similarity to the query (None: no check;
#: skipped when the query could not be embedded). A hit outside the `vector_k`
#: nearest statements counts as below it.
KEYWORD_MIN_SIMILARITY: Final = 0.62
#: Entities found by vector similarity: at most ENTITY_K, each at least this similar.
ENTITY_K: Final = 5
ENTITY_MIN_SIMILARITY: Final = 0.75
#: Statements returned per turn at most (None: as many as the budget allows).
MAX_STATEMENTS: Final = 8
#: Statements taken per expanded entity (its strongest, most recent ones).
EXPANSION_LIMIT: Final = 15
#: The marker appended to an uncertain inference.
UNCERTAIN_MARKER: Final = "unconfirmed inference"


@dataclass(frozen=True)
class Ranked:
    """A candidate with its fused relevance and final score (for tests and debugging)."""

    statement: Statement
    relevance: float
    score: float


def render(statement: Statement, *, now: datetime, uncertain_below: float = UNCERTAIN_BELOW) -> str:
    """The statement as one memory line: its text, plus a bracketed note when needed.

    The note marks an uncertain inference (inferred, confidence below
    `uncertain_below`) and gives validity dates (UTC days): "since" a past
    `valid_from`, "from" a future one, "until" an `invalid_at`. For example
    "Goes to the gym on Mondays [unconfirmed inference; since 2026-03-02]".
    """
    notes = []
    if statement.origin is Origin.INFERRED and statement.confidence < uncertain_below:
        notes.append(UNCERTAIN_MARKER)
    dates = []
    if statement.valid_from is not None:
        word = "since" if statement.valid_from <= now else "from"
        dates.append(f"{word} {statement.valid_from.date().isoformat()}")
    if statement.invalid_at is not None:
        dates.append(f"until {statement.invalid_at.date().isoformat()}")
    if dates:
        notes.append(", ".join(dates))
    text = " ".join(statement.text.split())
    return f"{text} [{'; '.join(notes)}]" if notes else text


class Retriever:
    """Builds the `MemoryContext` for a turn from the knowledge graph.

    `embedder` embeds queries; it must have the same model and dimensions as
    the graph's embedder, whose vectors it searches. `count_tokens` must match
    the context builder's counter. All weighting constants are parameters,
    defaulting to the module constants documented above.
    """

    def __init__(
        self,
        graph: GraphStore,
        embedder: Embedder,
        *,
        count_tokens: Callable[[str], int] = estimate_tokens,
        clock: Callable[[], datetime] = utc_now,
        rrf_k: int = RRF_K,
        confidence_weight: float = CONFIDENCE_WEIGHT,
        recency_weight: float = RECENCY_WEIGHT,
        recency_half_life: timedelta = RECENCY_HALF_LIFE,
        uncertain_below: float = UNCERTAIN_BELOW,
        vector_k: int = VECTOR_K,
        vector_min_similarity: float = VECTOR_MIN_SIMILARITY,
        vector_margin: float | None = VECTOR_MARGIN,
        keyword_k: int = KEYWORD_K,
        keyword_min_similarity: float | None = KEYWORD_MIN_SIMILARITY,
        entity_k: int = ENTITY_K,
        entity_min_similarity: float = ENTITY_MIN_SIMILARITY,
        expansion_limit: int = EXPANSION_LIMIT,
        max_statements: int | None = MAX_STATEMENTS,
        expand_user: bool = False,
    ) -> None:
        mine, theirs = embedder.capabilities, graph.embedder.capabilities
        if (mine.model, mine.dimensions) != (theirs.model, theirs.dimensions):
            raise ValueError(
                f"the retriever's embedder ({mine.model}, {mine.dimensions}) differs from the"
                f" graph's ({theirs.model}, {theirs.dimensions}); their vectors are not comparable"
            )
        if not 0 <= confidence_weight <= 1 or not 0 <= recency_weight <= 1:
            raise ValueError("confidence_weight and recency_weight must be within [0, 1]")
        if recency_half_life <= timedelta(0):
            raise ValueError("recency_half_life must be positive")
        if vector_margin is not None and vector_margin < 0:
            raise ValueError("vector_margin must not be negative")
        if max_statements is not None and max_statements < 1:
            raise ValueError("max_statements must be at least 1")
        self._graph = graph
        self._embedder = embedder
        self._count = count_tokens
        self._clock = clock
        self.rrf_k = rrf_k
        self.confidence_weight = confidence_weight
        self.recency_weight = recency_weight
        self.recency_half_life = recency_half_life
        self.uncertain_below = uncertain_below
        self.vector_k = vector_k
        self.vector_min_similarity = vector_min_similarity
        self.vector_margin = vector_margin
        self.keyword_k = keyword_k
        self.keyword_min_similarity = keyword_min_similarity
        self.entity_k = entity_k
        self.entity_min_similarity = entity_min_similarity
        self.expansion_limit = expansion_limit
        self.max_statements = max_statements
        #: Whether the last query embedding failed (to log a failure and recovery once).
        self._embedder_failing = False
        #: Whether the user entity is expanded when it is matched. Off by default:
        #: most statements are about the user, so expanding it adds no signal.
        self.expand_user = expand_user

    async def retrieve(self, query: str, budget: int) -> MemoryContext:
        """The memory context for a turn whose user input is `query`.

        `profile` is the current profile and `instructions` the texts of all
        active instructions (oldest first); neither counts against `budget`.
        `statements` are the best-ranked current non-instruction statements,
        rendered by `render`, in rank order, as many as fit `budget` tokens
        when each is counted as the context builder renders it ("- " + line),
        and at most `max_statements`. A blank query or non-positive budget
        yields no statements. If the embedder fails, the vector channels are
        skipped and keyword and name matching still run (a warning is logged
        when it starts failing, and an info line when it works again).
        """
        profile = await self._graph.current_profile()
        instructions = await self._graph.active_instructions()
        context = MemoryContext(
            profile=profile.content if profile else "",
            instructions=[s.text for s in instructions],
        )
        if not query.strip() or budget <= 0:
            return context
        now = self._clock()
        lines: list[str] = []
        remaining = budget
        for ranked in await self.rank(query):
            if self.max_statements is not None and len(lines) >= self.max_statements:
                break
            line = render(ranked.statement, now=now, uncertain_below=self.uncertain_below)
            cost = self._count(f"- {line}")
            if cost <= remaining:
                lines.append(line)
                remaining -= cost
        return context.model_copy(update={"statements": lines})

    async def _embed_query(self, query: str) -> list[float] | None:
        try:
            [vector] = await self._embedder.embed([query], kind="query")
            if len(vector) != self._embedder.capabilities.dimensions:
                raise ValueError(
                    f"query embedding has {len(vector)} dimensions,"
                    f" expected {self._embedder.capabilities.dimensions}"
                )
        except Exception as exc:
            if not self._embedder_failing:
                self._embedder_failing = True
                log.warning(
                    "query embedding failed (%s: %s); retrieving without vectors until it"
                    " works again",
                    type(exc).__name__,
                    exc,
                )
            else:
                log.debug("query embedding still failing: %s: %s", type(exc).__name__, exc)
            return None
        if self._embedder_failing:
            self._embedder_failing = False
            log.info("query embedding works again; retrieving with vectors")
        return [float(x) for x in vector]

    async def _entities(self, query: str, vector: list[float] | None) -> list[Entity]:
        """Entities named in the query, then those similar to it; de-duplicated."""
        found: dict[int, Entity] = {e.id: e for e in await self._graph.entities_mentioned_in(query)}
        if vector is not None and self.entity_k > 0:
            for hit in await self._graph.similar_entities_by_vector(vector, self.entity_k):
                if hit.score >= self.entity_min_similarity:
                    found.setdefault(hit.entity.id, hit.entity)
        return [e for e in found.values() if self.expand_user or e.id != USER_ENTITY_ID]

    async def rank(self, query: str) -> list[Ranked]:
        """All candidates for `query` that pass the gates, best first (no budget or
        `max_statements` applied)."""
        vector = await self._embed_query(query)
        lists: list[list[Statement]] = []
        similarity: dict[int, float] = {}
        if vector is not None:
            hits = await self._graph.similar_statements_by_vector(
                vector, k=self.vector_k, kinds=SEARCHED_KINDS
            )
            similarity = {h.statement.id: h.score for h in hits}
            floor = self.vector_min_similarity
            if hits and self.vector_margin is not None:
                floor = max(floor, hits[0].score - self.vector_margin)
            lists.append([h.statement for h in hits if h.score >= floor])
        keywords = " ".join(w for w in fts.tokens(query) if w not in fts.STOPWORDS)
        keyword_hits = []
        if keywords:
            keyword_hits = [
                h.statement
                for h in await self._graph.keyword_statements(
                    keywords, k=self.keyword_k, kinds=SEARCHED_KINDS
                )
            ]
        if vector is not None and self.keyword_min_similarity is not None:
            least = self.keyword_min_similarity
            keyword_hits = [s for s in keyword_hits if similarity.get(s.id, -2.0) >= least]
        lists.append(keyword_hits)
        expanded: list[Statement] = []
        for entity in await self._entities(query, vector):
            expanded.extend(
                await self._graph.neighbours(
                    entity.id, kinds=SEARCHED_KINDS, limit=self.expansion_limit
                )
            )
        lists.append(expanded)
        return self.fuse(lists)

    def fuse(self, lists: list[list[Statement]]) -> list[Ranked]:
        """Fuse ranked lists, weight, enforce stated-over-inferred, de-duplicate, sort."""
        relevance: dict[int, float] = {}
        statements: dict[int, Statement] = {}
        for ranked in lists:
            seen: set[int] = set()
            for rank, statement in enumerate(ranked, start=1):
                if statement.id in seen:
                    continue
                seen.add(statement.id)
                statements[statement.id] = statement
                relevance[statement.id] = relevance.get(statement.id, 0.0) + 1 / (self.rrf_k + rank)
        now = self._clock()
        score = {sid: relevance[sid] * self._weight(s, now) for sid, s in statements.items()}

        # A stated statement outranks every inferred one with the same subject and predicate.
        floor: dict[tuple[int, str], float] = {}
        for sid, s in statements.items():
            if s.origin is Origin.STATED:
                key = (s.subject_id, fts.normalise(s.predicate))
                floor[key] = min(floor.get(key, math.inf), score[sid])
        for sid, s in statements.items():
            key = (s.subject_id, fts.normalise(s.predicate))
            if s.origin is Origin.INFERRED and key in floor and score[sid] >= floor[key]:
                score[sid] = floor[key] * 0.999

        order = sorted(statements, key=lambda sid: (-score[sid], sid))
        result: list[Ranked] = []
        texts: set[str] = set()
        for sid in order:
            key = fts.normalise(statements[sid].text)
            if key in texts:
                continue  # the same sentence twice (e.g. stated and inferred): keep the best
            texts.add(key)
            result.append(Ranked(statements[sid], relevance[sid], score[sid]))
        return result

    def _weight(self, statement: Statement, now: datetime) -> float:
        confidence = (1 - self.confidence_weight) + self.confidence_weight * statement.confidence
        age = max(timedelta(0), now - statement.last_observed_at)
        decay = 0.5 ** (age / self.recency_half_life)
        recency = (1 - self.recency_weight) + self.recency_weight * decay
        return confidence * recency
