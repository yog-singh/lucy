"""The core profile: a short generated summary of the user (docs/memory.md "Core profile",
"5. Rewrite the profile").

`ProfileWriter.rewrite` selects the strongest current statements about the
user, asks the `summary` model for a plain-text profile of at most about
`max_tokens` tokens, checks it, and saves it as a new profile version (the
store keeps the old ones).

Selection (`select_profile_input`): current statements only; never
instructions (they are loaded into every turn separately) and never uncertain
inferences (inferred with confidence below `min_inferred_confidence`). The
candidates are grouped into sections:

- identity: the user's facts that are not about a person or a project;
- people: the most-mentioned people around the user (by how many of the
  user's statements involve them), with their relationship to the user and a
  few of their own statements;
- work and projects: the user's facts about projects and organisations, and a
  few statements about the most-mentioned projects;
- goals, preferences, habits: the user's statements of those kinds;
- recent decisions: the most recently recorded decisions, about anyone.

Within a section, statements rank by confidence times a recency weight that
halves every `recency_half_life` since last observed. Sections are filled
round-robin until `input_token_budget` is used up, so every section is
represented before any gets long.

The rewrite is skipped when the rendered input (and the prompt) is unchanged
since the last rewrite and that rewrite is still the current profile (a hash
is kept in `memory_meta` under `INPUT_HASH_KEY`), unless forced. With no
eligible statements nothing is saved and no model is called.

Guard against invention: `unsupported_terms` lists capitalised words and
numbers in the output that occur in none of the input statements. If there
are any (or the text is too long) the model is asked once more with a firmer
instruction; the attempt with fewer unsupported terms is kept, cut at a
sentence boundary if still too long, and the remaining terms are logged and
reported in the result. This is a heuristic: it catches invented names,
places and numbers, not invented relationships between known words.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final, Literal

from lucy.core.context import estimate_tokens
from lucy.core.gateway import LLMGateway, Priority
from lucy.core.jobs import utc_now
from lucy.core.types import Message, Role
from lucy.memory.graph_store import DERIVED_META_PREFIX
from lucy.memory.models import (
    DEFAULT_USER_NAME,
    USER_ENTITY_ID,
    Entity,
    Origin,
    ProfileVersion,
    Statement,
    StatementKind,
)
from lucy.memory.ports import GraphStore
from lucy.prompts import load_prompt

log = logging.getLogger(__name__)

#: `memory_meta` key holding the hash of the last rewrite's input and the profile id it produced.
INPUT_HASH_KEY: Final = f"{DERIVED_META_PREFIX}profile.input"
#: Entity types treated as people, and as work (projects, organisations).
PERSON_TYPES: Final = frozenset({"person"})
WORK_TYPES: Final = frozenset(
    {"project", "organisation", "organization", "company", "employer", "team", "job", "work"}
)


# --- Shared text helpers (also used by entity maintenance) --------------------


async def complete_text(gateway: LLMGateway, role: str, messages: Sequence[Message]) -> str:
    """Stream a plain-text answer from `role` in the background lane and return it.

    `Preempted` propagates.
    """
    parts: list[str] = []
    stream = gateway.stream(role, messages, priority=Priority.BACKGROUND)
    async with aclosing(stream):
        async for chunk in stream:
            if chunk.text:
                parts.append(chunk.text)
    return "".join(parts)


_MARKDOWN_LINE: Final = re.compile(r"^\s*(?:#{1,6}\s+|[-*+•]\s+|\d+[.)]\s+|>\s*)")


def plain_text(text: str) -> str:
    """`text` as plain sentences: Markdown headings, bullets and emphasis removed,
    lines joined, whitespace collapsed."""
    lines = []
    for line in text.replace("\r", "").split("\n"):
        line = _MARKDOWN_LINE.sub("", line).strip()
        if not line:
            continue
        if lines and not re.search(r"[.!?:;]$", lines[-1]):
            lines[-1] += "."
        lines.append(line)
    joined = " ".join(lines)
    joined = re.sub(r"(\*\*|__|`)", "", joined)
    joined = re.sub(r"(?<!\w)[*_](\S[^*_]*?)[*_](?!\w)", r"\1", joined)
    return re.sub(r"\s+", " ", joined).strip()


_SENTENCE_END: Final = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def sentences(text: str) -> list[str]:
    """`text` split into sentences (after ., ! or ? followed by whitespace)."""
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """The longest prefix of whole sentences within `max_tokens` (estimated).

    If even the first sentence is too long it is cut at a word boundary and
    ends with an ellipsis.
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    kept: list[str] = []
    for sentence in sentences(text):
        if estimate_tokens(" ".join([*kept, sentence])) > max_tokens:
            break
        kept.append(sentence)
    if kept:
        return " ".join(kept)
    words = text.split()
    result = ""
    for word in words:
        if estimate_tokens(f"{result} {word}…".strip()) > max_tokens:
            break
        result = f"{result} {word}".strip()
    return f"{result}…"


#: Words that may start a sentence (or appear capitalised) without being names.
COMMON_WORDS: Final = frozenset(
    """
    a an the this that these those there here it its he she they them his her hers their theirs
    him we our you your i me my one ones who whom whose which what where when why how
    and but or nor so yet also as if then than though although while because since until unless
    after before during between among about above below across along around at by for from in
    into of off on onto out over through to toward towards under up upon with within without
    both each every either neither all any some most many much more few several other another
    such no not none only just even still already again often usually sometimes always never
    generally currently recently lately now today soon overall additionally furthermore moreover
    however meanwhile besides outside inside beyond despite instead otherwise therefore thus
    finally first second third next last later earlier personally professionally socially
    outside is are was were be been being has have had do does did will would can could should
    may might must shall lucy user users
    """.split()
)
_ABBREVIATIONS: Final = {
    "mon": "monday", "tue": "tuesday", "tues": "tuesday", "wed": "wednesday",
    "thu": "thursday", "thur": "thursday", "thurs": "thursday", "fri": "friday",
    "sat": "saturday", "sun": "sunday", "jan": "january", "feb": "february",
    "mar": "march", "apr": "april", "jun": "june", "jul": "july", "aug": "august",
    "sep": "september", "sept": "september", "oct": "october", "nov": "november",
    "dec": "december",
}  # fmt: skip
_WORD: Final = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+")


def _key(word: str) -> str:
    word = word.lower().replace("’", "'")
    return word[:-2] if word.endswith("'s") else word


def _vocabulary(sources: Iterable[str]) -> set[str]:
    vocab: set[str] = set()
    for source in sources:
        for word in _WORD.findall(source):
            key = _key(word)
            vocab.add(key)
            vocab.update(key.split("'"))
            if key in _ABBREVIATIONS:
                vocab.add(_ABBREVIATIONS[key])
            if key.endswith("s"):
                vocab.add(key[:-1])
    vocab |= {full for abbr, full in _ABBREVIATIONS.items() if abbr in vocab}
    return vocab


def unsupported_terms(text: str, sources: Iterable[str]) -> list[str]:
    """Capitalised words and numbers in `text` that occur in none of `sources`.

    Comparison ignores case, possessive "'s", a plural "s" and common
    weekday/month abbreviations. Common function words (`COMMON_WORDS`) are
    never flagged, so sentence-initial "She" or "Overall" pass. Each term is
    reported once, in order of appearance.
    """
    vocab = _vocabulary(sources)
    flagged: dict[str, None] = {}
    for match in _WORD.finditer(text):
        word = match.group()
        key = _key(word)
        if word[0].isdigit():
            if key not in vocab:
                flagged.setdefault(word)
            continue
        if not word[0].isupper() or key in COMMON_WORDS:
            continue
        variants = {key, key[:-1] if key.endswith("s") else key, _ABBREVIATIONS.get(key, key)}
        if not variants & vocab:
            flagged.setdefault(re.sub(r"['’]s$", "", word))
    return list(flagged)


# --- Selection ------------------------------------------------------------------


@dataclass(frozen=True)
class ProfilePolicy:
    """What goes into the profile, and how long it may be."""

    max_tokens: int = 500
    """Upper bound on the profile's length (estimated tokens), enforced in code."""
    target_words: int = 250
    """The length asked of the model; comfortably under `max_tokens`."""
    input_token_budget: int = 1500
    """Upper bound on the statements given to the model (estimated tokens)."""
    min_inferred_confidence: float = 0.6
    """Inferred statements below this are uncertain and left out."""
    recency_half_life: timedelta = timedelta(days=180)
    key_people: int = 6
    statements_per_person: int = 3
    key_projects: int = 3
    statements_per_project: int = 2
    recent_decisions: int = 5


@dataclass(frozen=True)
class ProfileInput:
    """The statements chosen for the profile, by section, and their rendering."""

    user_name: str
    sections: list[tuple[str, list[Statement]]]
    text: str
    """The rendered input given to the model (empty when there is nothing to say)."""

    @property
    def statements(self) -> list[Statement]:
        return [s for _, items in self.sections for s in items]

    @property
    def statement_ids(self) -> list[int]:
        return [s.id for s in self.statements]


def _eligible(statement: Statement, policy: ProfilePolicy, now: datetime) -> bool:
    if statement.kind is StatementKind.INSTRUCTION or not statement.is_current(now):
        return False
    return not (
        statement.origin is Origin.INFERRED
        and statement.confidence < policy.min_inferred_confidence
    )


def _score(statement: Statement, policy: ProfilePolicy, now: datetime) -> float:
    age = max((now - statement.last_observed_at) / policy.recency_half_life, 0.0)
    return statement.confidence * math.pow(0.5, age)


def _render_line(statement: Statement) -> str:
    suffix = " (inferred)" if statement.origin is Origin.INFERRED else ""
    return f"- {statement.text.strip()}{suffix}"


async def select_profile_input(
    graph: GraphStore, *, policy: ProfilePolicy | None = None, now: datetime
) -> ProfileInput:
    """Choose and render the statements the profile is written from (see the module doc)."""
    policy = policy or ProfilePolicy()
    user = await graph.user_entity()
    user_name = user.name if user.name != DEFAULT_USER_NAME else "the user"
    about_user = [s for s in await graph.neighbours(USER_ENTITY_ID) if _eligible(s, policy, now)]

    def other(statement: Statement) -> int | None:
        if statement.subject_id != USER_ENTITY_ID:
            return statement.subject_id
        return statement.object_id if statement.object_id != USER_ENTITY_ID else None

    others = {eid for s in about_user if (eid := other(s)) is not None}
    entities: dict[int, Entity] = {e.id: e for e in await graph.get_entities(others)}

    def type_of(eid: int | None) -> str:
        entity = entities.get(eid) if eid is not None else None
        return entity.type.strip().lower() if entity else ""

    mentions = Counter(eid for s in about_user if (eid := other(s)) is not None)
    best = {eid: max(s.confidence for s in about_user if other(s) == eid) for eid in mentions}

    def top(types: frozenset[str], n: int) -> list[int]:
        ids = [eid for eid in mentions if type_of(eid) in types]
        ids.sort(key=lambda eid: (-mentions[eid], -best[eid], eid))
        return ids[:n]

    key_people = top(PERSON_TYPES, policy.key_people)
    key_projects = top(WORK_TYPES, policy.key_projects)

    identity: list[Statement] = []
    people: list[Statement] = []
    work: list[Statement] = []
    by_kind: dict[StatementKind, list[Statement]] = {
        StatementKind.GOAL: [],
        StatementKind.PREFERENCE: [],
        StatementKind.HABIT: [],
        StatementKind.DECISION: [],
    }
    for statement in about_user:
        eid = other(statement)
        if statement.kind in by_kind:
            by_kind[statement.kind].append(statement)
        elif type_of(eid) in PERSON_TYPES:
            if eid in key_people:
                people.append(statement)
        elif type_of(eid) in WORK_TYPES:
            work.append(statement)
        else:
            identity.append(statement)

    seen = {s.id for s in about_user}

    async def extra(entity_ids: list[int], per_entity: int, into: list[Statement]) -> None:
        for eid in entity_ids:
            found = [
                s
                for s in await graph.neighbours(eid, limit=per_entity * 4 + 4)
                if s.id not in seen and _eligible(s, policy, now)
            ]
            found.sort(key=lambda s: (-_score(s, policy, now), s.id))
            for statement in found[:per_entity]:
                seen.add(statement.id)
                into.append(statement)

    await extra(key_people, policy.statements_per_person, people)
    await extra(key_projects, policy.statements_per_project, work)

    decisions = by_kind[StatementKind.DECISION] + [
        s
        for s in await graph.list_statements(kinds=[StatementKind.DECISION])
        if s.id not in seen and _eligible(s, policy, now)
    ]
    decisions.sort(key=lambda s: (s.recorded_at, s.id), reverse=True)
    decisions = decisions[: policy.recent_decisions]

    ranked = [
        ("Identity and life", identity),
        ("People", people),
        ("Work and projects", work),
        ("Goals", by_kind[StatementKind.GOAL]),
        ("Preferences", by_kind[StatementKind.PREFERENCE]),
        ("Habits", by_kind[StatementKind.HABIT]),
    ]
    for _, items in ranked:
        items.sort(key=lambda s: (-_score(s, policy, now), s.id))
    ranked.append(("Recent decisions", decisions))  # already newest first

    # Round-robin fill within the token budget.
    header = f"Name: {user_name}\n"
    used = estimate_tokens(header)
    chosen: list[list[Statement]] = [[] for _ in ranked]
    cursors = [0] * len(ranked)
    full = False
    while not full and any(c < len(items) for c, (_, items) in zip(cursors, ranked, strict=True)):
        for i, (title, items) in enumerate(ranked):
            if cursors[i] >= len(items):
                continue
            statement = items[cursors[i]]
            cost = estimate_tokens(_render_line(statement)) + 1
            if not chosen[i]:
                cost += estimate_tokens(f"## {title}") + 1
            if used + cost > policy.input_token_budget:
                full = True
                break
            used += cost
            chosen[i].append(statement)
            cursors[i] += 1

    sections = [(title, items) for (title, _), items in zip(ranked, chosen, strict=True) if items]
    if not sections:
        return ProfileInput(user_name=user_name, sections=[], text="")
    lines = [header.rstrip()]
    for title, items in sections:
        lines.append(f"\n## {title}")
        lines.extend(_render_line(s) for s in items)
    return ProfileInput(user_name=user_name, sections=sections, text="\n".join(lines) + "\n")


# --- Writing --------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileResult:
    """What `ProfileWriter.rewrite` did.

    `status`: "written" (a new version was saved), "unchanged" (input as
    before; nothing called or saved) or "empty" (no eligible statements).
    """

    status: Literal["written", "unchanged", "empty"]
    profile: ProfileVersion | None = None
    statement_ids: list[int] = field(default_factory=list)
    input_tokens: int = 0
    attempts: int = 0
    truncated: bool = False
    flagged: list[str] = field(default_factory=list)
    """Capitalised words or numbers in the saved profile found in no input statement."""


class ProfileWriter:
    """Rewrites the core profile from the graph with the `role` model."""

    def __init__(
        self,
        gateway: LLMGateway,
        graph: GraphStore,
        *,
        role: str = "summary",
        policy: ProfilePolicy | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._gateway = gateway
        self._graph = graph
        self._role = role
        self._policy = policy or ProfilePolicy()
        self._clock = clock

    def _system_prompt(self, user_name: str) -> str:
        return (
            load_prompt("profile")
            .replace("{name}", user_name)
            .replace("{max_words}", str(self._policy.target_words))
        )

    async def rewrite(self, *, force: bool = False) -> ProfileResult:
        """Regenerate and save the profile if its input changed (or `force`).

        `Preempted` propagates; nothing is saved then, so a rerun starts over.
        """
        now = self._clock()
        selected = await select_profile_input(self._graph, policy=self._policy, now=now)
        if not selected.sections:
            log.info("profile: no eligible statements; nothing written")
            return ProfileResult(status="empty")
        system = self._system_prompt(selected.user_name)
        digest = hashlib.sha256(
            json.dumps([system, selected.text, self._policy.max_tokens]).encode()
        ).hexdigest()
        input_tokens = estimate_tokens(selected.text)
        if not force:
            stored = await self._graph.get_meta(INPUT_HASH_KEY)
            current = await self._graph.current_profile()
            if stored and current is not None:
                try:
                    data = json.loads(stored)
                except ValueError:
                    data = {}
                if data.get("hash") == digest and data.get("profile_id") == current.id:
                    return ProfileResult(
                        status="unchanged",
                        profile=current,
                        statement_ids=selected.statement_ids,
                        input_tokens=input_tokens,
                    )

        sources = [s.text for s in selected.statements] + [selected.user_name]
        messages = [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=selected.text),
        ]
        text, attempts, flagged = await self._generate(messages, sources)
        truncated = estimate_tokens(text) > self._policy.max_tokens
        if truncated:
            text = truncate_to_tokens(text, self._policy.max_tokens)
            flagged = unsupported_terms(text, sources)
        if flagged:
            log.warning("profile mentions terms not in its input statements: %s", flagged)
        profile = await self._graph.save_profile(text)
        await self._graph.set_meta(
            INPUT_HASH_KEY, json.dumps({"hash": digest, "profile_id": profile.id})
        )
        return ProfileResult(
            status="written",
            profile=profile,
            statement_ids=selected.statement_ids,
            input_tokens=input_tokens,
            attempts=attempts,
            truncated=truncated,
            flagged=flagged,
        )

    async def _generate(
        self, messages: list[Message], sources: list[str]
    ) -> tuple[str, int, list[str]]:
        """Ask once, and once more if the text is too long or mentions unsupported terms."""
        limit = self._policy.max_tokens
        first = plain_text(await complete_text(self._gateway, self._role, messages))
        first_flags = unsupported_terms(first, sources)
        problems = []
        if estimate_tokens(first) > limit or not first:
            words = len(first.split())
            problems.append(
                f"It is {words} words long. Rewrite it in at most"
                f" {self._policy.target_words} words: keep the most important statements"
                " and drop the rest."
            )
        if first_flags:
            problems.append(
                "It mentions " + ", ".join(first_flags) + ", which appear in none of the"
                " statements. Use only names, places and numbers that appear in the statements."
            )
        if not problems:
            return first, 1, first_flags
        retry = [
            *messages,
            Message(role=Role.ASSISTANT, content=first),
            Message(
                role=Role.USER,
                content="Your profile breaks the rules. "
                + " ".join(problems)
                + " Write the corrected profile as plain sentences, nothing else.",
            ),
        ]
        second = plain_text(await complete_text(self._gateway, self._role, retry))
        second_flags = unsupported_terms(second, sources)
        if not second or len(second_flags) > len(first_flags):
            return first or second, 2, first_flags if first else second_flags
        return second, 2, second_flags
