"""Shared helpers for the extraction, resolution and reconciliation tests."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from lucy.core.events import AssistantMessage, ToolFinished, TurnEnded, UserMessage
from lucy.core.types import LLMChunk, ToolResult
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.graph_support import log

#: Thresholds that suit `HashingEmbedder` (bag-of-words cosine), not a real model.
TEST_POLICY_VALUES: dict[str, float] = {
    "statement_similarity_floor": 0.5,
    "near_identical_similarity": 0.95,
    "entity_match_similarity": 0.9,
    "entity_ambiguous_similarity": 0.5,
    "entity_type_mismatch_penalty": 0.2,
}


def reply(obj: Any) -> list[LLMChunk]:
    """A scripted `FakeLLM` response containing `obj` as JSON."""
    return [LLMChunk(text=json.dumps(obj))]


def cand(
    text: str,
    *,
    subject: tuple[str, str] = ("user", "person"),
    predicate: str | None = None,
    entity: tuple[str, str] | None = None,
    value: str | None = None,
    kind: str = "fact",
    origin: str = "stated",
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> dict[str, Any]:
    """One candidate statement as the extraction model would write it."""
    return {
        "subject": {"name": subject[0], "type": subject[1]},
        "predicate": predicate or text,
        "object_entity": None if entity is None else {"name": entity[0], "type": entity[1]},
        "object_value": value,
        "text": text,
        "kind": kind,
        "origin": origin,
        "valid_from": valid_from,
        "valid_until": valid_until,
    }


def extraction(*statements: dict[str, Any], user_name: str | None = None) -> list[LLMChunk]:
    """A scripted extraction response."""
    return reply({"user_name": user_name, "statements": list(statements)})


def decision(
    action: str, target: int | None = None, also_ends: list[int] | None = None
) -> list[LLMChunk]:
    """A scripted reconciliation response (`target` and `also_ends` are shown numbers)."""
    data: dict[str, Any] = {"action": action, "target": target}
    if also_ends is not None:
        data["also_ends"] = also_ends
    return reply(data)


def same(answer: bool) -> list[LLMChunk]:
    """A scripted entity-match response."""
    return reply({"same": answer})


async def turn(
    memory: SQLiteMemoryStore,
    turn_id: str,
    user: str | None,
    assistant: str | None = "OK.",
    *,
    ts: datetime,
    tool: str | None = None,
    ended: bool = True,
    session: str = "s1",
) -> list[int]:
    """Log one turn's events; returns the content episode ids (user message first)."""
    ids = []
    common = {"session_id": session, "turn_id": turn_id, "ts": ts}
    if user is not None:
        ids.append(await log(memory, UserMessage(text=user, **common)))
    if tool is not None:
        result = ToolResult(call_id="c1", content=tool)
        ids.append(await log(memory, ToolFinished(result=result, **common)))
    if assistant is not None:
        ids.append(await log(memory, AssistantMessage(text=assistant, **common)))
    if ended:
        await log(memory, TurnEnded(reason="completed", **common))
    return ids
