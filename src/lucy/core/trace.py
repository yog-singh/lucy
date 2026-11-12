"""Per-turn traces (ADR 0004).

A `TurnTrace` records what the model was actually shown at every step of a turn
and how the turn ended, so "why did she say that?" can be answered from storage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from lucy.core.context import SectionReport
from lucy.core.types import FinishReason, Message


class StepFallback(BaseModel):
    """How a step was served when its role's stream produced nothing (`LLMGateway`).

    `served_by` is the fallback role that was asked instead; `reason` says why
    the first attempt was given up; `lost_seconds` is the time spent on it.
    `at` is when the fallback attempt started.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    served_by: str
    reason: str
    lost_seconds: float
    at: AwareDatetime


class StepTrace(BaseModel):
    """One LLM call within a turn.

    `messages` are exactly the messages sent; `tools` are the names of the tools
    offered. `first_token_at` is when the first text or tool call arrived;
    `finish_reason` is unset if the model never reported one (for example
    because the step was interrupted). `fallback` is set when the role's stream
    produced nothing and another role was asked instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    messages: list[Message]
    tools: list[str] = Field(default_factory=list)
    started_at: AwareDatetime
    first_token_at: AwareDatetime | None = None
    ended_at: AwareDatetime
    finish_reason: FinishReason | None = None
    fallback: StepFallback | None = None


class TurnTrace(BaseModel):
    """Everything needed to explain one turn after the fact.

    `context` is the `ContextBuilder` report (empty if the turn failed before
    the context was built). `error` is set when `outcome` is "error".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn_id: str
    session_id: str
    started_at: AwareDatetime
    ended_at: AwareDatetime
    steps: list[StepTrace] = Field(default_factory=list)
    context: list[SectionReport] = Field(default_factory=list)
    outcome: Literal["completed", "interrupted", "error"]
    error: str | None = None
