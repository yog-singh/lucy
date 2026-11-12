"""The single entry point for model calls: LLMs by role, in two lanes (ADR 0005, 0007).

Lane rules:

- FOREGROUND streams never wait.
- A BACKGROUND stream does not start while any FOREGROUND stream is active; it
  waits until none is.
- When a FOREGROUND stream starts, every BACKGROUND stream already running is
  pre-empted: its adapter stream is closed and its consumer gets `Preempted` at
  its current or next await on the stream. Pre-empted work is not resumed;
  callers retry it (a retried BACKGROUND stream waits for the foreground lane
  to go idle again, like any new one).

A stream counts as active (and, for BACKGROUND, as running) from its first
iteration until it finishes, raises, is cancelled or is closed by the consumer.

Foreground guarantee: starting a FOREGROUND stream never awaits anything to do
with background streams. Pre-emption is requested synchronously before the
foreground stream asks its adapter for the first chunk: an in-flight background
read is cancelled, and a background adapter stream suspended between chunks is
handed to a closing task. The background adapters' cleanup (closing their HTTP
responses, which makes the server stop generating) then runs concurrently with
the foreground request, on the event loop's next iterations; it involves no
waiting on the model.

Consumers that may stop iterating early should close the stream deterministically,
e.g. with `contextlib.aclosing`; otherwise the lane is only released when the
abandoned generator is finalised.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from lucy.core import registry as _registry
from lucy.core.config import AdapterConfig, LLMSettings
from lucy.core.ports import LLM
from lucy.core.types import LLMChunk, Message, Role, ToolSpec

logger = logging.getLogger(__name__)


class GatewayError(LookupError):
    """An unknown role, or an adapter that is not an LLM."""


class Preempted(Exception):
    """A BACKGROUND stream was stopped because a FOREGROUND stream started.

    The work is not resumed; retry it later. Deliberately not a subclass of
    `asyncio.CancelledError`: the consumer's task is not being cancelled.
    """


class StructuredOutputError(ValueError):
    """The model did not produce valid output for the requested type, even after retries.

    `raw` is the model's last raw output; `errors` describes why it was rejected.
    """

    def __init__(self, message: str, *, raw: str, errors: str) -> None:
        super().__init__(message)
        self.raw = raw
        self.errors = errors


class Priority(Enum):
    """Which lane a model call runs in."""

    FOREGROUND = "foreground"
    BACKGROUND = "background"


_DONE: Final = object()
"""Returned by `_next` when the adapter stream is exhausted."""


async def _next(iterator: AsyncIterator[LLMChunk]) -> object:
    # StopAsyncIteration must not escape a task into an async generator, where
    # it would become a RuntimeError; map it to a sentinel instead.
    try:
        return await anext(iterator)
    except StopAsyncIteration:
        return _DONE


def _open(
    llm: LLM,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    response_schema: dict[str, Any] | None,
) -> AsyncIterator[LLMChunk]:
    # `response_schema` is only passed when set, so adapters written against the
    # older port signature keep working for plain streams.
    if response_schema is None:
        return llm.stream(messages, tools)
    return llm.stream(messages, tools, response_schema=response_schema)


async def _aclose(iterator: AsyncIterator[LLMChunk]) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        await aclose()


async def _aclose_logged(iterator: AsyncIterator[LLMChunk]) -> None:
    try:
        await _aclose(iterator)
    except Exception:
        logger.exception("closing a pre-empted background stream failed")


@dataclass(eq=False)
class _Background:
    """A running BACKGROUND stream, as seen by the pre-emption logic."""

    iterator: AsyncIterator[LLMChunk]
    preempted: bool = False
    step: asyncio.Task[object] | None = None
    """The in-flight read from the adapter, if the consumer is awaiting one."""
    closer: asyncio.Task[None] | None = None
    """Closes the adapter stream when pre-empted between reads."""

    def preempt(self) -> None:
        if self.preempted:
            return
        self.preempted = True
        if self.step is not None and not self.step.done():
            self.step.cancel()
        elif self.closer is None:
            self.closer = asyncio.create_task(_aclose_logged(self.iterator))


# --- Structured output helpers -------------------------------------------------

_FENCE: Final = re.compile(r"\A```[A-Za-z0-9_-]*[ \t]*\n?(?P<body>.*?)\n?[ \t]*```\Z", re.DOTALL)


def _strip_fence(text: str) -> str:
    """`text` without surrounding whitespace and a surrounding Markdown code fence."""
    text = text.strip()
    match = _FENCE.match(text)
    return match.group("body").strip() if match else text


def _describe(exc: ValidationError) -> str:
    """A compact, model-readable list of validation errors."""
    lines = []
    for error in exc.errors(include_url=False)[:10]:
        location = ".".join(str(part) for part in error["loc"]) or "(document)"
        lines.append(f"- {location}: {error['msg']}")
    if exc.error_count() > 10:
        lines.append(f"- ... and {exc.error_count() - 10} more")
    return "\n".join(lines)


def _schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "Respond with a single JSON document that conforms to the JSON Schema below. "
        "Output only the JSON: no explanation and no Markdown.\n\n"
        f"JSON Schema:\n{json.dumps(schema, indent=2)}"
    )


def _correction(errors: str, *, native: bool) -> str:
    text = f"Your previous output was not valid:\n{errors}\n\nReply again with the corrected JSON"
    if native:
        return text + " only."
    return text + ", following the JSON Schema given above, and nothing else."


# --- Gateway --------------------------------------------------------------------


class LLMGateway:
    """Hands out LLMs by role and schedules their streams by priority."""

    def __init__(self, llms: Mapping[str, LLM]) -> None:
        self._llms = dict(llms)
        self._foreground_active = 0
        self._background_waiting = 0
        self._background_running: set[_Background] = set()
        self._foreground_idle = asyncio.Event()
        self._foreground_idle.set()

    @classmethod
    def from_settings(
        cls, llm_settings: LLMSettings, registry: _registry.Registry | None = None
    ) -> LLMGateway:
        """Create one adapter per distinct role config through `registry` (default: global).

        Roles whose `AdapterConfig` is identical share one adapter instance.
        """
        create = registry.create if registry is not None else _registry.create
        created: list[tuple[AdapterConfig, LLM]] = []
        llms: dict[str, LLM] = {}
        for role, config in llm_settings.roles.items():
            llm = next((inst for cfg, inst in created if cfg == config), None)
            if llm is None:
                llm = create("llm", config.adapter, **config.options)
                if not isinstance(llm, LLM):
                    raise GatewayError(
                        f"adapter {config.adapter!r} for role {role!r} is not an LLM"
                    )
                created.append((config, llm))
            llms[role] = llm
        return cls(llms)

    def for_role(self, role: str) -> LLM:
        """The LLM configured for `role`."""
        try:
            return self._llms[role]
        except KeyError:
            known = ", ".join(sorted(self._llms)) or "none"
            raise GatewayError(f"no LLM for role {role!r}; configured roles: {known}") from None

    def roles(self) -> list[str]:
        """Sorted names of the configured roles."""
        return sorted(self._llms)

    @property
    def foreground_active(self) -> int:
        """Number of FOREGROUND streams currently active."""
        return self._foreground_active

    @property
    def background_waiting(self) -> int:
        """Number of BACKGROUND streams waiting for the foreground lane to go idle."""
        return self._background_waiting

    @property
    def background_running(self) -> int:
        """Number of BACKGROUND streams currently running (including ones being pre-empted)."""
        return len(self._background_running)

    async def stream(
        self,
        role: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        priority: Priority = Priority.FOREGROUND,
        response_schema: dict[str, Any] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Stream the response of the `role` LLM, scheduled by `priority`.

        `response_schema` is passed to the adapter (see `LLM.stream`). Raises
        `GatewayError` (on first iteration) if `role` is unknown, and `Preempted`
        if a BACKGROUND stream is pre-empted.
        """
        llm = self.for_role(role)
        lane = self._foreground if priority is Priority.FOREGROUND else self._background
        # Close the lane generator deterministically when this one is closed.
        async with aclosing(lane(llm, messages, tools, response_schema)) as chunks:
            async for chunk in chunks:
                yield chunk

    async def _foreground(
        self,
        llm: LLM,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        response_schema: dict[str, Any] | None,
    ) -> AsyncGenerator[LLMChunk, None]:
        self._foreground_active += 1
        self._foreground_idle.clear()
        try:
            for background in list(self._background_running):
                background.preempt()
            iterator = _open(llm, messages, tools, response_schema)
            try:
                async for chunk in iterator:
                    yield chunk
            finally:
                # Close the adapter's generator however iteration ended.
                await _aclose(iterator)
        finally:
            self._foreground_active -= 1
            if self._foreground_active == 0:
                self._foreground_idle.set()

    async def _background(
        self,
        llm: LLM,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        response_schema: dict[str, Any] | None,
    ) -> AsyncGenerator[LLMChunk, None]:
        self._background_waiting += 1
        try:
            while self._foreground_active:
                await self._foreground_idle.wait()
        finally:
            self._background_waiting -= 1
        # No await between the wait above and registering below, so a foreground
        # stream cannot start unnoticed in between.
        iterator = _open(llm, messages, tools, response_schema)
        running = _Background(iterator)
        self._background_running.add(running)
        try:
            while True:
                if running.preempted:
                    raise Preempted("background stream pre-empted by a foreground stream")
                # Each read runs in its own task so that pre-emption can cancel it
                # without cancelling the consumer's task.
                running.step = asyncio.create_task(_next(iterator))
                try:
                    item = await running.step
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if running.preempted and (task is None or not task.cancelling()):
                        raise Preempted(
                            "background stream pre-empted by a foreground stream"
                        ) from None
                    raise
                finally:
                    step, running.step = running.step, None
                    if not step.done():  # the consumer itself was cancelled
                        step.cancel()
                        await asyncio.wait([step])
                if item is _DONE:
                    return
                assert isinstance(item, LLMChunk)
                yield item
        finally:
            self._background_running.discard(running)
            if running.closer is not None:
                await running.closer
            else:
                await _aclose(iterator)

    async def generate[T: BaseModel](
        self,
        role: str,
        messages: Sequence[Message],
        model_type: type[T],
        *,
        priority: Priority = Priority.BACKGROUND,
        retries: int = 1,
    ) -> T:
        """Ask the `role` LLM for a JSON document and validate it as `model_type`.

        If the adapter declares `capabilities.json_schema`, the schema of
        `model_type` is sent as `response_schema`; otherwise (ADR 0008) an
        instruction containing the schema is appended to the prompt. Surrounding
        whitespace and a Markdown code fence around the JSON are tolerated.

        Invalid output is retried up to `retries` times, with the validation
        errors added to the conversation; after that `StructuredOutputError`
        (carrying the last raw output) is raised. `Preempted`, adapter errors and
        `GatewayError` propagate without retrying.
        """
        if retries < 0:
            raise ValueError("retries must be >= 0")
        llm = self.for_role(role)
        schema = model_type.model_json_schema()
        native = llm.capabilities.json_schema
        conversation = list(messages)
        if not native:
            conversation.append(Message(role=Role.USER, content=_schema_instruction(schema)))
        raw = ""
        errors = ""
        for _attempt in range(retries + 1):
            stream = self.stream(
                role,
                conversation,
                priority=priority,
                response_schema=schema if native else None,
            )
            parts: list[str] = []
            async with aclosing(stream):
                async for chunk in stream:
                    if chunk.text:
                        parts.append(chunk.text)
            raw = "".join(parts)
            try:
                return model_type.model_validate_json(_strip_fence(raw))
            except ValidationError as exc:
                errors = _describe(exc)
            conversation += [
                Message(role=Role.ASSISTANT, content=raw),
                Message(role=Role.USER, content=_correction(errors, native=native)),
            ]
        raise StructuredOutputError(
            f"role {role!r} did not return a valid {model_type.__name__} after "
            f"{retries + 1} attempt(s):\n{errors}",
            raw=raw,
            errors=errors,
        )
