"""Model residency: keeping the chat model's memory on its server bounded.

Why: Ollama's MLX runner keeps a prompt cache that grows with every request (by
0.7 to 1.0 GB for `gemma4:12b-mlx`) up to a fixed ceiling of about 16 GB that
no setting changes. Unloading the model frees it. So when Lucy has been idle
for a moment after model work, `Residency` unloads the model, loads it again at
once and pre-fills the next turn's prompt prefix, so the next turn still starts
warm (docs/voice.md, "Keeping the MLX build with bounded memory").

It talks to the model only through the optional `ResidentLLM` adapter methods
(declared by `LLMCaps.residency`); adapters without them take no part. Roles
whose adapters share a `residency_key` (one model on one server) share one
state: unloading is per model, not per role. Models outside the gateway (the
embedder, the speech models) are never touched.

Busy: Lucy is busy, and nothing is unloaded, while any of these holds:

- a turn is in flight: from its `UserMessage` until its `TurnEnded` (seen on
  the bus); this covers the gaps between a tool call and the next model step,
  when no stream is open;
- a voice client reports the user speaking or an utterance being transcribed
  (`SpeechActivity` on the bus);
- the gateway has a FOREGROUND stream, or a BACKGROUND stream running or
  waiting;
- the job runner is running a job, or a job is due (`JobActivity`);
- a warm-up of its own is in flight.

Policy: once Lucy has not been busy for `idle_seconds`, and some model request
has been made through the gateway since the last unload (warm-ups do not
count), and, when no voice session is active, `text_quiet_seconds` have passed
since the last turn ended, and the model takes at least `min_size_bytes` (when
set): unload every model used since the last unload, then re-warm the chat
model. In voice the unload and re-warm fit inside the playback of Lucy's reply;
in text chat a quick reply would otherwise arrive before the re-warm finished
and start cold, hence the longer quiet period there. That period adapts: when a
message arrives during a text-mode cycle, the user evidently replies at about
that pace, so the quiet period doubles (up to `TEXT_QUIET_LIMIT` times the
setting); a text-mode cycle that finishes undisturbed sets it back. Measured
with the 10 s default and replies 12 s apart, every turn otherwise collided
with a re-warm and took about 3.5 s to its first word instead of 0.6 s.

Re-warm: the next turn's prefix (`PromptPrefix`, from the session most recently
active) is sent twice with a one-token limit, ending in an empty user message
and then in a one-character one. The server's cache keeps a resume point only
where a request branches from a cached path, so a single request would leave
nothing the next turn could reuse; the second request branches exactly where
the next turn's user text will start. With no prefix (no session yet, or a
layout without a stable prefix) the model is only loaded. No further step
starts once a turn has begun, and the second request is skipped once speech
has begun.

A turn during a warm-up: the gateway awaits `before_foreground` before every
FOREGROUND stream. No further step starts; a first pre-fill in flight is
cancelled (it alone leaves nothing the turn can reuse, and the turn has to
process the same prompt anyway); any other step in flight (the unload, a load,
the second pre-fill, after which the turn finds its prefix cached) is awaited,
for at most `max_turn_wait` seconds. Speech starting during a warm-up skips the
second pre-fill, and loads the model if it was unloaded and not yet reloaded.

Voice: when the user starts speaking and the model is not known to be loaded,
it is loaded at once (without generating), so the turn finds it in memory.

Failures: any error in a step is logged and ends that cycle; the next real
request simply loads the model as it would without this policy. Warm-ups never
go through the gateway, so they are neither turns nor background jobs: they
pre-empt nothing, are never pre-empted, take no fallback, and leave no trace or
episode.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol

from lucy.core.bus import EventBus
from lucy.core.events import SpeechActivity, TurnEnded, UserMessage
from lucy.core.gateway import LLMGateway
from lucy.core.ports import LLM, ResidentLLM
from lucy.core.types import Message, Prefill, PromptPrefix, Role

logger = logging.getLogger(__name__)

PrefixSource = Callable[[str], Awaitable[PromptPrefix | None]]
"""Returns the next turn's prompt prefix for a session (`Orchestrator.next_turn_prefix`)."""

Step = Literal["idle", "measure", "unload", "load", "prefill_1", "prefill_2", "done"]

TEXT_QUIET_LIMIT: Final = 4
"""The adaptive text quiet period grows to at most this many times the setting."""

LOADED_FOR: Final = 240.0
"""Seconds after its last use that a model is still assumed loaded (Ollama's default
keep-alive is 5 minutes); later, speech starting loads it again to be sure."""


class JobActivity(Protocol):
    """What the policy needs to know about background jobs (`JobRunner` has it)."""

    @property
    def running(self) -> bool: ...

    async def has_due(self) -> bool: ...


def resident(llm: LLM) -> ResidentLLM | None:
    """`llm` if it declares and implements residency, else None."""
    if getattr(llm.capabilities, "residency", False) and isinstance(llm, ResidentLLM):
        return llm
    return None


@dataclass(eq=False)
class _Model:
    """One model on one server, shared by the roles whose adapters have its key."""

    key: str
    llm: ResidentLLM
    used: bool = False
    """A model request went through the gateway since the last unload."""
    state: Literal["unknown", "loaded", "unloaded"] = "unknown"
    last_used: float = float("-inf")


@dataclass(eq=False)
class _Cycle:
    """An unload and re-warm, or a load at the start of speech, in progress."""

    reason: str
    task: asyncio.Task[None] | None = None
    step: Step = "idle"
    stop: bool = False
    """A turn has begun: start no further step."""
    speech: bool = False
    """Speech has begun: skip the second pre-fill."""


@dataclass
class CycleRecord:
    """What one cycle did, for the log and for tests."""

    reason: str
    started: float = field(default_factory=time.time)
    """Wall-clock time (seconds since the epoch) the cycle started."""
    unloaded: list[str] = field(default_factory=list)
    size_before: int | None = None
    size_after: int | None = None
    unload_seconds: float | None = None
    warm_steps: list[str] = field(default_factory=list)
    prefill: list[Prefill] = field(default_factory=list)
    warm_seconds: float | None = None
    outcome: str = "done"


class Residency:
    """The unload and re-warm policy (see the module docstring).

    Construct it after the gateway and before turns can start (it subscribes to
    the bus at once), run `run` as a task, and `aclose` it before closing the
    adapters. `tick` and `handle` are the policy's steps, public for tests.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        gateway: LLMGateway,
        chat_role: str = "chat",
        jobs: JobActivity | None = None,
        prefix: PrefixSource | None = None,
        idle_seconds: float = 1.0,
        text_quiet_seconds: float = 10.0,
        min_size_bytes: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        poll_interval: float = 0.25,
        step_timeout: float = 60.0,
        max_turn_wait: float = 15.0,
    ) -> None:
        chat = resident(gateway.for_role(chat_role))
        if chat is None:
            raise ValueError(f"the {chat_role!r} role's adapter does not support residency")
        self._gateway = gateway
        self._chat_role = chat_role
        self._jobs = jobs
        self._prefix = prefix
        self._idle_seconds = idle_seconds
        self._text_quiet = text_quiet_seconds
        self._text_quiet_now = text_quiet_seconds
        self._min_size = min_size_bytes
        self._clock = clock
        self._poll = poll_interval
        self._step_timeout = step_timeout
        self._max_turn_wait = max_turn_wait
        self._models: dict[str, _Model] = {}
        for role in gateway.roles():
            llm = resident(gateway.for_role(role))
            if llm is not None and llm.residency_key not in self._models:
                self._models[llm.residency_key] = _Model(llm.residency_key, llm)
        self._chat = self._models[chat.residency_key]
        self._turns: dict[str, int] = {}
        self._speech: dict[str, str] = {}
        self._last_session: str | None = None
        self._last_turn_end = clock()
        self._idle_since: float | None = None
        self._cycle: _Cycle | None = None
        self._closed = False
        self.records: deque[CycleRecord] = deque(maxlen=100)
        """The latest cycles, oldest first."""
        self._subscription = bus.subscribe(UserMessage, TurnEnded, SpeechActivity)
        gateway.on_request(self._on_request)
        gateway.set_foreground_gate(self.before_foreground)

    @staticmethod
    def supported(gateway: LLMGateway, chat_role: str = "chat") -> bool:
        """Whether the chat role's adapter can take part."""
        return resident(gateway.for_role(chat_role)) is not None

    @property
    def models(self) -> list[str]:
        """The residency keys of the models managed."""
        return list(self._models)

    # --- Signals --------------------------------------------------------------

    def handle(self, event: object) -> None:
        """Take note of a bus event (turn start and end, speech activity)."""
        if isinstance(event, UserMessage):
            self._turns[event.session_id] = self._turns.get(event.session_id, 0) + 1
            self._last_session = event.session_id
            if self._cycle is not None:
                self._cycle.stop = True
        elif isinstance(event, TurnEnded):
            count = self._turns.get(event.session_id, 0) - 1
            if count > 0:
                self._turns[event.session_id] = count
            else:
                self._turns.pop(event.session_id, None)
            self._last_session = event.session_id
            self._last_turn_end = self._clock()
        elif isinstance(event, SpeechActivity):
            if event.state == "ended":
                self._speech.pop(event.session_id, None)
            else:
                self._speech[event.session_id] = event.state
            if event.state == "speaking":
                self._speech_started()

    def _on_request(self, role: str, llm: LLM) -> None:
        model = self._models.get(getattr(llm, "residency_key", ""))
        if model is not None:
            model.used = True
            model.state = "loaded"
            model.last_used = self._clock()

    @property
    def text_quiet_seconds(self) -> float:
        """The text quiet period in force now (it adapts; see the module docstring)."""
        return self._text_quiet_now

    @property
    def voice_active(self) -> bool:
        """Whether a voice session is running (it shortens the idle rule)."""
        return bool(self._speech)

    async def busy_reason(self) -> str | None:
        """Why Lucy is busy now, or None when she is idle."""
        if any(self._turns.values()):
            return "turn"
        if any(state in ("speaking", "transcribing") for state in self._speech.values()):
            return "speech"
        gateway = self._gateway
        if gateway.foreground_active:
            return "foreground stream"
        if gateway.background_running or gateway.background_waiting:
            return "background stream"
        if self._cycle is not None:
            return "warm-up"
        jobs = self._jobs
        if jobs is not None:
            if jobs.running:
                return "job"
            try:
                if await jobs.has_due():
                    return "job due"
            except Exception as exc:
                logger.debug("residency: cannot ask the job store (%s); treating as busy", exc)
                return "job store unavailable"
        return None

    # --- Policy -----------------------------------------------------------------

    async def tick(self) -> None:
        """One check: start an unload and re-warm cycle if the policy says so."""
        if self._closed:
            return
        now = self._clock()
        if await self.busy_reason() is not None:
            self._idle_since = None
            return
        if self._idle_since is None:
            self._idle_since = now
        if now - self._idle_since < self._idle_seconds:
            return
        if not self.voice_active and now - self._last_turn_end < self._text_quiet_now:
            return
        used = [m for m in self._models.values() if m.used]
        if not used:
            return
        self._start(_Cycle(reason="voice idle" if self.voice_active else "idle"), used)

    def _speech_started(self) -> None:
        cycle = self._cycle
        if cycle is not None:
            cycle.speech = True
            return
        chat = self._chat
        loaded = chat.state == "loaded" and self._clock() - chat.last_used < LOADED_FOR
        if not loaded and not self._closed:
            self._start(_Cycle(reason="speech"), [])

    def _start(self, cycle: _Cycle, unload: list[_Model]) -> None:
        self._cycle = cycle
        cycle.task = asyncio.create_task(self._run_cycle(cycle, unload), name="lucy-residency")

    async def before_foreground(self) -> None:
        """Let a FOREGROUND stream go ahead of a warm-up (see the module docstring)."""
        cycle = self._cycle
        if cycle is None or cycle.task is None or cycle.task.done():
            return
        cycle.stop = True
        if cycle.step == "prefill_1":
            cycle.task.cancel()
        done, _ = await asyncio.wait({cycle.task}, timeout=self._max_turn_wait)
        if not done:
            logger.warning(
                "residency: a warm-up step (%s) took over %gs; cancelling it for a turn",
                cycle.step,
                self._max_turn_wait,
            )
            cycle.task.cancel()
            await asyncio.wait({cycle.task}, timeout=1.0)

    # --- One cycle ----------------------------------------------------------------

    async def _step[T](self, cycle: _Cycle, step: Step, call: Awaitable[T]) -> T:
        cycle.step = step
        async with asyncio.timeout(self._step_timeout):
            return await call

    async def _run_cycle(self, cycle: _Cycle, unload: list[_Model]) -> None:
        record = CycleRecord(reason=cycle.reason)
        try:
            if unload and not await self._unload(cycle, unload, record):
                return
            if cycle.stop:
                record.outcome = "stopped for a turn"
                return
            await self._warm(cycle, record)
        except asyncio.CancelledError:
            record.outcome = f"cancelled during {cycle.step}"
            if cycle.step in ("load", "prefill_1", "prefill_2"):
                self._chat.state = "unknown"
            raise
        except Exception as exc:
            record.outcome = f"failed during {cycle.step}: {type(exc).__name__}: {exc}"
            logger.warning("residency: %s failed (%s); carrying on", cycle.step, _describe(exc))
            if cycle.step in ("load", "prefill_1", "prefill_2"):
                self._chat.state = "unknown"
        finally:
            cycle.step = "done"
            if self._cycle is cycle:
                self._cycle = None
            if cycle.reason == "idle" and not self._closed:
                self._adapt_text_quiet(interrupted=cycle.stop)
            self._idle_since = None
            self.records.append(record)
            self._log(record)

    def _adapt_text_quiet(self, *, interrupted: bool) -> None:
        if interrupted:
            limit = self._text_quiet * TEXT_QUIET_LIMIT
            self._text_quiet_now = min(max(self._text_quiet_now * 2, 1.0), limit)
            logger.info(
                "residency: a message arrived during the re-warm; text quiet period now %gs",
                self._text_quiet_now,
            )
        else:
            self._text_quiet_now = self._text_quiet

    async def _unload(self, cycle: _Cycle, models: Sequence[_Model], record: CycleRecord) -> bool:
        """Unload `models` (the chat model first); False if the cycle should end here."""
        ordered = sorted(models, key=lambda m: m is not self._chat)
        for model in ordered:
            model.used = False  # a failure must not retry on every tick; new use re-arms
            size = await self._step(cycle, "measure", model.llm.loaded_size())
            if model is self._chat:
                record.size_before = size
            if size is None:
                model.state = "unloaded"
                continue
            if self._min_size is not None and size < self._min_size:
                model.state = "loaded"
                continue
            if cycle.stop:
                break
            started = self._clock()
            await self._step(cycle, "unload", model.llm.unload())
            model.state = "unloaded"
            record.unloaded.append(model.key)
            record.unload_seconds = round(self._clock() - started, 3)
        if self._chat.key in record.unloaded:
            return True
        if cycle.stop:
            record.outcome = "stopped for a turn"
        elif record.unloaded:
            record.outcome = "nothing to re-warm"
        elif record.size_before is None:
            record.outcome = "not loaded"
        else:
            record.outcome = "below the size limit"
        return False

    async def _warm(self, cycle: _Cycle, record: CycleRecord) -> None:
        started = self._clock()
        prefix = await self._next_prefix() if cycle.reason != "speech" else None
        llm = self._chat.llm
        if prefix is not None:
            role_llm = resident(self._gateway.for_role(prefix.role))
            if role_llm is not None and role_llm.residency_key == self._chat.key:
                llm = role_llm
            else:
                prefix = None
        if prefix is None or cycle.speech:
            await self._step(cycle, "load", llm.load())
            record.warm_steps.append("load")
        else:
            for step, text in (("prefill_1", ""), ("prefill_2", "x")):
                if cycle.stop or (step == "prefill_2" and cycle.speech):
                    record.warm_steps.append(f"skipped {step}")
                    break
                messages = [*prefix.messages, Message(role=Role.USER, content=text)]
                usage = await self._step(cycle, step, llm.prefill(messages, prefix.tools))
                record.prefill.append(usage)
                record.warm_steps.append(step)
        record.warm_seconds = round(self._clock() - started, 3)
        if not any(step in ("load", "prefill_1") for step in record.warm_steps):
            record.outcome = "stopped for a turn"
            return
        self._chat.state = "loaded"
        self._chat.last_used = self._clock()
        if not cycle.stop:
            with suppress(Exception):
                record.size_after = await self._step(cycle, "measure", llm.loaded_size())

    async def _next_prefix(self) -> PromptPrefix | None:
        if self._prefix is None or self._last_session is None:
            return None
        try:
            async with asyncio.timeout(self._step_timeout):
                return await self._prefix(self._last_session)
        except Exception as exc:
            logger.warning("residency: cannot build the next prompt prefix (%s)", _describe(exc))
            return None

    def _log(self, record: CycleRecord) -> None:
        if record.outcome in ("below the size limit", "not loaded"):
            logger.debug("residency: %s, nothing unloaded", record.outcome)
            return
        parts = []
        if record.unloaded:
            parts.append(
                f"unloaded {', '.join(k.split('|')[-1] for k in record.unloaded)}"
                f" at {_gb(record.size_before)} in {record.unload_seconds or 0:.2f}s"
            )
        if record.warm_steps:
            tokens = [p.prompt_tokens for p in record.prefill if p.prompt_tokens is not None]
            what = f"a {max(tokens)}-token prefix" if tokens else "the model"
            took = "" if record.warm_seconds is None else f" in {record.warm_seconds:.2f}s"
            parts.append(
                f"re-warmed {what} ({', '.join(record.warm_steps)}){took},"
                f" now {_gb(record.size_after)}"
            )
        if record.outcome != "done":
            parts.append(record.outcome)
        logger.info("residency (%s): %s", record.reason, "; ".join(parts) or record.outcome)

    # --- Lifecycle ----------------------------------------------------------------

    async def run(self) -> None:
        """Follow the bus and check the policy every `poll_interval` seconds, until cancelled."""
        events = asyncio.create_task(self._follow(), name="lucy-residency-events")
        try:
            while True:
                await asyncio.sleep(self._poll)
                try:
                    await self.tick()
                except Exception:
                    logger.exception("residency check failed")
        finally:
            events.cancel()
            await asyncio.gather(events, return_exceptions=True)

    async def _follow(self) -> None:
        async for event in self._subscription:
            self.handle(event)

    async def aclose(self) -> None:
        """Stop for good: cancel a warm-up in flight and detach from the gateway and bus."""
        self._closed = True
        self._gateway.set_foreground_gate(None)
        self._subscription.close()
        cycle = self._cycle
        if cycle is not None and cycle.task is not None and not cycle.task.done():
            cycle.task.cancel()
            await asyncio.gather(cycle.task, return_exceptions=True)


def _gb(size: int | None) -> str:
    return "?" if size is None else f"{size / 1e9:.1f} GB"


def _describe(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timed out"
    return f"{type(exc).__name__}: {exc}"
