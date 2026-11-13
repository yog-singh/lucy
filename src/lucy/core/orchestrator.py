"""The agent loop: turns inbound events into turns (ADR 0003, 0004, 0006).

The orchestrator keeps no conversation state between turns. Each turn rebuilds
the conversation from the episode log, asks memory for context, runs the chat
LLM with tools for at most `max_steps` steps, and publishes and persists what
happened. Turns of one session run one at a time in arrival order; different
sessions run concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from lucy.core.bus import EventBus
from lucy.core.confirm import outcome_for, pending_confirmation
from lucy.core.context import ContextBuilder, SectionReport
from lucy.core.events import (
    AnyEvent,
    AssistantMessage,
    ConfirmationOutcome,
    ConfirmationRequested,
    ConfirmationResolved,
    Event,
    Interrupt,
    PlaybackInterrupted,
    SentenceReady,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UserMessage,
)
from lucy.core.gateway import LLMGateway, Priority
from lucy.core.history import events_to_messages, tool_message
from lucy.core.ports import ConfirmableTool, MemoryStore, Tool, TraceSink, tool_spec
from lucy.core.sentences import SentenceSplitter
from lucy.core.trace import StepTrace, TurnTrace
from lucy.core.types import (
    FinishReason,
    Message,
    PromptPrefix,
    Risk,
    Role,
    ToolCall,
    ToolOutput,
    ToolResult,
)

logger = logging.getLogger(__name__)

Outcome = Literal["completed", "interrupted", "error"]

NOT_RUN: dict[ConfirmationOutcome, str] = {
    "declined": "Not run: the user said no.",
    "superseded": (
        "Not run: the user did not confirm and said something else; the request is "
        "cancelled. If they still want it, call the tool again and they will be asked again."
    ),
    "expired": "Not run: the confirmation expired.",
    "unheard": "Not run: the question was cut off before the user heard it.",
    "changed": "Not run: it changed since the user was asked.",
}
"""The tool result given to the model when a confirmation ends without a run."""


def _now() -> datetime:
    return datetime.now(UTC)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


@dataclass
class _Session:
    """Per-session scheduling state. Exists only while the session has work."""

    pending: deque[UserMessage] = field(default_factory=deque)
    worker: asyncio.Task[None] | None = None
    turn: asyncio.Task[None] | None = None
    turn_started: bool = False


@dataclass
class _Step:
    """The LLM step in progress, for the interrupted record and the trace."""

    messages: list[Message]
    tools: list[str]
    started_at: datetime = field(default_factory=_now)
    first_token_at: datetime | None = None
    finish_reason: FinishReason | None = None
    text: str = ""

    def trace(self, role: str) -> StepTrace:
        return StepTrace(
            role=role,
            messages=self.messages,
            tools=self.tools,
            started_at=self.started_at,
            first_token_at=self.first_token_at,
            ended_at=_now(),
            finish_reason=self.finish_reason,
        )


@dataclass
class _Turn:
    session_id: str
    turn_id: str
    started_at: datetime = field(default_factory=_now)
    steps: list[StepTrace] = field(default_factory=list)
    context: list[SectionReport] = field(default_factory=list)
    step: _Step | None = None  # the step currently streaming, if any


class Orchestrator:
    """Consumes `UserMessage` and `Interrupt` events from the bus and runs turns.

    It also appends every `PlaybackInterrupted` it receives (published by a
    voice client, ADR 0016) to the episode log, and does nothing else with it:
    the event only changes how that turn is rebuilt from the log later.

    Readiness: the bus subscription is created in `__init__`, not in `run`, so
    every inbound event published after the orchestrator is constructed is
    queued and handled once `run` is scheduled. Callers construct it, start
    `run` as a task, and may publish immediately.

    Shutdown: when the bus is closed, `run` handles the events already queued,
    then cancels running and queued turns (a cancelled running turn is recorded
    as interrupted, persisted, and published only if the bus is still open) and
    returns. Cancelling `run` does the same and then re-raises `CancelledError`.

    An `Interrupt` cancels the session's turn only once that turn has started
    running; an `Interrupt` that arrives while no turn of the session is running
    is ignored, and queued turns still run afterwards.

    Dangerous tools (ADR 0006) are never run on the model's say-so. A dangerous
    call is turned into a persisted `ConfirmationRequested` holding the exact
    call, and the tool's preview is put to the user as the turn's reply, by code.
    The session's very next typed or spoken message answers it, within
    `confirm_timeout` seconds: before any LLM call of that turn, plain code
    (`lucy.core.confirm`) decides, logs a `ConfirmationResolved`, and on a yes
    runs the call from the log. The model then sees the outcome as the result of
    a tool call and writes the reply.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        gateway: LLMGateway,
        memory: MemoryStore,
        context: ContextBuilder,
        tools: Sequence[Tool] = (),
        trace: TraceSink | None = None,
        chat_role: str = "chat",
        max_steps: int = 8,
        history_limit: int = 200,
        confirm_timeout: float = 60.0,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if confirm_timeout <= 0:
            raise ValueError("confirm_timeout must be positive")
        self._bus = bus
        self._gateway = gateway
        self._memory = memory
        self._context = context
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            if tool.risk == Risk.DANGEROUS and not callable(getattr(tool, "preview", None)):
                raise ValueError(f"dangerous tool {tool.name!r} has no preview to confirm with")
            self._tools[tool.name] = tool
        self._tool_specs = [tool_spec(tool) for tool in tools]
        self._trace = trace
        self._chat_role = chat_role
        self._max_steps = max_steps
        self._history_limit = history_limit
        self._confirm_timeout = timedelta(seconds=confirm_timeout)
        self._sessions: dict[str, _Session] = {}
        self._running = False
        self._subscription = bus.subscribe(UserMessage, Interrupt, PlaybackInterrupted)

    async def run(self) -> None:
        """Handle inbound events until the bus is closed, then stop all turns and return."""
        if self._running:
            raise RuntimeError("Orchestrator.run is already running")
        self._running = True
        try:
            async for event in self._subscription:
                if isinstance(event, PlaybackInterrupted):
                    await self._record_playback(event)
                else:
                    self._dispatch(event)
        finally:
            self._subscription.close()
            await self._shutdown()

    async def next_turn_prefix(self, session_id: str) -> PromptPrefix | None:
        """How the next turn of `session_id` will begin its first model request: the
        messages before the new user message, the role and the tools offered.

        None when the context layout has no prefix that stays the same from turn
        to turn (`ContextBuilder.prefix_messages`) or the session has no history.
        The conversation is rebuilt as the next turn will rebuild it: the same
        window of stored episodes plus a placeholder for the coming user message
        (so the last turn's tool results are already shortened, as they will be
        then), with the current profile and standing instructions. Nothing is
        stored or published. Used to pre-fill the model server's cache
        (`lucy.core.residency`).
        """
        if self._context.layout != "stable_prefix":
            return None
        episodes = await self._memory.recent_episodes(session_id, self._history_limit)
        if not episodes:
            return None
        placeholder = UserMessage(session_id=session_id, text="")
        history = events_to_messages([*episodes, placeholder][-self._history_limit :])
        window = self._gateway.for_role(self._chat_role).capabilities.context_window
        budget = self._context.token_budgets(window)["memories"]
        query = next((e.text for e in reversed(episodes) if isinstance(e, UserMessage)), "")
        memory = await self._memory.retrieve(query=query, budget=budget)
        built = self._context.build(memory=memory, history=history, context_window=window)
        return PromptPrefix(
            role=self._chat_role,
            messages=self._context.prefix_messages(built),
            tools=list(self._tool_specs),
        )

    # --- Scheduling -----------------------------------------------------------

    def _dispatch(self, event: Event) -> None:
        if isinstance(event, UserMessage):
            session = self._sessions.setdefault(event.session_id, _Session())
            session.pending.append(event)
            if session.worker is None:
                session.worker = asyncio.create_task(self._serve(event.session_id, session))
        elif isinstance(event, Interrupt):
            session = self._sessions.get(event.session_id)
            if session is not None and session.turn is not None and session.turn_started:
                session.turn.cancel()

    async def _record_playback(self, event: PlaybackInterrupted) -> None:
        """Persist what a voice client reports was actually spoken; failures are logged."""
        try:
            await self._memory.append_episode(event)
        except Exception:
            logger.exception("could not record playback interruption of turn %s", event.turn_id)

    async def _serve(self, session_id: str, session: _Session) -> None:
        """Run the session's queued turns one at a time, then retire the session."""
        while session.pending:
            message = session.pending.popleft()
            session.turn_started = False
            session.turn = asyncio.create_task(self._run_turn(session, message))
            # wait() rather than await: an interrupted turn must not stop this loop.
            await asyncio.wait({session.turn})
            session.turn = None
        # No await since the loop condition was checked, so nothing was queued meanwhile.
        del self._sessions[session_id]

    async def _shutdown(self) -> None:
        tasks: list[asyncio.Task[None]] = []
        for session in self._sessions.values():
            session.pending.clear()
            for task in (session.worker, session.turn):
                if task is not None:
                    task.cancel()
                    tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        self._sessions.clear()

    # --- One turn -------------------------------------------------------------

    async def _run_turn(self, session: _Session, message: UserMessage) -> None:
        session.turn_started = True
        turn = _Turn(session_id=message.session_id, turn_id=str(uuid.uuid4()))
        outcome: Outcome = "error"
        error: str | None = None
        try:
            outcome, error = await self._turn_body(turn, message)
        except asyncio.CancelledError:
            outcome = "interrupted"
            await self._end_interrupted(turn)
            raise
        except Exception as exc:
            logger.exception("turn %s failed", turn.turn_id)
            outcome, error = "error", _describe(exc)
            self._close_step(turn)
            await self._end_safely(turn, "error", error)
        finally:
            await self._record_trace(turn, outcome, error)

    async def _turn_body(self, turn: _Turn, message: UserMessage) -> tuple[Outcome, str | None]:
        await self._memory.append_episode(message.model_copy(update={"turn_id": turn.turn_id}))

        episodes = await self._memory.recent_episodes(turn.session_id, self._history_limit)
        # Decided before any LLM call, from the log alone (ADR 0006).
        request = pending_confirmation(episodes, message)
        answered = await self._resolve(turn, request, episodes, message) if request else []
        history = events_to_messages(episodes)
        window = self._gateway.for_role(self._chat_role).capabilities.context_window
        budget = self._context.token_budgets(window)["memories"]
        memory = await self._memory.retrieve(query=message.text, budget=budget)
        built = self._context.build(memory=memory, history=history, context_window=window)
        turn.context = built.report
        messages = [*built.messages, *answered]

        for step_number in range(1, self._max_steps + 1):
            text, calls = await self._run_step(turn, messages)
            if not calls:
                await self._emit(TurnEnded(**self._ids(turn), reason="completed"))
                return "completed", None
            if step_number == self._max_steps:
                # The results could never reach the model, so the calls are not run;
                # rebuilding history drops them as unanswered.
                break
            messages.append(Message(role=Role.ASSISTANT, content=text, tool_calls=calls))
            question: str | None = None
            waiting_for = ""
            for call in calls:
                tool = self._tools.get(call.name)
                if question is not None:
                    content = f"Not run: waiting for the user's confirmation of {waiting_for}."
                    result = ToolResult(call_id=call.id, content=content, is_error=True)
                elif tool is not None and tool.risk == Risk.DANGEROUS:
                    result, question = await self._request_confirmation(turn, call, tool)
                    waiting_for = call.name
                else:
                    await self._emit(ToolStarted(**self._ids(turn), call=call))
                    result = await self._run_tool(call)
                await self._emit(ToolFinished(**self._ids(turn), result=result))
                messages.append(tool_message(result))
            if question is not None:
                # The question is the reply; the model does not get to word it or act on.
                await self._say(turn, question)
                await self._emit(TurnEnded(**self._ids(turn), reason="completed"))
                return "completed", None

        error = f"step limit of {self._max_steps} LLM calls reached without a final answer"
        await self._emit(TurnEnded(**self._ids(turn), reason="error", error=error))
        return "error", error

    async def _run_step(self, turn: _Turn, messages: list[Message]) -> tuple[str, list[ToolCall]]:
        """Stream one LLM step, publishing text as it arrives; commit its AssistantMessage."""
        step = _Step(messages=list(messages), tools=[spec.name for spec in self._tool_specs])
        turn.step = step
        splitter = SentenceSplitter()
        calls: list[ToolCall] = []
        stream = self._gateway.stream(
            self._chat_role, step.messages, self._tool_specs, priority=Priority.FOREGROUND
        )
        async with aclosing(stream) as chunks:
            async for chunk in chunks:
                if step.first_token_at is None and (chunk.text or chunk.tool_call):
                    step.first_token_at = _now()
                if chunk.text:
                    step.text += chunk.text
                    await self._publish(TextDelta(**self._ids(turn), text=chunk.text))
                    for sentence in splitter.feed(chunk.text):
                        await self._publish(SentenceReady(**self._ids(turn), text=sentence))
                if chunk.tool_call is not None:
                    calls.append(chunk.tool_call)
                if chunk.finish_reason is not None:
                    step.finish_reason = chunk.finish_reason
        for sentence in splitter.flush():
            await self._publish(SentenceReady(**self._ids(turn), text=sentence))
        # The step's text is committed from here on; an interrupt no longer records it.
        turn.step = None
        turn.steps.append(step.trace(self._chat_role))
        await self._emit(AssistantMessage(**self._ids(turn), text=step.text, tool_calls=calls))
        return step.text, calls

    async def _run_tool(self, call: ToolCall) -> ToolResult:
        """Run one tool call; failures become error results.

        The caller enforces the risk tier: a dangerous call reaches this only
        once the user has confirmed it (`_resolve`).
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.id, content=f"Unknown tool {call.name!r}.", is_error=True
            )
        try:
            output = await tool.run(dict(call.arguments))
        except Exception as exc:
            logger.warning("tool %s failed: %s", call.name, _describe(exc))
            content = f"Tool {call.name!r} failed: {_describe(exc)}"
            return ToolResult(call_id=call.id, content=content, is_error=True)
        return ToolResult(call_id=call.id, content=output.content, is_error=output.is_error)

    # --- Confirmation (ADR 0006) ------------------------------------------------

    async def _preview(self, call: ToolCall, tool: Tool) -> ToolOutput:
        """The tool's preview of `call`; a failure becomes an error output."""
        try:
            return await cast(ConfirmableTool, tool).preview(dict(call.arguments))
        except Exception as exc:
            logger.warning("preview of %s failed: %s", call.name, _describe(exc))
            return ToolOutput(content=f"Tool {call.name!r} failed: {_describe(exc)}", is_error=True)

    async def _request_confirmation(
        self, turn: _Turn, call: ToolCall, tool: Tool
    ) -> tuple[ToolResult, str | None]:
        """Log a request to confirm `call`. Returns the call's result for the model and
        the question to put to the user (None if the call cannot be previewed)."""
        preview = await self._preview(call, tool)
        if preview.is_error:
            return ToolResult(call_id=call.id, content=preview.content, is_error=True), None
        question = preview.content
        await self._emit(
            ConfirmationRequested(
                **self._ids(turn),
                call=call,
                question=question,
                expires_at=_now() + self._confirm_timeout,
            )
        )
        content = f"Not run yet. The user is being asked: {question}"
        return ToolResult(call_id=call.id, content=content), question

    async def _say(self, turn: _Turn, text: str) -> None:
        """Deliver code-authored `text` exactly as a model reply is delivered."""
        await self._publish(TextDelta(**self._ids(turn), text=text))
        splitter = SentenceSplitter()
        for sentence in [*splitter.feed(text), *splitter.flush()]:
            await self._publish(SentenceReady(**self._ids(turn), text=sentence))
        await self._emit(AssistantMessage(**self._ids(turn), text=text))

    async def _resolve(
        self,
        turn: _Turn,
        request: ConfirmationRequested,
        episodes: Sequence[AnyEvent],
        message: UserMessage,
    ) -> list[Message]:
        """Answer the pending `request` with `message`; on a yes, run the logged call.

        Returns the messages to add after the user's message: a code-authored
        assistant tool call (the logged call under a fresh id) and its result.
        """
        outcome = outcome_for(request, episodes, message)
        tool = self._tools.get(request.call.name)
        if outcome == "confirmed":
            # The target must still be the one the user was asked about.
            if tool is None:
                outcome = "changed"
            else:
                again = await self._preview(request.call, tool)
                if again.is_error or again.content != request.question:
                    outcome = "changed"
        # Logged before anything runs: a crash can lose the run but never repeat it.
        await self._emit(
            ConfirmationResolved(**self._ids(turn), request_id=request.id, outcome=outcome)
        )
        call = request.call.model_copy(update={"id": str(uuid.uuid4())})
        await self._emit(AssistantMessage(**self._ids(turn), tool_calls=[call]))
        if outcome == "confirmed":
            await self._emit(ToolStarted(**self._ids(turn), call=call))
            result = await self._run_tool(call)
        else:
            result = ToolResult(call_id=call.id, content=NOT_RUN[outcome], is_error=True)
        await self._emit(ToolFinished(**self._ids(turn), result=result))
        return [Message(role=Role.ASSISTANT, content="", tool_calls=[call]), tool_message(result)]

    # --- Endings --------------------------------------------------------------

    def _close_step(self, turn: _Turn) -> _Step | None:
        """Move an unfinished step into the trace and return it."""
        step, turn.step = turn.step, None
        if step is not None:
            turn.steps.append(step.trace(self._chat_role))
        return step

    async def _end_interrupted(self, turn: _Turn) -> None:
        step = self._close_step(turn)
        try:
            if step is not None and step.text:
                await self._emit(
                    AssistantMessage(**self._ids(turn), text=step.text, interrupted=True)
                )
            await self._emit(TurnEnded(**self._ids(turn), reason="interrupted"))
        except Exception:
            logger.exception("could not record the interruption of turn %s", turn.turn_id)

    async def _end_safely(self, turn: _Turn, reason: Outcome, error: str | None) -> None:
        try:
            await self._emit(TurnEnded(**self._ids(turn), reason=reason, error=error))
        except Exception:
            logger.exception("could not record the end of turn %s", turn.turn_id)

    async def _record_trace(self, turn: _Turn, outcome: Outcome, error: str | None) -> None:
        if self._trace is None:
            return
        trace = TurnTrace(
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            started_at=turn.started_at,
            ended_at=_now(),
            steps=turn.steps,
            context=turn.context,
            outcome=outcome,
            error=error,
        )
        try:
            await self._trace.record(trace)
        except Exception:
            logger.exception("trace sink failed for turn %s", turn.turn_id)

    # --- Output ---------------------------------------------------------------

    @staticmethod
    def _ids(turn: _Turn) -> dict[str, str]:
        return {"session_id": turn.session_id, "turn_id": turn.turn_id}

    async def _emit(self, event: AnyEvent) -> None:
        """Persist, then publish, an event that belongs in the episode log."""
        await self._memory.append_episode(event)
        await self._publish(event)

    async def _publish(self, event: Event) -> None:
        """Publish to live clients; skipped once the bus is closed (during shutdown)."""
        if not self._bus.closed:
            await self._bus.publish(event)
