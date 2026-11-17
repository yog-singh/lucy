"""The composition root: builds Lucy's runtime from settings and runs it.

This module sits outside `core/` and may import anything. It loads the built-in
adapter modules, creates the memory store and the LLMs through the registry,
and wires them into an `Orchestrator` on a fresh `EventBus`. Clients (the CLI,
later the server) talk to the running app only through `bus`.

Memory learning (ADR 0012) is wired in when an embedder is configured, the
memory adapter is the SQLite store, the settings have the `extraction` and
`summary` LLM roles, and `[learning] enabled` is true. Then the app also builds,
on the same database file but each on its own connection:

- a `SQLiteGraphStore` and a `Retriever`, attached to the memory store so every
  turn gets the profile, the standing instructions and retrieved statements;
- a `SQLiteJobStore` and a `JobRunner` with the memory job handlers (extract,
  decay, profile, entity summaries, duplicate merging, reflection,
  re-embedding). Extraction and reflection share one `Reconciler` (both use the
  `extraction` role); duplicate merging uses `summary`.

With `background` (the default) it also requests extraction after every
`TurnEnded`, schedules maintenance at startup and hourly, and runs the job
runner as a task. Otherwise (as for the `lucy memory` tools) the pieces are
built but nothing runs on its own.

Keeping the profile fresh: maintenance rewrites the profile once a day, which
alone left it stale for up to a day after the user changed something. So the
extraction handler is wrapped: when an extraction job finishes and no other
extraction job is waiting (the backlog has drained), a profile rewrite is
requested `profile_delay` seconds later, and a pending request is pushed back
to that time whenever another extraction finishes. A chat with many quick turns
therefore gets one rewrite, after the user has been quiet for `profile_delay`
(and `ProfileWriter` skips the model call when its input did not change). The
delay is 0 without `background` (`lucy memory work`, the eval), where nothing
else is waiting to run. A profile run that found nothing to write does not
count as the day's run, so the next maintenance check tries again.

Jobs that failed for good only because no handler was registered for their kind
(as `memory.reflect` did before reflection was wired in) are put back once at
startup when their kind now has a handler: the newest such job per kind is
requeued (unless one of that kind is already waiting) and the older ones deleted.

`clock` is "now" for everything that reasons about time (the graph store,
extraction, reconciliation, reflection, decay, the profile, retrieval and the
maintenance schedule), so tests and the eval can simulate weeks. The job queue
and runner always use the real clock: jobs scheduled at a simulated time in the
past are simply due.

Model residency (`lucy.core.residency`, `[residency]` in the settings): when the
chat role's adapter supports it and it is enabled, a `Residency` policy unloads
the chat model after model work once Lucy is idle and re-warms the next turn's
prompt prefix, which bounds the model server's memory. It runs as a task from
`start` to `stop`, with or without `background`.

Learning can never break or slow a chat turn: retrieval runs under a timeout
and falls back to the profile and instructions only, then to nothing; job
failures stay inside the runner; and a learning component that fails to start
only switches learning off (logged), leaving plain chat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType, TracebackType
from typing import Any, Final, Self

from lucy.adapters.embed_openai_compat import EmbedderError
from lucy.adapters.llm_openai_compat import LLMError
from lucy.core import registry as _registry
from lucy.core.bus import EventBus, Subscription
from lucy.core.config import DEFAULT_MEMORY, McpServerSettings, Settings
from lucy.core.context import ContextBuilder, estimate_tokens
from lucy.core.events import TurnEnded
from lucy.core.gateway import LLMGateway, Preempted, add_fallbacks
from lucy.core.jobs import Handler, Job, JobRunner, JobStatus, utc_now
from lucy.core.orchestrator import Orchestrator
from lucy.core.ports import (
    STT,
    TTS,
    Camera,
    Embedder,
    FaceRecognizer,
    MemoryStore,
    Tool,
    ToolProvider,
    TraceSink,
    tool_spec,
)
from lucy.core.registry import Registry, RegistryError
from lucy.core.reminders import REMINDER_TIMER, ReminderService, TimerHandler
from lucy.core.residency import Residency
from lucy.core.toolsets import META_TOOL, ToolGroup, ToolSelector, keyword_words
from lucy.core.trace import TurnTrace
from lucy.core.types import MemoryContext, Risk, ToolSpec, TTSCaps
from lucy.core.workspace import Workspace
from lucy.memory.decay import DecayPolicy
from lucy.memory.extraction import (
    DEFAULT_TRANSIENT_ERRORS,
    EXTRACT_JOB_KIND,
    Extractor,
    extraction_handler,
    request_extraction,
)
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.maintenance import (
    DECAY_JOB,
    MERGE_JOB,
    PROFILE_JOB,
    REEMBED_JOB,
    SUMMARIES_JOB,
    decay_handler,
    merge_handler,
    profile_handler,
    reembed_handler,
    request_profile,
    schedule_maintenance,
    summaries_handler,
)
from lucy.memory.profile import ProfileWriter
from lucy.memory.reconcile import Reconciler
from lucy.memory.reflection import REFLECT_JOB, Reflector, reflection_handler
from lucy.memory.retrieval import Retriever
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.memory.sqlite_reminders import SQLiteReminderStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from lucy.prompts import load_prompt
from lucy.tools import DEFAULT_CORE, GROUPS, build_tools
from lucy.tools.face_enrolment import VISION_TOOLS
from lucy.tools.reminders import REMINDER_TOOLS, build_store, local_announcement
from lucy.tools.voice_enrolment import ENROL_TOOLS, ENROLMENT_TIMER, handle_enrolment
from lucy.vision.faces import FaceBook
from lucy.vision.service import FaceRules, VisionService

logger = logging.getLogger(__name__)

BUILTIN_ADAPTER_MODULES: Final = (
    "lucy.adapters.llm_openai_compat",
    "lucy.adapters.embed_openai_compat",
    "lucy.memory.sqlite_store",
    "lucy.adapters.stt_parakeet_mlx",  # imports MLX only when a model is loaded
    "lucy.adapters.stt_indic_flex",  # its model runs in a worker process (ADR 0023)
    "lucy.adapters.tts_openai_compat",
    "lucy.adapters.tts_magpie",  # its model runs in a worker process (ADR 0024)
    "lucy.adapters.camera_opencv",  # imports OpenCV only when a frame is taken (ADR 0026)
    "lucy.adapters.faces_insightface",  # its worker starts on the first look with faces
)
"""Modules imported at startup so their adapters register themselves."""

LEARNING_ROLES: Final = ("extraction", "summary")
"""LLM roles the learning loop needs. Duplicate-merge checks use `summary`."""

RETRIEVAL_TIMEOUT: Final = 3.0
"""Seconds a turn waits for memory retrieval before answering without retrieved
statements (and again for the profile and instructions alone)."""

MAINTENANCE_CHECK_SECONDS: Final = 3600.0
"""How often a running app checks whether maintenance is due."""

PROFILE_DELAY: Final = 60.0
"""Seconds of quiet after the extraction backlog drains before the profile is
rewritten (with `background`; see the module docstring)."""

TRANSIENT_JOB_ERRORS: Final = (*DEFAULT_TRANSIENT_ERRORS, LLMError, EmbedderError)
"""Errors that never make extraction skip a turn (a server outage says nothing about it)."""


def build_tool_providers(settings: Settings) -> list[ToolProvider]:
    """The tool providers configured in `settings` (ADR 0022): an MCP provider when
    `[mcp] enabled` lists servers, else none (and the MCP SDK is never imported)."""
    if not settings.mcp.enabled:
        return []
    from lucy.adapters.mcp.provider import McpToolProvider
    from lucy.adapters.mcp.secrets import KeychainSecrets

    return [McpToolProvider(settings.mcp, secrets=KeychainSecrets(), root=settings.mcp.path)]


_NAMED_TOOL: Final = re.compile(r"\bthe (\w+) tool\b")
_ABOUT_CONFIRMING: Final = re.compile(r"\bask the user for confirmation themselves\b")


def system_prompt(
    tools: bool | Collection[str],
    *,
    confirming: bool | None = None,
    extra_guidance: Sequence[str] = (),
    hindi: bool = True,
) -> str:
    """The fixed part of the system prompt: the persona, plus the tool guidance
    (`prompts/tools.md`) when any tool is enabled.

    `hindi` is whether Hindi replies (in Devanagari) are wanted: when false, the
    persona's reply-language paragraph is replaced by `prompts/language_en.md`
    (reply in English; the voice cannot speak Hindi). See `voice_speaks_hindi`.

    `tools` is whether tools are enabled, or the names of the enabled tools: then a
    guidance sentence about "the weather tool" (any "the NAME tool") is left out when
    that tool is not enabled. Measured: with the weather sentence present but the tool
    disabled, the model called a `weather` tool it was not offered.

    `confirming` is whether any enabled tool asks the user before it runs
    (`Risk.DANGEROUS`, ADR 0006); the sentence about such tools is left out when
    not. It defaults to true when `tools` is True and false for a list of names.

    `extra_guidance` are sentences from tool providers (the user's per-tool MCP
    `guidance`), appended as one paragraph before filtering, so each must name its
    tool as "the NAME tool" and is kept only while that tool is enabled.
    """
    persona = load_prompt("persona").strip()
    if not hindi:
        persona = "\n\n".join(
            load_prompt("language_en").strip() if p.startswith(_REPLY_LANGUAGE) else p
            for p in persona.split("\n\n")
        )
    if not tools:
        return persona
    guidance = load_prompt("tools").strip()
    if extra_guidance:
        guidance += "\n\n" + " ".join(" ".join(s.split()) for s in extra_guidance)
    if not isinstance(tools, bool):
        enabled = set(tools)

        asks = bool(confirming)

        def keep(sentence: str) -> bool:
            if not asks and _ABOUT_CONFIRMING.search(sentence):
                return False
            return all(name in enabled for name in _NAMED_TOOL.findall(sentence))

        paragraphs = [
            " ".join(s for s in re.split(r"(?<=[.!?])\s+", paragraph) if keep(s))
            for paragraph in guidance.split("\n\n")
        ]
        guidance = "\n\n".join(p for p in paragraphs if p)
    return f"{persona}\n\n{guidance}"


_REPLY_LANGUAGE: Final = "Reply in the language of the user's latest message"


def voice_speaks_hindi(settings: Settings, registry: Registry | None = None) -> bool:
    """Whether Lucy may reply in Hindi: true without a `[tts]` section (text only),
    else whether the configured TTS adapter's class declares "hi" in its
    `capabilities.languages`. Decided from the configuration for every client, voice
    or text, so a conversation continued in `lucy chat` and `lucy voice` gets the same
    instruction and the prompt prefix stays fixed. An adapter class without class-level
    capabilities (the Ollama/phonon voice), or one not registered, counts as English
    only (an unknown adapter fails later, when `lucy voice` creates it)."""
    if settings.tts is None:
        return True
    get = registry.get if registry is not None else _registry.get
    try:
        cls = get("tts", settings.tts.adapter)
    except RegistryError:
        return False
    caps = getattr(cls, "capabilities", None)
    return isinstance(caps, TTSCaps) and "hi" in caps.languages


def _guidance_sentences(extra_guidance: Sequence[str]) -> list[list[str]]:
    guidance = load_prompt("tools").strip()
    if extra_guidance:
        guidance += "\n\n" + " ".join(" ".join(s.split()) for s in extra_guidance)
    return [re.split(r"(?<=[.!?])\s+", paragraph) for paragraph in guidance.split("\n\n")]


def group_guidance(
    groups: Mapping[str, ToolGroup], core: Collection[str], extra_guidance: Sequence[str] = ()
) -> dict[str, str]:
    """The guidance sentences that `system_prompt(core)` leaves out, by tool group
    (ADR 0025): a sentence goes to the group whose tools, with the core ones, cover
    every tool it names. A sentence naming tools of two groups is left out."""
    found: dict[str, list[str]] = {}
    always = set(core)
    for paragraph in _guidance_sentences(extra_guidance):
        for sentence in paragraph:
            named = set(_NAMED_TOOL.findall(sentence))
            if not named or named <= always:
                continue
            for name, group in groups.items():
                if named <= always | set(group.tools):
                    found.setdefault(name, []).append(sentence)
                    break
    return {name: " ".join(sentences) for name, sentences in found.items()}


def tool_groups(settings: Settings, tools: Sequence[Tool]) -> list[ToolGroup]:
    """The groups of `tools` (ADR 0025): the built-in ones (`lucy.tools.GROUPS`), then
    one per MCP server, or per `group` its settings name."""
    groups = list(GROUPS)
    mcp: dict[str, list[str]] = {}
    configs: dict[str, McpServerSettings] = {}
    for tool in tools:
        server = getattr(tool, "server", None)
        if isinstance(server, str) and server in settings.mcp.servers:
            config = settings.mcp.servers[server]
            name = config.group or server
            mcp.setdefault(name, []).append(tool.name)
            configs.setdefault(name, config)
    for name, names in mcp.items():
        config = configs[name]
        existing = next((i for i, g in enumerate(groups) if g.name == name), None)
        if existing is not None:  # MCP tools added to a built-in group
            g = groups[existing]
            groups[existing] = ToolGroup(
                g.name, g.description, (*g.tools, *names), (*g.keywords, *config.group_keywords)
            )
            continue
        keywords = tuple(config.group_keywords) or keyword_words(name, *names)
        groups.append(
            ToolGroup(name, config.group_description or f"tools of {name}", tuple(names), keywords)
        )
    return groups


def build_selector(
    settings: Settings, tools: Sequence[Tool], extra_guidance: Sequence[str] = ()
) -> ToolSelector | None:
    """The tool selector for `[tools] dynamic` (ADR 0025), or None when it is off or
    every configured tool is core."""
    if not settings.tools.dynamic or not tools:
        return None
    names = {t.name for t in tools}
    core = settings.tools.core
    if core is None:
        core = [n for n in DEFAULT_CORE if n in names]
    groups = tool_groups(settings, tools)
    probe = ToolSelector(tools, groups, core=core, keep_turns=settings.tools.keep_turns)
    if not probe.active:
        return None
    return ToolSelector(
        tools,
        groups,
        core=core,
        keep_turns=settings.tools.keep_turns,
        guidance=group_guidance(probe.groups, [*probe.core, META_TOOL], extra_guidance),
    )


@dataclass(frozen=True)
class JobBacklog:
    """Background work left when the app stopped.

    `pending` counts jobs waiting to run (an interrupted job is put back and
    counted); `interrupted` is the kind of the job that was running when the
    app stopped, if any.
    """

    pending: int
    interrupted: str | None = None


class _GuardedRetriever:
    """Retrieval that cannot fail or stall a turn.

    The full retrieval gets `timeout` seconds; on failure or timeout the turn
    gets the profile and instructions alone (another `timeout`), and failing
    that an empty context. Cancellation (an interrupted turn) propagates.
    """

    def __init__(self, retriever: Retriever, graph: SQLiteGraphStore, timeout: float) -> None:
        self._retriever = retriever
        self._graph = graph
        self._timeout = timeout

    async def retrieve(self, query: str, budget: int) -> MemoryContext:
        try:
            async with asyncio.timeout(self._timeout):
                return await self._retriever.retrieve(query, budget)
        except Exception as exc:
            logger.warning(
                "memory retrieval failed (%s); answering with the profile and instructions only",
                _describe(exc, self._timeout),
            )
        try:
            async with asyncio.timeout(self._timeout):
                profile = await self._graph.current_profile()
                instructions = await self._graph.active_instructions()
        except Exception as exc:
            logger.warning(
                "memory unavailable (%s); answering without it", _describe(exc, self._timeout)
            )
            return MemoryContext()
        return MemoryContext(
            profile=profile.content if profile else "",
            instructions=[s.text for s in instructions],
        )


class _FallbackTraces:
    """A trace sink that records each step's LLM fallback (`gateway.add_fallbacks`).

    The orchestrator records a turn's trace in the task that ran the turn, which
    is where the gateway noted the fallbacks its streams took.
    """

    def __init__(self, sink: TraceSink) -> None:
        self._sink = sink

    async def record(self, trace: TurnTrace) -> None:
        await self._sink.record(add_fallbacks(trace))


def _describe(exc: BaseException, timeout: float) -> str:
    if isinstance(exc, TimeoutError):
        return f"timed out after {timeout:g}s"
    return f"{type(exc).__name__}: {exc}"


class LucyApp:
    """Lucy's runtime: memory, LLM gateway, context builder and orchestrator on one bus,
    plus the learning loop when it is configured (see the module docstring).

    Call `start` before using `bus` and `stop` when done, or use the app as an
    async context manager. `registry` defaults to the global adapter registry;
    the built-in adapter modules are always imported into the global one.

    `background=False` builds the learning pieces without running anything in
    the background (for tools that inspect or drive memory themselves).
    `retrieval_timeout` bounds memory retrieval per turn. `runner_options` are
    extra keyword arguments for the `JobRunner` (e.g. `defer_delay`, for tests).
    `clock` is "now" for the memory components (see the module docstring);
    `profile_delay` is the quiet period before a profile rewrite after
    extraction (default `PROFILE_DELAY` with `background`, else 0).
    `tool_builder` builds the chat model's tools from the settings (default
    `lucy.tools.build_tools` with the app's `clock`: the tools in `[tools]
    enabled`, none by default);
    tests pass one that injects a mock HTTP transport. With no tools, no tool
    specs are offered and the system prompt is the persona alone.
    `provider_builder` builds the `ToolProvider`s (ADR 0022; default
    `build_tool_providers`, none yet); they are started during `start`, their
    tools offered after the built-in ones, and they are closed on `stop`.

    `speech=True` (for `lucy voice`) also builds the `[stt]` and `[tts]`
    adapters and warms them up during `start` (loading the speech recogniser,
    asking the TTS for a word), and closes them on `stop`; they are then
    `stt` and `tts`. Without it no speech adapter is created or loaded.

    An app is started at most once. `stop` is idempotent, also safe after a
    failed `start`, and leaves nothing running: background requests stop
    first, the job runner is cancelled (a job in flight goes back to the queue
    without losing an attempt), the orchestrator has returned, and every store
    and adapter has been closed once.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        registry: Registry | None = None,
        background: bool = True,
        retrieval_timeout: float = RETRIEVAL_TIMEOUT,
        runner_options: Mapping[str, Any] | None = None,
        clock: Callable[[], datetime] = utc_now,
        profile_delay: float | None = None,
        tool_builder: Callable[[Settings], Sequence[Tool]] | None = None,
        provider_builder: Callable[[Settings], Sequence[ToolProvider]] | None = None,
        speech: bool = False,
    ) -> None:
        self.settings = settings
        self._speech = speech
        self._stt: STT | None = None
        self._tts: TTS | None = None
        self._tool_builder = tool_builder or (
            lambda s: build_tools(s, clock=clock, reminder_store=self._reminder_store)
        )
        self._reminder_store: SQLiteReminderStore | None = None
        self._vision: VisionService | None = None
        self._reminders: ReminderService | None = None
        self._tools: list[Tool] = []
        self._provider_builder = provider_builder or build_tool_providers
        self._providers: list[ToolProvider] = []
        self._own_tools: list[Tool] = []
        self._registry = registry
        self._background = background
        self._retrieval_timeout = retrieval_timeout
        self._runner_options = dict(runner_options or {})
        self._clock = clock
        if profile_delay is None:
            profile_delay = PROFILE_DELAY if background else 0.0
        self._profile_delay = profile_delay
        self._started = False
        self._bus: EventBus | None = None
        self._memory: MemoryStore | None = None
        self._gateway: LLMGateway | None = None
        self._orchestrator: Orchestrator | None = None
        self._run_task: asyncio.Task[None] | None = None
        # Learning.
        self._learning_off: str | None = "not started"
        self._embedder: Embedder | None = None
        self._graph: SQLiteGraphStore | None = None
        self._jobs: SQLiteJobStore | None = None
        self._retriever: Retriever | None = None
        self._runner: JobRunner | None = None
        self._handlers: dict[str, Handler] = {}
        self._in_flight: Job | None = None
        self._turn_ends: Subscription | None = None
        self._background_tasks: list[asyncio.Task[None]] = []
        self._backlog: JobBacklog | None = None
        self._residency: Residency | None = None
        self._residency_task: asyncio.Task[None] | None = None

    # --- Public surface ---------------------------------------------------------

    @property
    def bus(self) -> EventBus:
        """The event bus clients publish to and subscribe on.

        Available once `start` has created it; after `stop` it is closed.
        """
        if self._bus is None:
            raise RuntimeError("LucyApp is not running; call start() first")
        return self._bus

    @property
    def memory(self) -> MemoryStore:
        """The memory store. Available after `start`."""
        if self._memory is None:
            raise RuntimeError("LucyApp is not running; call start() first")
        return self._memory

    @property
    def gateway(self) -> LLMGateway:
        """The LLM gateway. Available after `start`."""
        if self._gateway is None:
            raise RuntimeError("LucyApp is not running; call start() first")
        return self._gateway

    @property
    def clock(self) -> Callable[[], datetime]:
        """The clock the memory components use."""
        return self._clock

    @property
    def learning_active(self) -> bool:
        """Whether the learning loop is wired in (graph, retrieval and jobs)."""
        return self._learning_off is None

    @property
    def learning_off_reason(self) -> str | None:
        """Why learning is not active (None when it is), e.g. "no [embedder] configured"."""
        return self._learning_off

    @property
    def graph(self) -> SQLiteGraphStore | None:
        """The knowledge graph, when learning is active."""
        return self._graph

    @property
    def jobs(self) -> SQLiteJobStore | None:
        """The background job queue, when learning is active."""
        return self._jobs

    @property
    def runner(self) -> JobRunner | None:
        """The job runner (running as a task only with `background`), when learning is active."""
        return self._runner

    @property
    def retriever(self) -> Retriever | None:
        """The retriever attached to the memory store, when learning is active."""
        return self._retriever

    @property
    def reminders(self) -> ReminderService | None:
        """Fires and delivers due reminders (ADR 0020), when a reminder tool is enabled.

        A client that can tell when it is idle runs `wait_due` / `fire` and publishes
        the `TimerFired`; None when no reminder tool is enabled."""
        return self._reminders

    @property
    def vision(self) -> VisionService | None:
        """The vision service (ADR 0026), when `[vision]` is enabled and something uses it.
        Only voice clients glance with it (stage 3); `lucy chat` never does."""
        return self._vision

    @property
    def residency(self) -> Residency | None:
        """The model residency policy, when it is running."""
        return self._residency

    @property
    def job_handlers(self) -> Mapping[str, Handler]:
        """The memory job handlers by job kind (empty when learning is off)."""
        return MappingProxyType(self._handlers)

    @property
    def backlog(self) -> JobBacklog | None:
        """Background work left when `stop` ran; None before that or without learning."""
        return self._backlog

    async def pending_jobs(self) -> int:
        """Jobs waiting or running now (0 without learning)."""
        if self._jobs is None:
            return 0
        counts = await self._jobs.counts()
        return counts[JobStatus.PENDING] + counts[JobStatus.RUNNING]

    async def schedule_maintenance(self, *, force: bool = False) -> list[Job]:
        """Enqueue the maintenance that is due (with `force`, all of it) and wake the
        runner. Also requests extraction if episodes are waiting for it."""
        if self._graph is None or self._jobs is None:
            return []
        hours = self.settings.learning.maintenance_interval_hours
        interval = timedelta(0) if force else timedelta(hours=hours)
        enqueued: list[Job] = []
        if await self._graph.unextracted_episodes(limit=1):
            enqueued.append(await request_extraction(self._jobs))
        enqueued += await schedule_maintenance(
            self._jobs, self._graph, now=self._clock(), interval=interval
        )
        if self._runner is not None:
            self._runner.notify()
        return enqueued

    def _build_vision(self, create: Callable[..., Any]) -> VisionService | None:
        """The `VisionService` (ADR 0026) when `[vision]` is enabled and a tool or the
        glance (stage 3) uses it. Creating the camera adapter does not open the camera."""
        config = self.settings.vision
        used = bool(VISION_TOOLS & set(self.settings.tools.enabled)) or config.glance
        if not config.enabled or not used:
            return None
        assert self._gateway is not None
        camera = create("camera", config.camera.adapter, **config.camera.options)
        if not isinstance(camera, Camera):
            raise TypeError(f"camera adapter {config.camera.adapter!r} is not a Camera")
        recognizer, book, off = self._build_faces(create)
        faces = config.faces
        return VisionService(
            gateway=self._gateway,
            camera=camera,
            role=config.role,
            timeout=config.timeout,
            recognizer=recognizer,
            book=book,
            rules=FaceRules(
                threshold=faces.threshold,
                min_score=faces.min_score,
                min_size=faces.min_size,
                enrol_frames=faces.enrol_frames,
            ),
            faces_off=off,
        )

    def _build_faces(
        self, create: Callable[..., Any]
    ) -> tuple[FaceRecognizer | None, FaceBook | None, str | None]:
        """The face recogniser and book (stage 2), or why face matching is off. Missing
        models switch it off with one warning naming `lucy vision download`; `look`
        keeps working without names. Creating the adapter starts nothing."""
        faces = self.settings.vision.faces
        if not faces.enabled:
            return None, None, None
        options = {"models": faces.models, "pack": faces.pack, **faces.options}
        recognizer = create("faces", faces.adapter, **options)
        if not isinstance(recognizer, FaceRecognizer):
            raise TypeError(f"faces adapter {faces.adapter!r} is not a FaceRecognizer")
        book = FaceBook(Workspace(self.settings.workspace.path))
        if getattr(recognizer, "present", True) is False:
            off = (
                f"the face models ({faces.pack}) are not downloaded; the user can run "
                "`lucy vision download` in a terminal, then restart Lucy"
            )
            logger.warning("face matching is off: %s", off)
            return recognizer, book, off
        return recognizer, book, None

    # --- Lifecycle ----------------------------------------------------------------

    async def start(self) -> None:
        """Build every component and start the orchestrator (and the learning loop).

        On failure everything built so far is torn down and the error is re-raised.
        A learning component failing to start does not fail `start`: learning is
        switched off and the reason logged.
        """
        if self._started:
            raise RuntimeError("LucyApp.start() can only be called once")
        self._started = True
        try:
            _registry.load_adapters(BUILTIN_ADAPTER_MODULES)
            create = self._registry.create if self._registry is not None else _registry.create

            memory_config = self.settings.memory or DEFAULT_MEMORY
            memory = create("memory", memory_config.adapter, **memory_config.options)
            if not isinstance(memory, MemoryStore):
                await _close_quietly(memory, "close")
                raise TypeError(f"memory adapter {memory_config.adapter!r} is not a MemoryStore")
            self._memory = memory

            # Limitation: if creating a later role's adapter fails, adapters already
            # created inside from_settings are not reachable here and are not closed.
            self._gateway = LLMGateway.from_settings(self.settings.llm, registry=self._registry)
            self._bus = EventBus()

            await self._start_learning(create)

            trace = _FallbackTraces(memory) if isinstance(memory, TraceSink) else None
            clock = self._clock
            timer_handlers: dict[str, TimerHandler] = {}
            if REMINDER_TOOLS & set(self.settings.tools.enabled):
                # One connection for the tools and the delivery (the tools share it).
                self._reminder_store = build_store(self.settings, clock)
                self._reminders = ReminderService(
                    self._reminder_store, memory, local_announcement, clock=clock
                )
                timer_handlers[REMINDER_TIMER] = self._reminders.handle
            self._tools = list(self._tool_builder(self.settings))
            self._own_tools = list(self._tools)  # closed by the app; the rest by providers
            self._providers = list(self._provider_builder(self.settings))
            extra_guidance: list[str] = []
            for provider in self._providers:
                await provider.start()
                self._tools.extend(provider.tools())
                # Optional on the port: the MCP provider's per-tool guidance.
                extra_guidance.extend(getattr(provider, "guidance", list)())
            for tool in self._tools:
                if hasattr(tool, "voice_client"):  # the voice enrolment tools
                    tool.voice_client = self._speech  # type: ignore[attr-defined]
            vision = self._vision = self._build_vision(create)
            if vision is not None:
                for tool in self._tools:
                    if hasattr(tool, "vision_service"):  # look and the face tools
                        tool.vision_service = vision  # type: ignore[attr-defined]
            if ENROL_TOOLS & {t.name for t in self._tools}:
                timer_handlers[ENROLMENT_TIMER] = handle_enrolment
            selector = build_selector(self.settings, self._tools, extra_guidance)
            context = ContextBuilder(
                system_prompt(
                    [t.name for t in self._tools]
                    if selector is None
                    else [*selector.core, META_TOOL],
                    confirming=any(t.risk == Risk.DANGEROUS for t in self._tools),
                    extra_guidance=extra_guidance,
                    hindi=voice_speaks_hindi(self.settings, self._registry),
                ),
                clock=lambda: clock().astimezone(),
                layout=self.settings.context.layout,
                history_tokens=self.settings.context.history_tokens,
            )
            # With dynamic tools, the turn after a quiet spell offers only the core set.
            offered = self._tools if selector is None else selector.specs(())
            _warn_long_prefix(context, offered, self.settings.context.history_tokens)
            self._orchestrator = Orchestrator(
                bus=self._bus,
                gateway=self._gateway,
                memory=memory,
                context=context,
                tools=self._tools,
                trace=trace,
                confirm_timeout=self.settings.tools.confirm_timeout,
                timer_handlers=timer_handlers,
                selector=selector,
                # Built whether or not tools are dynamic: they judge stale exchanges.
                tool_groups=tool_groups(self.settings, self._tools),
            )
            self._run_task = asyncio.create_task(self._orchestrator.run(), name="lucy-orchestrator")
            self._start_residency(self._orchestrator)
            if self._background and self.learning_active:
                self._start_background()
            if self._speech:
                await self._start_speech(create)
        except BaseException:
            await self.stop()
            raise
        logger.info(
            "Lucy started (LLM roles: %s; memory learning: %s; tools: %s; model residency: %s)",
            ", ".join(self._gateway.roles()),
            "active" if self.learning_active else f"off, {self._learning_off}",
            ", ".join(tool.name for tool in self._tools) or "none",
            self._residency_summary(),
        )

    def _start_residency(self, orchestrator: Orchestrator) -> None:
        """Run the residency policy if it is enabled and the chat adapter supports it."""
        config = self.settings.residency
        assert self._gateway is not None and self._bus is not None
        if not config.enabled or not Residency.supported(self._gateway):
            return
        self._residency = Residency(
            bus=self._bus,
            gateway=self._gateway,
            jobs=self._runner,
            prefix=orchestrator.next_turn_prefix,
            idle_seconds=config.idle_seconds,
            text_quiet_seconds=config.text_quiet_seconds,
            min_size_bytes=None if config.min_size_gb is None else int(config.min_size_gb * 1e9),
        )
        self._residency_task = asyncio.create_task(self._residency.run(), name="lucy-residency")

    def _residency_summary(self) -> str:
        config = self.settings.residency
        if self._residency is None:
            return "off" if not config.enabled else "off, the chat adapter does not support it"
        rule = f"idle {config.idle_seconds:g}s, text {config.text_quiet_seconds:g}s"
        if config.min_size_gb is not None:
            rule += f", from {config.min_size_gb:g} GB"
        return f"on ({rule})"

    async def _stop_residency(self) -> None:
        residency, self._residency = self._residency, None
        task, self._residency_task = self._residency_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if residency is not None:
            await residency.aclose()

    @property
    def stt(self) -> STT:
        """The speech recogniser. Available after `start` with `speech=True`."""
        if self._stt is None:
            raise RuntimeError("speech is not running; create the app with speech=True")
        return self._stt

    @property
    def tts(self) -> TTS:
        """The speech synthesiser. Available after `start` with `speech=True`."""
        if self._tts is None:
            raise RuntimeError("speech is not running; create the app with speech=True")
        return self._tts

    async def _start_speech(self, create: Callable[..., Any]) -> None:
        """Create the STT and TTS adapters and warm both up (concurrently)."""
        stt_config, tts_config = self.settings.stt, self.settings.tts
        missing = [name for name, c in (("[stt]", stt_config), ("[tts]", tts_config)) if c is None]
        if missing:
            raise ValueError(
                f"voice needs {' and '.join(missing)} in the settings (see lucy.example.toml)"
            )
        assert stt_config is not None and tts_config is not None
        self._stt = create("stt", stt_config.adapter, **stt_config.options)
        if not isinstance(self._stt, STT):
            raise TypeError(f"stt adapter {stt_config.adapter!r} is not an STT")
        self._tts = create("tts", tts_config.adapter, **tts_config.options)
        if not isinstance(self._tts, TTS):
            raise TypeError(f"tts adapter {tts_config.adapter!r} is not a TTS")
        await asyncio.gather(_warm_up(self._stt), _warm_up(self._tts))

    def _learning_blocker(self) -> str | None:
        """Which piece learning lacks, or None if it can be wired in."""
        if not self.settings.learning.enabled:
            return "disabled in the [learning] settings"
        if self.settings.embedder is None:
            return "no [embedder] configured"
        if not isinstance(self._memory, SQLiteMemoryStore):
            return "the memory adapter is not the SQLite store"
        assert self._gateway is not None
        missing = [role for role in LEARNING_ROLES if role not in self._gateway.roles()]
        if missing:
            return f"no {', '.join(repr(r) for r in missing)} LLM role configured"
        return None

    async def _start_learning(self, create: Callable[..., Any]) -> None:
        """Wire in the learning loop if possible; never raises except on cancellation."""
        reason = self._learning_blocker()
        if reason is not None:
            self._learning_off = reason
            logger.info("memory learning is off (%s): chat replays the conversation only", reason)
            return
        memory, gateway, embedder_config = self._memory, self._gateway, self.settings.embedder
        assert isinstance(memory, SQLiteMemoryStore)
        assert gateway is not None and embedder_config is not None
        try:
            embedder = create("embedder", embedder_config.adapter, **embedder_config.options)
            self._embedder = embedder
            if not isinstance(embedder, Embedder):
                raise TypeError(f"embedder adapter {embedder_config.adapter!r} is not an Embedder")
            graph = self._graph = SQLiteGraphStore(memory.path, embedder, clock=self._clock)
            await graph.stats()  # connects, loads sqlite-vec, migrates
            jobs = self._jobs = SQLiteJobStore(memory.path)
            await jobs.counts()
            retrieval = self.settings.retrieval.model_dump(exclude_none=True)
            self._retriever = Retriever(graph, embedder, clock=self._clock, **retrieval)
            self._handlers = self._build_handlers(gateway, graph, jobs)
            requeued = await jobs.requeue_unhandled(self._handlers)
            if requeued:
                logger.info(
                    "put back %s that failed for want of a handler: %s",
                    "a job" if len(requeued) == 1 else f"{len(requeued)} jobs",
                    ", ".join(sorted({job.kind for job in requeued})),
                )
            tracked = {kind: self._track(handler) for kind, handler in self._handlers.items()}
            self._runner = JobRunner(jobs, tracked, defer_on=(Preempted,), **self._runner_options)
        except Exception as exc:
            reason = f"could not start it: {type(exc).__name__}: {exc}"
            logger.warning("memory learning is off (%s)", reason)
            await self._close_learning()
            self._learning_off = reason
            return
        memory.use_retriever(_GuardedRetriever(self._retriever, graph, self._retrieval_timeout))
        self._learning_off = None
        if self._background:
            assert self._bus is not None
            # Subscribe before the orchestrator runs, so no turn's end is missed.
            self._turn_ends = self._bus.subscribe(TurnEnded)
            try:
                await self.schedule_maintenance()
            except Exception:
                logger.exception("could not schedule memory maintenance at startup")

    def _decay_policy(self) -> DecayPolicy:
        learning, default = self.settings.learning, DecayPolicy()
        grace, half_life = learning.decay_grace_days, learning.decay_half_life_days
        return DecayPolicy(
            grace=default.grace if grace is None else timedelta(days=grace),
            half_life=default.half_life if half_life is None else timedelta(days=half_life),
        )

    def _build_handlers(
        self, gateway: LLMGateway, graph: SQLiteGraphStore, jobs: SQLiteJobStore
    ) -> dict[str, Handler]:
        clock = self._clock
        # Extraction and reflection reconcile with the same model, policy and clock.
        reconciler = Reconciler(graph, gateway, role="extraction", clock=clock)
        extract = extraction_handler(
            Extractor(graph, gateway, clock=clock, reconciler=reconciler),
            jobs,
            max_turns=self.settings.learning.extract_max_turns,
            transient_errors=TRANSIENT_JOB_ERRORS,
        )
        return {
            EXTRACT_JOB_KIND: self._then_refresh_profile(extract, jobs),
            DECAY_JOB: decay_handler(graph, policy=self._decay_policy(), clock=clock),
            PROFILE_JOB: profile_handler(ProfileWriter(gateway, graph, clock=clock), graph=graph),
            SUMMARIES_JOB: summaries_handler(gateway, graph),
            MERGE_JOB: merge_handler(gateway, graph, role="summary"),
            REFLECT_JOB: reflection_handler(Reflector(graph, gateway, reconciler, clock=clock)),
            REEMBED_JOB: reembed_handler(graph, jobs),
        }

    def _then_refresh_profile(self, extract: Handler, jobs: SQLiteJobStore) -> Handler:
        """`extract`, then a profile rewrite request once the extraction backlog has
        drained (see the module docstring)."""

        async def run(job: Job) -> None:
            await extract(job)
            waiting = await jobs.list_jobs(statuses=[JobStatus.PENDING])
            if any(j.kind == EXTRACT_JOB_KIND for j in waiting):
                return  # more extraction follows; the last one requests the rewrite
            await request_profile(jobs, delay=self._profile_delay)

        return run

    def _track(self, handler: Handler) -> Handler:
        """`handler`, recording the job in flight (for the exit message)."""

        async def run(job: Job) -> None:
            self._in_flight = job
            try:
                await handler(job)
            finally:
                self._in_flight = None

        return run

    def _start_background(self) -> None:
        assert self._runner is not None and self._turn_ends is not None
        self._background_tasks = [
            asyncio.create_task(self._runner.run(), name="lucy-jobs"),
            asyncio.create_task(self._request_extractions(self._turn_ends), name="lucy-extract"),
            asyncio.create_task(self._maintain(), name="lucy-maintenance"),
        ]

    async def _request_extractions(self, turn_ends: Subscription) -> None:
        """After every turn, ask for extraction and wake the runner."""
        assert self._jobs is not None and self._runner is not None
        async for _ in turn_ends:
            try:
                await request_extraction(self._jobs)
            except Exception:
                logger.exception("could not request memory extraction")
            self._runner.notify()

    async def _maintain(self) -> None:
        """Hourly: enqueue whatever maintenance is due."""
        while True:
            await asyncio.sleep(MAINTENANCE_CHECK_SECONDS)
            try:
                await self.schedule_maintenance()
            except Exception:
                logger.exception("could not schedule memory maintenance")

    async def stop(self) -> None:
        """Stop the background work and the orchestrator, close every store and adapter.

        Idempotent. Never waits for a background job to finish: a job in flight
        is cancelled and returned to the queue.
        """
        try:
            if self._bus is not None:
                self._bus.close()  # no new turns, no new extraction requests
            await self._stop_residency()
            if self._turn_ends is not None:
                self._turn_ends.close()
            background, self._background_tasks = self._background_tasks, []
            interrupted = self._in_flight.kind if self._in_flight is not None else None
            for task in background:
                task.cancel()
            task, self._run_task = self._run_task, None
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    if not task.done():  # stop() itself was cancelled
                        task.cancel()
                        raise
                except Exception:
                    logger.exception("the orchestrator failed")
            self._orchestrator = None
            if background:
                await asyncio.wait(background)
                for done in background:
                    if not done.cancelled() and done.exception() is not None:
                        logger.error("background task failed", exc_info=done.exception())
            if self._jobs is not None and self._backlog is None:
                await self._record_backlog(interrupted)
        finally:
            await self._stop_residency()
            await self._close_learning()
            gateway, self._gateway = self._gateway, None
            if gateway is not None:
                closed: set[int] = set()
                for role in gateway.roles():
                    llm = gateway.for_role(role)
                    if id(llm) not in closed:
                        closed.add(id(llm))
                        await _close_quietly(llm, "aclose")
            # Providers before the tools and stores: their tools may use them.
            providers, self._providers = self._providers, []
            for provider in reversed(providers):
                await _close_quietly(provider, "aclose")
            tools, self._own_tools, self._tools = self._own_tools, [], []
            for tool in tools:
                await _close_quietly(tool, "aclose")
            vision, self._vision = self._vision, None
            if vision is not None:
                await vision.aclose()
            self._reminders = None
            reminder_store, self._reminder_store = self._reminder_store, None
            if reminder_store is not None:
                await _close_quietly(reminder_store, "close")
            stt, tts, self._stt, self._tts = self._stt, self._tts, None, None
            for speech in (stt, tts):
                if speech is not None:
                    await _close_quietly(speech, "aclose")
            memory, self._memory = self._memory, None
            if memory is not None:
                await _close_quietly(memory, "close")

    async def _record_backlog(self, interrupted: str | None) -> None:
        try:
            async with asyncio.timeout(2):
                pending = await self.pending_jobs()
        except Exception:
            logger.warning("could not count the background jobs left", exc_info=True)
            return
        self._backlog = JobBacklog(pending=pending, interrupted=interrupted)

    async def _close_learning(self) -> None:
        """Detach retrieval and close the graph, job store and embedder (each once)."""
        if isinstance(self._memory, SQLiteMemoryStore):
            self._memory.use_retriever(None)
        self._runner = None
        self._retriever = None
        graph, self._graph = self._graph, None
        if graph is not None:
            await _close_quietly(graph, "close")
        jobs, self._jobs = self._jobs, None
        if jobs is not None:
            await _close_quietly(jobs, "close")
        embedder, self._embedder = self._embedder, None
        if embedder is not None:
            await _close_quietly(embedder, "aclose")

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()


PREFIX_WARN_TOKENS: Final = 4500
"""Estimated prompt prefix (system prompt, tool specifications, history cap) above
which startup warns: about 5 s of pre-fill at the 850 tokens/s measured with
gemma4:12b-mlx, so the residency re-warm may not finish before the next
utterance (ADR 0017)."""


def prefix_estimate(
    persona: str, tools: Sequence[Tool | ToolSpec], history_tokens: int | None
) -> int:
    """Estimated tokens of the prompt before the latest user message, at most."""
    specs = sum(
        estimate_tokens(json.dumps((t if isinstance(t, ToolSpec) else tool_spec(t)).model_dump()))
        for t in tools
    )
    return estimate_tokens(persona) + specs + (history_tokens or 0)


def _warn_long_prefix(
    context: ContextBuilder, tools: Sequence[Tool | ToolSpec], history_tokens: int | None
) -> None:
    estimate = prefix_estimate(context.persona, tools, history_tokens)
    if estimate > PREFIX_WARN_TOKENS:
        logger.warning(
            "the chat prompt's prefix is about %d tokens (system prompt and %d tool "
            "specifications, plus up to %s of history); pre-filling it after the model "
            "is reloaded takes about %.0f s, so a quick next turn may wait for it. "
            "Fewer tools, `fixed` MCP arguments or a smaller [context] history_tokens "
            "shorten it.",
            estimate,
            len(tools),
            "no limit" if history_tokens is None else history_tokens,
            estimate / 850,
        )


async def _warm_up(component: Any) -> None:
    """Await `component.warm_up()` if it has one."""
    warm_up = getattr(component, "warm_up", None)
    if warm_up is not None:
        await warm_up()


async def _close_quietly(component: Any, method: str) -> None:
    """Await `component.<method>()` if it exists; log and swallow errors."""
    close = getattr(component, method, None)
    if close is None:
        return
    try:
        result = close()
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("closing %s failed", type(component).__name__)
