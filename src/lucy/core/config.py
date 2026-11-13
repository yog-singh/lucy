"""Settings loaded from a TOML file (ADR 0007).

Each section names an adapter plus free-form options for its constructor. LLMs
are configured per role; `chat` is required.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from lucy.core.context import ContextLayout


class ConfigError(ValueError):
    """The settings file is missing, unreadable or invalid."""


class AdapterConfig(BaseModel):
    """Which adapter to use for a port, and the options passed to its constructor.

    In TOML every key other than `adapter` is collected into `options`, e.g.
    `{ adapter = "ollama", model = "qwen3:14b" }` gives
    `options == {"model": "qwen3:14b"}`. In Python it can also be built as
    `AdapterConfig(adapter="ollama", options={...})`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    adapter: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)

    own_keys: ClassVar[frozenset[str]] = frozenset({"adapter"})
    """Keys that are fields of this model; every other key goes into `options`."""

    @model_validator(mode="before")
    @classmethod
    def _collect_options(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if set(data) <= cls.own_keys | {"options"} and isinstance(data.get("options", {}), dict):
            return data  # already in canonical form
        options = {key: value for key, value in data.items() if key not in cls.own_keys}
        own = {key: data[key] for key in cls.own_keys if key in data}
        return {"adapter": data.get("adapter"), **own, "options": options}


class LLMRoleConfig(AdapterConfig):
    """An LLM role: its adapter config plus how the gateway routes its FOREGROUND streams.

    `fallback` names another role that serves a FOREGROUND stream once when this
    role's stream ends, or produces nothing for `first_output_timeout` seconds,
    without any text or tool call. `after_tool` names a role that serves this
    role's FOREGROUND streams whose last message is a tool result (the answer
    step after a lookup). See `LLMGateway.stream`. None of these keys is passed
    to the adapter.
    """

    fallback: str | None = Field(default=None, min_length=1)
    first_output_timeout: float | None = Field(default=None, gt=0)
    after_tool: str | None = Field(default=None, min_length=1)

    own_keys: ClassVar[frozenset[str]] = frozenset(
        {"adapter", "fallback", "first_output_timeout", "after_tool"}
    )


class LLMSettings(BaseModel):
    """LLM configuration: one adapter config per role."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roles: dict[str, LLMRoleConfig]

    @field_validator("roles", mode="before")
    @classmethod
    def _accept_adapter_configs(cls, roles: Any) -> Any:
        # A plain AdapterConfig (as built in Python) is a role without a fallback.
        if not isinstance(roles, dict):
            return roles
        return {
            name: config.model_dump()
            if isinstance(config, AdapterConfig) and not isinstance(config, LLMRoleConfig)
            else config
            for name, config in roles.items()
        }

    @field_validator("roles")
    @classmethod
    def _require_chat(cls, roles: dict[str, LLMRoleConfig]) -> dict[str, LLMRoleConfig]:
        if "chat" not in roles:
            raise ValueError("[llm.roles] must define a 'chat' role")
        for name, config in roles.items():
            for key in ("fallback", "after_tool"):
                other = getattr(config, key)
                if other is None:
                    continue
                if other == name:
                    raise ValueError(f"role {name!r} cannot be its own {key}")
                if other not in roles:
                    raise ValueError(
                        f"role {name!r} has {key} {other!r}, which is not a "
                        f"configured role; configured roles: {', '.join(sorted(roles))}"
                    )
        return roles


class LearningSettings(BaseModel):
    """The learning loop (ADR 0012): background extraction and maintenance.

    Learning runs only when an embedder is configured and memory is the SQLite
    store; `enabled = false` turns it off even then (chat then replays the
    session's conversation only).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    maintenance_interval_hours: float = Field(default=24, gt=0)
    """How often decay, entity summaries, duplicate merging and the profile rewrite run."""
    extract_max_turns: int = Field(default=4, ge=1)
    """Turns one extraction job processes before handing over to a follow-up job."""
    decay_grace_days: float | None = Field(default=None, ge=0)
    """Days an inferred statement keeps its confidence after it was last observed.
    None: the memory module's default (`lucy.memory.decay.DecayPolicy`)."""
    decay_half_life_days: float | None = Field(default=None, gt=0)
    """After the grace period, confidence halves every this many days of not being
    observed. None: the memory module's default."""


class RetrievalSettings(BaseModel):
    """Per-turn memory retrieval thresholds (`lucy.memory.retrieval.Retriever`).

    Every field is optional; None (absent) keeps the retriever's default. The
    similarity values are cosine similarities and depend on the embedding model
    (the defaults were measured for `embeddinggemma-2` with the "query: " /
    "passage: " prefixes): re-measure them when the embedder changes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vector_min_similarity: float | None = Field(default=None, ge=-1, le=1)
    """Absolute floor: vector matches less similar than this are dropped."""
    vector_margin: float | None = Field(default=None, ge=0)
    """Relative floor: vector matches more than this below the best match are dropped
    (2 or more turns it off)."""
    keyword_min_similarity: float | None = Field(default=None, ge=-1, le=1)
    """Keyword matches whose vector similarity to the query is below this are dropped
    (-1 turns the check off)."""
    entity_min_similarity: float | None = Field(default=None, ge=-1, le=1)
    """Entities similar to the query (not named in it) are expanded only from this
    similarity up."""
    max_statements: int | None = Field(default=None, ge=1)
    """At most this many retrieved statements are given to a turn."""


class _ToolOptions(BaseModel):
    """Options every internet tool has."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_chars: int = Field(default=6000, ge=200)
    """Longest result handed to the model, in characters (results go straight into its context)."""
    timeout: float = Field(default=10.0, gt=0)
    """Seconds one tool call may take in total, including retries and redirects."""


class WikipediaSettings(_ToolOptions):
    """The `wikipedia` tool: search Wikipedia and return the best article's introduction."""

    language: str = Field(default="en", pattern=r"^[a-z][a-z0-9-]{1,15}$")
    """Wikipedia language edition, as in its address (`en` for en.wikipedia.org)."""
    other_matches: int = Field(default=4, ge=0, le=10)
    """Titles of further matching articles listed after the best one."""


class WebSearchSettings(_ToolOptions):
    """The `web_search` tool: query a SearXNG instance's JSON API."""

    searxng_url: str | None = Field(default=None, pattern=r"^https?://[^\s/]+")
    """Base address of the SearXNG instance, e.g. "http://127.0.0.1:8890". Required when enabled."""
    max_results: int = Field(default=5, ge=1, le=20)
    """Results returned to the model."""


class FetchPageSettings(_ToolOptions):
    """The `fetch_page` tool: fetch a public web page and return its readable text."""

    max_bytes: int = Field(default=2_000_000, ge=1024)
    """Most bytes of a response body read (decompressed); the rest is not downloaded."""
    max_redirects: int = Field(default=5, ge=0, le=20)
    """Redirects followed, each one checked like the first address."""
    allow_private_hosts: list[str] = Field(default_factory=list)
    """Hosts on the local network the tool may fetch despite the public-address rule:
    host names (exact match), IP addresses or networks such as "192.168.1.0/24".
    Empty by default; add for example a local Home Assistant only when needed."""


class WeatherSettings(_ToolOptions):
    """The `weather` tool: current conditions and a daily forecast from Open-Meteo."""

    units: Literal["metric", "imperial"] = "metric"
    """"metric" (°C, km/h, mm) or "imperial" (°F, mph, inches)."""


class TimeSettings(BaseModel):
    """The `time` tool: the date and time in a named place, from the system's time-zone
    database. It has no options and makes no network request."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CalendarSettings(_ToolOptions):
    """The calendar tools (`calendar`, `add_calendar_event`, `edit_calendar_event`,
    `delete_calendar_event`): the macOS calendar store
    (EventKit), on this machine only. `timeout` bounds one call to the store."""

    calendars: list[str] = Field(default_factory=list)
    """Names of the calendars the tools read, as the Calendar app shows them; empty: all."""
    default_calendar: str | None = Field(default=None, min_length=1)
    """Calendar new events go to when the user names none; None: the system default."""
    permission_timeout: float = Field(default=60.0, gt=0)
    """Seconds to wait for the user to answer macOS's calendar permission dialog."""


class AddCalendarEventSettings(BaseModel):
    """The `add_calendar_event` tool. Its options are in `[tools.calendar]`."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class EditCalendarEventSettings(BaseModel):
    """The `edit_calendar_event` tool (always asks the user first). Its options are in
    `[tools.calendar]`."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DeleteCalendarEventSettings(BaseModel):
    """The `delete_calendar_event` tool (always asks the user first). Its options are in
    `[tools.calendar]`."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolsSettings(BaseModel):
    """Which tools the chat model may use, and each tool's options.

    Nothing is enabled by default, so Lucy makes no network requests unless the
    user opts in (`time` and the calendar tools need no network). Each name in
    `enabled` must be one of the option tables below.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: list[str] = Field(default_factory=list)
    confirm_timeout: float = Field(default=60.0, gt=0, le=3600)
    """Seconds the user has to answer when a tool asks before changing or deleting
    something (ADR 0006); a later answer runs nothing."""
    wikipedia: WikipediaSettings = Field(default_factory=WikipediaSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    fetch_page: FetchPageSettings = Field(default_factory=FetchPageSettings)
    time: TimeSettings = Field(default_factory=TimeSettings)
    weather: WeatherSettings = Field(default_factory=WeatherSettings)
    calendar: CalendarSettings = Field(default_factory=CalendarSettings)
    add_calendar_event: AddCalendarEventSettings = Field(default_factory=AddCalendarEventSettings)
    edit_calendar_event: EditCalendarEventSettings = Field(
        default_factory=EditCalendarEventSettings
    )
    delete_calendar_event: DeleteCalendarEventSettings = Field(
        default_factory=DeleteCalendarEventSettings
    )

    @classmethod
    def known_tools(cls) -> list[str]:
        """The tool names `enabled` accepts."""
        return [name for name in cls.model_fields if name not in ("enabled", "confirm_timeout")]

    @model_validator(mode="after")
    def _check_enabled(self) -> Self:
        known = self.known_tools()
        unknown = [name for name in self.enabled if name not in known]
        if unknown:
            raise ValueError(
                f"[tools] enabled lists unknown tool(s) {', '.join(map(repr, unknown))}; "
                f"known tools: {', '.join(known)}"
            )
        duplicates = sorted({name for name in self.enabled if self.enabled.count(name) > 1})
        if duplicates:
            raise ValueError(f"[tools] enabled lists {', '.join(map(repr, duplicates))} twice")
        if "web_search" in self.enabled and not self.web_search.searxng_url:
            raise ValueError(
                "web_search is enabled but [tools.web_search] has no searxng_url "
                '(e.g. searxng_url = "http://127.0.0.1:8890"; see deploy/searxng/README.md)'
            )
        return self


class VoiceSettings(BaseModel):
    """`lucy voice`: listening, endpointing and audio devices (see docs/voice.md).

    The speech models themselves are the `[stt]` and `[tts]` sections.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint_silence_ms: int = Field(default=600, ge=100, le=5000)
    """Silence after speech that ends the user's turn."""
    min_speech_ms: int = Field(default=160, ge=32, le=2000)
    """Continuous speech needed before a turn starts (so clicks and knocks do not)."""
    preroll_ms: int = Field(default=300, ge=0, le=2000)
    """Audio kept from before speech was detected, so the first syllable is not cut off."""
    max_utterance_s: float = Field(default=30.0, gt=0, le=300)
    """Longest single utterance; a longer one is ended and sent as it is."""
    tail_ms: int = Field(default=300, ge=0, le=5000)
    """How long the microphone stays ignored after Lucy stops speaking."""
    vad_threshold: float = Field(default=0.5, gt=0, lt=1)
    """Speech probability (Silero VAD) from which a frame counts as speech."""
    input_device: int | str | None = None
    """Microphone: a device number or (part of) a name from `lucy voice --list-devices`;
    None is the system default."""
    output_device: int | str | None = None
    """Speaker, as for `input_device`."""


class WakeSettings(BaseModel):
    """`lucy listen`: the wake phrase in front of the voice session (see docs/voice.md).

    `lucy voice` does not use it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phrase: str = Field(default="hey jarvis", min_length=1)
    """The wake phrase as spoken; it is removed from the start of the transcript."""
    model: str = "data/models/wakeword/hey_jarvis_v0.1.onnx"
    """The phrase model (ONNX); the mel and embedding models are expected beside it."""
    threshold: float = Field(default=0.5, gt=0, lt=1)
    """Wake-word score from which the phrase counts as said."""
    followup_seconds: float = Field(default=8.0, ge=0, le=120)
    """After a reply (or the wake phrase alone), how long Lucy keeps listening without
    the wake phrase before going back to waiting."""
    chime: bool = True
    """Play a short chime on waking and on going back to waiting."""


class ContextSettings(BaseModel):
    """How the chat prompt is arranged (see `lucy.core.context`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layout: ContextLayout = "stable_prefix"
    """`"system"`: the date and retrieved memories in the system message, before the
    history. `"stable_prefix"`: only what rarely changes before the history, and the
    date and memories with the latest user message, so each turn's prompt extends
    the previous one and the model server can reuse its cache."""


class ResidencySettings(BaseModel):
    """Bounding the chat model's memory on its server (`lucy.core.residency`).

    Takes effect only when the chat role's adapter supports it
    (`LLMCaps.residency`: the `ollama` adapter). When Lucy has been idle for
    `idle_seconds` after model work (and, outside voice, nobody has sent a
    message for `text_quiet_seconds`), the model is unloaded, which frees the
    server's prompt cache, then loaded again and the next turn's prompt prefix
    pre-filled, so the next turn still starts warm.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    idle_seconds: float = Field(default=1.0, ge=0.1, le=600)
    """Nothing running or due (no turn, speech, model request or job) for this long."""
    text_quiet_seconds: float = Field(default=10.0, ge=0, le=3600)
    """Outside a voice session, also this long since the last turn ended."""
    min_size_gb: float | None = Field(default=None, gt=0)
    """Optional: unload only when the loaded model takes at least this many GB."""


class Settings(BaseModel):
    """All of Lucy's settings. Only `llm` is required; other sections are optional."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    llm: LLMSettings
    stt: AdapterConfig | None = None
    tts: AdapterConfig | None = None
    embedder: AdapterConfig | None = None
    memory: AdapterConfig | None = None
    learning: LearningSettings = Field(default_factory=LearningSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    wake: WakeSettings = Field(default_factory=WakeSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    residency: ResidencySettings = Field(default_factory=ResidencySettings)


def load_settings(path: str | Path) -> Settings:
    """Read and validate settings from the TOML file at `path`.

    Raises `ConfigError` if the file does not exist, is not valid TOML, or does
    not match the expected shape.
    """
    path = Path(path)
    try:
        with path.open("rb") as file:
            raw = tomllib.load(file)
    except FileNotFoundError:
        raise ConfigError(f"settings file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read settings file {path}: {exc}") from exc
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid settings in {path}:\n{exc}") from exc
