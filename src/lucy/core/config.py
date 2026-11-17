"""Settings loaded from a TOML file (ADR 0007).

Each section names an adapter plus free-form options for its constructor. LLMs
are configured per role; `chat` is required.
"""

from __future__ import annotations

import re
import string
import tomllib
from pathlib import Path, PurePath
from typing import Any, ClassVar, Final, Literal, NamedTuple, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from lucy.core.context import ContextLayout
from lucy.core.types import Risk


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
    step after a lookup). `silent_retries` is how many times such a stream that
    ends early without output is asked of the same role again before the
    fallback (None: the gateway's default, `DEFAULT_SILENT_RETRIES`). See
    `LLMGateway.stream`. None of these keys is passed to the adapter.
    """

    fallback: str | None = Field(default=None, min_length=1)
    first_output_timeout: float | None = Field(default=None, gt=0)
    after_tool: str | None = Field(default=None, min_length=1)
    silent_retries: int | None = Field(default=None, ge=0)

    own_keys: ClassVar[frozenset[str]] = frozenset(
        {"adapter", "fallback", "first_output_timeout", "after_tool", "silent_retries"}
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


class RemindersSettings(BaseModel):
    """The reminder tools (`reminders`, `set_reminder`, `set_timer`, `cancel_reminder`).
    Reminders are kept in the SQLite memory database, so these tools need the SQLite
    memory store on a file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_chars: int = Field(default=6000, ge=200)
    """Longest list handed to the model, in characters."""


class SetReminderSettings(BaseModel):
    """The `set_reminder` tool. Its options are in `[tools.reminders]`."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SetTimerSettings(BaseModel):
    """The `set_timer` tool. Its options are in `[tools.reminders]`."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CancelReminderSettings(BaseModel):
    """The `cancel_reminder` tool. Its options are in `[tools.reminders]`."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class EnrollVoiceSettings(BaseModel):
    """The `enroll_voice` tool (act tier). Needs `[speakers] enabled` and a voice client."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ReplaceVoiceSettings(BaseModel):
    """The `replace_voice` tool (dangerous: asks first). Needs `[speakers] enabled`."""

    model_config = ConfigDict(frozen=True, extra="forbid")


_TOOLS_OPTIONS: Final = ("enabled", "confirm_timeout", "dynamic", "core", "keep_turns")
"""`[tools]` keys that are not tool names."""


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
    dynamic: bool = True
    """Offer the core tools on every turn and the other groups (calendar, reminders,
    an MCP server, ...) only while the conversation needs them (ADR 0025). False
    offers every enabled tool on every turn."""
    core: list[str] | None = None
    """With `dynamic`: the tools always offered, by name (built-in or MCP). Default:
    the web lookups, time and weather, whichever are enabled."""
    keep_turns: int = Field(default=6, ge=1, le=50)
    """With `dynamic`: a group stays offered for this many user turns after it was
    last used or mentioned."""
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
    reminders: RemindersSettings = Field(default_factory=RemindersSettings)
    set_reminder: SetReminderSettings = Field(default_factory=SetReminderSettings)
    set_timer: SetTimerSettings = Field(default_factory=SetTimerSettings)
    cancel_reminder: CancelReminderSettings = Field(default_factory=CancelReminderSettings)
    enroll_voice: EnrollVoiceSettings = Field(default_factory=EnrollVoiceSettings)
    replace_voice: ReplaceVoiceSettings = Field(default_factory=ReplaceVoiceSettings)

    @classmethod
    def known_tools(cls) -> list[str]:
        """The tool names `enabled` accepts."""
        return [name for name in cls.model_fields if name not in _TOOLS_OPTIONS]

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
        core = self.core or []
        twice = sorted({name for name in core if core.count(name) > 1})
        if twice:
            raise ValueError(f"[tools] core lists {', '.join(map(repr, twice))} twice")
        off = [n for n in core if n in known and n not in self.enabled]
        if off:
            raise ValueError(
                f"[tools] core lists {', '.join(map(repr, off))}, not in [tools] enabled"
            )
        if "web_search" in self.enabled and not self.web_search.searxng_url:
            raise ValueError(
                "web_search is enabled but [tools.web_search] has no searxng_url "
                '(e.g. searxng_url = "http://127.0.0.1:8890"; see deploy/searxng/README.md)'
            )
        return self


MCP_NAME_MAX: Final = 64
"""Longest tool name offered to the model (the common limit of chat model APIs)."""

_PLAIN_NAME: Final = re.compile(r"^[a-z0-9_]+$")
_ENV_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOOPBACK_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})

# Launchers that fetch and run a package by name (ADR 0022). The value is the
# subcommand that must follow the command, if any.
_NPM_LAUNCHERS: Final = {"npx": None, "bunx": None, "pnpm": "dlx", "yarn": "dlx"}
_PY_LAUNCHERS: Final = {"uvx": None, "pipx": "run", "uv": "tool run"}
_PACKAGE_OPTIONS: Final = {
    "npm": frozenset({"-p", "--package"}),
    "python": frozenset({"--from", "--spec"}),
}
"""Launcher options whose value is the package to run (then the first positional
argument is the command inside it)."""
_EXTRA_PACKAGE_OPTIONS: Final = {"npm": frozenset[str](), "python": frozenset({"--with"})}
"""Launcher options whose value is a further package installed alongside."""
_SEMVER: Final = r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
_NPM_PINNED: Final = re.compile(rf"^(?:@[a-z0-9][\w.-]*/)?[a-z0-9][\w.-]*@{_SEMVER}$", re.I)
_PEP440: Final = r"\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?"
_PY_PINNED: Final = re.compile(rf"^[a-z0-9][\w.-]*(?:\[[\w,.-]+\])?(?:==|@){_PEP440}$", re.I)


def mcp_tool_name(server: str, tool: str) -> str:
    """The name an MCP tool is offered to the model under, unless aliased with `as`:
    `<server>_<tool>`, lower-cased, each run of characters outside `[a-z0-9_]`
    replaced by one `_`, repeated `_` collapsed and leading/trailing `_` removed.
    The one place this rule lives (ADR 0022); may return "" for an unusable name."""
    name = re.sub(r"[^a-z0-9_]+", "_", f"{server}_{tool}".lower())
    return re.sub(r"_+", "_", name).strip("_")


def _check_preview_template(template: str) -> None:
    """A preview may contain only `{argument_name}` placeholders (and `{{`/`}}`)."""
    try:
        parts = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"preview {template!r} is not a valid template: {exc}") from None
    for _literal, field, spec, conversion in parts:
        if field is None:
            continue
        if not field.isidentifier():
            raise ValueError(
                f"preview {template!r} has placeholder {{{field}}}; placeholders must be "
                "argument names such as {to} (no positions, attributes or indexes)"
            )
        if spec or conversion:
            raise ValueError(
                f"preview {template!r} formats placeholder {{{field}}}; "
                "use plain {name} placeholders without ! or :"
            )


def _is_latest(arg: str) -> bool:
    lowered = arg.lower()
    return lowered == "latest" or lowered.endswith(("@latest", "==latest", ":latest"))


def _launcher_kind(command: str, args: list[str]) -> tuple[str, list[str]] | None:
    """("npm" | "python", the arguments after the launcher) for a package launcher."""
    base = PurePath(command).name
    for kind, launchers in (("npm", _NPM_LAUNCHERS), ("python", _PY_LAUNCHERS)):
        if base not in launchers:
            continue
        sub = launchers[base]
        if sub is None:
            return kind, args
        words = sub.split()
        if args[: len(words)] == words:
            return kind, args[len(words) :]
        return None  # e.g. `uv run` or `pnpm install`: not a package launch
    return None


def _check_pinned(command: str, args: list[str]) -> None:
    """Refuse launches that do not pin an exact version (rule in ADR 0022)."""
    latest = [arg for arg in [command, *args] if _is_latest(arg)]
    if latest:
        raise ValueError(f"launch argument {latest[0]!r} is not a pinned version")
    found = _launcher_kind(command, args)
    if found is None:
        return
    kind, rest = found
    pinned, example = (_NPM_PINNED, "name@1.2.3") if kind == "npm" else (_PY_PINNED, "name==1.2.3")
    specs: list[str] = []
    package: str | None = None
    first_positional: str | None = None
    position = 0
    while position < len(rest):
        arg = rest[position]
        option, _, inline = arg.partition("=")
        if option in _PACKAGE_OPTIONS[kind] or option in _EXTRA_PACKAGE_OPTIONS[kind]:
            value = inline if inline else (rest[position + 1] if position + 1 < len(rest) else "")
            position += 1 if inline else 2
            specs.extend(value.split(","))
            if option in _PACKAGE_OPTIONS[kind]:
                package = value
            continue
        if not arg.startswith("-"):
            first_positional = arg
            break
        position += 1
    if package is None:
        if first_positional is None:
            raise ValueError(f"{command!r} is given no package to run")
        specs.append(first_positional)
    for spec in specs:
        if not pinned.match(spec):
            raise ValueError(
                f"{PurePath(command).name} package {spec!r} must pin an exact version "
                f"(e.g. {example!r}); options before the package must be written "
                "--option=value"
            )


class McpToolSettings(BaseModel):
    """One tool of an MCP server that Lucy may offer to the chat model (ADR 0022).

    Only listed tools are offered. Everything the server says about a tool is
    untrusted; the user's settings here are what Lucy applies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    risk: Risk = Risk.DANGEROUS
    """Risk tier (ADR 0006); unset is dangerous: Lucy asks before every call."""
    alias: str | None = Field(default=None, alias="as")
    """Name offered to the model instead of `<server>_<tool>` (`as` in TOML)."""
    description: str | None = Field(default=None, min_length=1, max_length=200)
    """Replaces the server's description (otherwise its first sentence, at most 200
    characters)."""
    preview: str | None = Field(default=None, min_length=1, max_length=300)
    """The confirmation question for a dangerous tool, with `{argument}` placeholders
    filled from the call's arguments, e.g. "Send an email to {to} about {subject}?"."""
    guidance: str | None = Field(default=None, min_length=1, max_length=300)
    """One sentence added to the tool guidance in the system prompt."""
    fixed: dict[str, StrictBool | StrictInt | StrictFloat | StrictStr] = Field(default_factory=dict)
    """Argument values pinned by the user: hidden from the model's schema and always
    sent with these values, whatever the model passes (e.g. the account address a
    mail server signs in as). A key the server's schema lacks skips the tool."""

    @field_validator("fixed")
    @classmethod
    def _named_fixed(
        cls, fixed: dict[str, bool | int | float | str]
    ) -> dict[str, bool | int | float | str]:
        for key in fixed:
            if not key.strip():
                raise ValueError("fixed argument names must not be empty")
        return fixed

    @field_validator("alias")
    @classmethod
    def _plain_alias(cls, alias: str | None) -> str | None:
        if alias is not None and not (_PLAIN_NAME.match(alias) and len(alias) <= MCP_NAME_MAX):
            raise ValueError(
                f"tool alias {alias!r} must be 1 to {MCP_NAME_MAX} characters of a-z, 0-9 and _"
            )
        return alias

    @field_validator("preview")
    @classmethod
    def _valid_preview(cls, preview: str | None) -> str | None:
        if preview is not None:
            _check_preview_template(preview)
        return preview


class McpServerSettings(BaseModel):
    """One MCP server: a local program Lucy starts (`command`) or a remote address
    (`url`), and the tools of it Lucy may offer (ADR 0022)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str | None = Field(default=None, min_length=1)
    """Local (stdio) server: the program to start. Package launchers (npx, uvx, ...)
    must pin an exact version."""
    args: list[str] = Field(default_factory=list)
    """Arguments for `command`."""
    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables for `command` (not secrets: those go in `env_keychain`).
    The program gets only these, `env_from` and `env_keychain`, plus a minimal base."""
    url: str | None = Field(default=None, pattern=r"^https?://[^\s/]+")
    """Remote server (Streamable HTTP): its address; https unless on this machine."""
    timeout: float = Field(default=30.0, gt=0, le=600)
    """Seconds one tool call may take."""
    max_result_chars: int = Field(default=6000, ge=200)
    """Longest result handed to the model, in characters."""
    private: bool = False
    """The server returns private data (mail, files): while one of its results is in
    what the model sees, calls that could send data elsewhere ask the user first
    (`lucy.core.privacy`)."""
    token_keychain: str | None = Field(default=None, min_length=1)
    """Remote only: the macOS Keychain item (service name) holding the bearer token."""
    env_keychain: dict[str, str] = Field(default_factory=dict)
    """Local only: environment variable name -> Keychain item (service name) whose
    secret it is set to."""
    env_from: list[str] = Field(default_factory=list)
    """Local only: names of Lucy's own environment variables passed on unchanged."""
    tools: dict[str, McpToolSettings] = Field(default_factory=dict)
    """The server's tools Lucy may offer, by the server's tool name."""
    group: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,31}$")
    """The tool group its tools belong to with `[tools] dynamic` (ADR 0025); default
    the server's name."""
    group_description: str | None = Field(default=None, min_length=1, max_length=200)
    """One line telling the model what the group is for; default "tools of NAME"."""
    group_keywords: list[str] = Field(default_factory=list)
    """Words or phrases in the user's message that offer the group; default the
    group's name and the distinctive parts of its tool names."""

    @model_validator(mode="after")
    def _check_transport(self) -> Self:
        if (self.command is None) == (self.url is None):
            raise ValueError("set exactly one of command (a local server) or url (a remote one)")
        if self.url is not None:
            local_only = [
                key for key in ("args", "env", "env_keychain", "env_from") if getattr(self, key)
            ]
            if local_only:
                raise ValueError(
                    f"{', '.join(local_only)} apply only to a local server (command), not url"
                )
            parts = urlsplit(self.url)
            if parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
                raise ValueError(f"url {self.url!r} must use https (http only on this machine)")
            return self
        if self.token_keychain is not None:
            raise ValueError("token_keychain applies only to a remote server (url)")
        names = [*self.env, *self.env_keychain, *self.env_from]
        bad = [name for name in names if not _ENV_NAME.match(name)]
        if bad:
            raise ValueError(f"{bad[0]!r} is not an environment variable name")
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(
                f"environment variable {twice[0]!r} is set more than once "
                "(env, env_keychain, env_from)"
            )
        assert self.command is not None
        _check_pinned(self.command, self.args)
        return self


class McpExposedTool(NamedTuple):
    """An MCP tool as offered to the model: its name there, server and server-side name."""

    name: str
    server: str
    tool: str


class McpSettings(BaseModel):
    """MCP servers whose tools Lucy may use (ADR 0022). Nothing runs unless listed in
    `enabled`, and only tools approved with `lucy mcp approve NAME` are offered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: list[str] = Field(default_factory=list)
    """Names of the servers (tables under `[mcp.servers]`) in use."""
    max_tools: int = Field(default=8, ge=1, le=64)
    """Most MCP tools offered to the model, over all enabled servers."""
    path: str = "data/mcp"
    """Directory for the approval file (`lock.json`) and each server's working
    directory (`<path>/<server>/`)."""
    servers: dict[str, McpServerSettings] = Field(default_factory=dict)

    def exposed_tools(self) -> list[McpExposedTool]:
        """The tools of the enabled servers, in `enabled` then listing order."""
        return [
            McpExposedTool(settings.alias or mcp_tool_name(server, tool), server, tool)
            for server in self.enabled
            if server in self.servers
            for tool, settings in self.servers[server].tools.items()
        ]

    def exposed_tool_names(self) -> list[str]:
        """The names the enabled servers' tools are offered to the model under."""
        return [exposed.name for exposed in self.exposed_tools()]

    @model_validator(mode="after")
    def _check(self) -> Self:
        bad = [name for name in self.servers if not _PLAIN_NAME.match(name)]
        if bad:
            raise ValueError(f"MCP server name {bad[0]!r} must use only a-z, 0-9 and _")
        unknown = [name for name in self.enabled if name not in self.servers]
        if unknown:
            raise ValueError(
                f"[mcp] enabled lists {', '.join(map(repr, unknown))}, which has no "
                f"[mcp.servers.NAME] table; configured: {', '.join(self.servers) or 'none'}"
            )
        twice = sorted({name for name in self.enabled if self.enabled.count(name) > 1})
        if twice:
            raise ValueError(f"[mcp] enabled lists {', '.join(map(repr, twice))} twice")
        builtin = set(ToolsSettings.known_tools())
        owner: dict[str, str] = {}
        for exposed in self.exposed_tools():
            where = f"{exposed.server}.{exposed.tool}"
            if not exposed.name or len(exposed.name) > MCP_NAME_MAX:
                raise ValueError(
                    f"MCP tool {where} gives the name {exposed.name!r}; give it a name "
                    f"with as = ... (1 to {MCP_NAME_MAX} characters of a-z, 0-9 and _)"
                )
            if exposed.name in builtin:
                raise ValueError(
                    f"MCP tool {where} would be named {exposed.name!r}, a built-in tool; "
                    "rename it with as = ..."
                )
            if exposed.name in owner:
                raise ValueError(
                    f"MCP tools {owner[exposed.name]} and {where} would both be named "
                    f"{exposed.name!r}; rename one with as = ..."
                )
            owner[exposed.name] = where
        if len(owner) > self.max_tools:
            raise ValueError(
                f"the enabled MCP servers list {len(owner)} tools, more than max_tools "
                f"({self.max_tools}); list fewer or raise max_tools"
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


class DisplaySettings(BaseModel):
    """How replies are shown in `lucy chat`, `lucy voice` and `lucy listen`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    latin: bool = True
    """Show Devanagari in replies in Latin letters (`lucy.voice.translit.to_latin`). The
    stored conversation and the speech keep the Devanagari the model wrote (ADR 0024)."""


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


class WorkspaceSettings(BaseModel):
    """Lucy's own directory for files her code writes and reads (`lucy.core.workspace`).

    Not reachable from the chat model: no tool reads or writes it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = "data/workspace"


class SpeakersSettings(BaseModel):
    """Who is speaking (stage 1: each voice utterance is tagged; nothing else changes).

    A voice match is a label, never an authorisation: a recording of an enrolled
    speaker scores as high as the speaker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    weights: str = "data/models/speaker/voxceleb_resnet34_LM.onnx"
    """The WeSpeaker ResNet34 ONNX model; `lucy speakers download` fetches it."""
    threshold: float = Field(default=0.40, gt=-1, lt=1)
    """Cosine score from which a voice counts as an enrolled speaker. The library's
    default; calibrate it on your own clips."""
    wait_ms: int = Field(default=150, ge=0, le=5000)
    """How long a turn waits for identification after transcription is done."""
    keep_utterances_days: float = Field(default=1.0, ge=0)
    """How long utterance clips are kept; 0 deletes each right after identification."""
    max_utterances_mb: float = Field(default=200.0, ge=0)
    """Size cap for kept utterance clips; the oldest go first."""
    enrol_sentences: int = Field(default=5, ge=3, le=8)
    """How many sentences a voice enrolment asks for (`enroll_voice`, `replace_voice`)."""


class ContextSettings(BaseModel):
    """How the chat prompt is arranged (see `lucy.core.context`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layout: ContextLayout = "stable_prefix"
    """`"system"`: the date and retrieved memories in the system message, before the
    history. `"stable_prefix"`: only what rarely changes before the history, and the
    date and memories with the latest user message, so each turn's prompt extends
    the previous one and the model server can reuse its cache."""

    history_tokens: int | None = Field(default=1200, ge=0)
    """Most estimated tokens (characters / 4) of conversation before the latest user
    message that a prompt carries, whatever the chat model's `context_window`;
    None leaves only the window's share. Older messages are dropped whole (a tool
    call with its results). Every token of the prompt's prefix is processed again
    when the residency policy re-warms the model (about 850 tokens/s with
    gemma4:12b-mlx), so this bounds that cost. Measured 2026-11-17: without it the
    200-event window carried about 4,900 tokens of history."""


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


DEFAULT_MEMORY: Final = AdapterConfig(adapter="sqlite", options={"path": "data/lucy.db"})
"""The memory store used when the settings have no `[memory]` section."""


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
    mcp: McpSettings = Field(default_factory=McpSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    display: DisplaySettings = Field(default_factory=DisplaySettings)
    wake: WakeSettings = Field(default_factory=WakeSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    residency: ResidencySettings = Field(default_factory=ResidencySettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    speakers: SpeakersSettings = Field(default_factory=SpeakersSettings)


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
