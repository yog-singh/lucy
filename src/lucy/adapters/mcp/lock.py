"""The MCP lock file: what the user approved, per server (ADR 0022, run 5).

`<[mcp] path>/lock.json` (the same root `McpServer` uses) records, per server,
the launch command and arguments (or URL) as approved, what the server reported
about itself, the negotiated protocol and the SDK version at approval time, and
per approved tool the full `RemoteTool` with a hash.

The hash covers what reaches the model or decides behaviour: the tool's name,
title (Lucy speaks it in the generic confirmation question and uses it in the
fallback description), description and input schema. Annotations and the output
schema are pinned for display but not hashed: Lucy never applies annotations
(risk is the user's) and does not use the output schema, so a change to either
cannot change what the model sees or what a call does.

Written as stable JSON (sorted keys), atomically, owner-only (0600). A missing
file is an empty lock; a corrupt file or one written by a newer Lucy is refused.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from lucy.adapters.mcp.sdk import RemoteTool, ServerInfo
from lucy.core.config import McpServerSettings

__all__ = [
    "LOCK_FORMAT",
    "LockError",
    "LockFile",
    "PinnedServer",
    "PinnedTool",
    "ServerReview",
    "ToolReview",
    "approve_command",
    "approved_tools",
    "launch_of",
    "lock_path",
    "review",
    "tool_hash",
]

LOCK_NAME: Final = "lock.json"
LOCK_FORMAT: Final = 1
"""Bumped when the layout changes; a newer file is refused, not guessed at."""

ToolState = Literal["approved", "changed", "new", "gone"]


class LockError(Exception):
    """The lock file cannot be read or written (message in plain words)."""


def lock_path(root: str | Path) -> Path:
    return Path(root) / LOCK_NAME


def approve_command(server: str) -> str:
    return f"lucy mcp approve {server}"


def launch_of(settings: McpServerSettings) -> dict[str, Any]:
    """The part of a server's settings the approval pins: command and args, or URL."""
    if settings.url is not None:
        return {"url": settings.url}
    return {"command": settings.command, "args": list(settings.args)}


def _tool_dict(tool: RemoteTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "title": tool.title,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "output_schema": tool.output_schema,
        "annotations": tool.annotations,
    }


def _tool_from(data: Mapping[str, Any]) -> RemoteTool:
    return RemoteTool(
        name=str(data["name"]),
        title=data.get("title"),
        description=data.get("description"),
        input_schema=dict(data.get("input_schema") or {}),
        output_schema=data.get("output_schema"),
        annotations=dict(data.get("annotations") or {}),
    )


def tool_hash(tool: RemoteTool) -> str:
    """sha256 over name, title, description and input schema (canonical JSON)."""
    hashed = {
        "name": tool.name,
        "title": tool.title,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }
    text = json.dumps(hashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PinnedTool:
    tool: RemoteTool
    hash: str


@dataclass(frozen=True, slots=True)
class PinnedServer:
    launch: dict[str, Any]
    server_name: str | None
    server_version: str | None
    protocol_version: str | None
    era: str | None
    sdk_version: str | None
    approved_at: str
    tools: dict[str, PinnedTool] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "launch": self.launch,
            "server_name": self.server_name,
            "server_version": self.server_version,
            "protocol_version": self.protocol_version,
            "era": self.era,
            "sdk_version": self.sdk_version,
            "approved_at": self.approved_at,
            "tools": {
                name: {"hash": pin.hash, "tool": _tool_dict(pin.tool)}
                for name, pin in self.tools.items()
            },
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PinnedServer:
        return cls(
            launch=dict(data["launch"]),
            server_name=data.get("server_name"),
            server_version=data.get("server_version"),
            protocol_version=data.get("protocol_version"),
            era=data.get("era"),
            sdk_version=data.get("sdk_version"),
            approved_at=str(data["approved_at"]),
            tools={
                name: PinnedTool(_tool_from(entry["tool"]), str(entry["hash"]))
                for name, entry in dict(data.get("tools") or {}).items()
            },
        )

    @classmethod
    def approve(
        cls,
        settings: McpServerSettings,
        info: ServerInfo,
        sdk_version: str,
        tools: Iterable[RemoteTool],
        *,
        now: datetime | None = None,
    ) -> PinnedServer:
        moment = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
        return cls(
            launch=launch_of(settings),
            server_name=info.name,
            server_version=info.version,
            protocol_version=info.protocol_version,
            era=info.era,
            sdk_version=sdk_version,
            approved_at=moment.isoformat(),
            tools={tool.name: PinnedTool(tool, tool_hash(tool)) for tool in tools},
        )


class LockFile:
    """The approvals for every server, loaded from and saved to `path`."""

    def __init__(self, path: Path, servers: dict[str, PinnedServer] | None = None) -> None:
        self.path = path
        self.servers: dict[str, PinnedServer] = servers or {}

    @classmethod
    def load(cls, root: str | Path) -> LockFile:
        path = lock_path(root)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls(path)
        except OSError as exc:
            raise LockError(
                f"Lucy couldn't read the MCP lock file {path}: {exc.strerror}"
            ) from None
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("not an object")
            fmt = data.get("format")
            if isinstance(fmt, int) and fmt > LOCK_FORMAT:
                raise LockError(
                    f"The MCP lock file {path} was written by a newer Lucy (format {fmt}; "
                    f"this one reads format {LOCK_FORMAT}). Update Lucy, or move the file "
                    "away and approve the servers again."
                )
            if fmt != LOCK_FORMAT:
                raise ValueError(f"unknown format {fmt!r}")
            servers = {
                str(name): PinnedServer.from_json(entry)
                for name, entry in dict(data.get("servers") or {}).items()
            }
        except LockError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise LockError(
                f"The MCP lock file {path} is damaged ({exc}). Move it away and approve "
                "the servers again with `lucy mcp approve NAME`."
            ) from None
        return cls(path, servers)

    def to_text(self) -> str:
        data = {
            "format": LOCK_FORMAT,
            "servers": {name: pin.to_json() for name, pin in self.servers.items()},
        }
        return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    def save(self) -> None:
        """Write atomically (temporary file, fsync, rename), owner-only."""
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(prefix=".lock-", suffix=".json", dir=directory)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(self.to_text())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


# --- Comparison ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolReview:
    name: str
    state: ToolState
    changes: tuple[str, ...] = ()
    """For `changed`: what differs, in plain words."""
    pinned: RemoteTool | None = None
    live: RemoteTool | None = None


@dataclass(frozen=True, slots=True)
class ServerReview:
    launch_changed: bool
    tools: tuple[ToolReview, ...]
    unconfigured: tuple[str, ...]
    """Tools the server offers that the settings do not list (informational)."""

    def by_state(self, state: ToolState) -> list[ToolReview]:
        return [t for t in self.tools if t.state == state]

    @property
    def clean(self) -> bool:
        return not self.launch_changed and all(t.state == "approved" for t in self.tools)


def _differences(old: RemoteTool, new: RemoteTool) -> tuple[str, ...]:
    changes = []
    if old.description != new.description:
        changes.append("description")
    if old.input_schema != new.input_schema:
        changes.append("input schema")
    if old.title != new.title:
        changes.append("title")
    return tuple(changes)


def review(
    pinned: PinnedServer | None,
    settings: McpServerSettings,
    live: Sequence[RemoteTool] | None,
) -> ServerReview:
    """Classify each configured tool against the lock. `live` is the server's tool list
    (None: not connected, so pinned tools count as approved and nothing is `gone`).
    A changed command, arguments or URL makes every configured tool need approval."""
    launch_changed = pinned is not None and pinned.launch != launch_of(settings)
    offered = {tool.name: tool for tool in live} if live is not None else None
    reviews: list[ToolReview] = []
    for name in settings.tools:
        pin = pinned.tools.get(name) if pinned is not None and not launch_changed else None
        current = offered.get(name) if offered is not None else None
        if pin is None:  # never approved (live is None when not offered either)
            reviews.append(ToolReview(name, "new", live=current))
        elif offered is None:
            reviews.append(ToolReview(name, "approved", pinned=pin.tool))
        elif current is None:
            reviews.append(ToolReview(name, "gone", pinned=pin.tool))
        elif tool_hash(current) == pin.hash:
            reviews.append(ToolReview(name, "approved", pinned=pin.tool, live=current))
        else:
            reviews.append(
                ToolReview(name, "changed", _differences(pin.tool, current), pin.tool, current)
            )
    unconfigured = tuple(n for n in (offered or {}) if n not in settings.tools)
    return ServerReview(launch_changed, tuple(reviews), unconfigured)


def approved_tools(pinned: PinnedServer | None, settings: McpServerSettings) -> list[RemoteTool]:
    """The pinned tools that may be offered without connecting: listed in the
    settings and approved under the same command/URL as configured now."""
    if pinned is None or pinned.launch != launch_of(settings):
        return []
    return [pinned.tools[name].tool for name in settings.tools if name in pinned.tools]
