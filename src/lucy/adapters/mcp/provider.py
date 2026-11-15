"""`McpToolProvider`: the approved tools of every enabled MCP server (ADR 0022, run 5).

- **Tools from the lock only.** At construction the provider reads the lock file
  (no connection, no process): a tool is offered when it is listed in the settings
  and approved in the lock under the command/URL configured now. Anything else waits
  for `lucy mcp approve NAME`; one startup notice per server says how many.
- **Live check before the first call.** Each server starts on its first tool call.
  Before that call is sent (and again after every restart, since the program may
  have been updated), the provider lists the server's tools and compares them with
  the lock. A tool whose live definition no longer matches its pin refuses every
  call with a plain message naming the re-approval command; unchanged tools work.
  The model's tool list is fixed for the process lifetime, so "disabled" means the
  tool refuses at call time. The check costs one extra local round trip on the
  first call per start, and closes the window in which a changed tool could run.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lucy.adapters.mcp.lock import (
    LockFile,
    PinnedServer,
    approve_command,
    approved_tools,
    review,
)
from lucy.adapters.mcp.sdk import CallResult, McpError
from lucy.adapters.mcp.secrets import Secrets
from lucy.adapters.mcp.server import McpServer
from lucy.adapters.mcp.tool import McpRefusedError, McpTool, build_mcp_tools
from lucy.core.config import McpSettings
from lucy.core.ports import Tool

__all__ = ["McpToolProvider", "PinnedCaller", "guidance_sentence"]

logger = logging.getLogger(__name__)

CLOSE_TIMEOUT = 10.0
"""Longest wait for all servers to close (each bounds itself; this is a backstop)."""
LIST_TIMEOUT = 15.0


def guidance_sentence(tool_name: str, text: str) -> str:
    """A user's guidance sentence in the "the NAME tool" form the prompt filter keys
    on: one line, ending in a full stop, prefixed when it does not name its tool."""
    line = " ".join(text.split()).rstrip()
    if f"the {tool_name} tool" not in line:
        line = f"About the {tool_name} tool: {line}"
    if not line.endswith((".", "!", "?")):
        line += "."
    return line


class PinnedCaller:
    """An `McpCaller` over one `McpServer` that refuses tools whose live definition
    differs from the lock. The comparison runs before the first call of each start."""

    def __init__(self, server: McpServer, pinned: PinnedServer | None) -> None:
        self.server = server
        self._pinned = pinned
        self._checked_start: int | None = None
        self._refused: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @property
    def refused(self) -> Mapping[str, str]:
        return dict(self._refused)

    async def _verify(self) -> None:
        await self.server.ensure_started()
        if self._checked_start == self.server.start_count:
            return
        async with self._lock:
            if self._checked_start == self.server.start_count:
                return
            live = await self.server.list_tools(LIST_TIMEOUT)
            result = review(self._pinned, self.server.settings, live)
            refused: dict[str, str] = {}
            for item in result.tools:
                if item.state in ("changed", "gone"):
                    what = ", ".join(item.changes) or "its definition"
                    if item.state == "gone":
                        message = (
                            f"The {self.server.name} service no longer offers this tool. "
                            f"Run `{approve_command(self.server.name)}` to review it."
                        )
                    else:
                        message = (
                            f"The {self.server.name} service changed this tool ({what}) "
                            "since you approved it, so Lucy won't use it until you approve "
                            f"it again with `{approve_command(self.server.name)}`."
                        )
                    refused[item.name] = message
                    logger.warning(
                        "MCP server %s: tool %s %s since approval; refusing calls to it",
                        self.server.name,
                        item.name,
                        "is gone" if item.state == "gone" else f"changed ({what})",
                    )
            self._refused = refused
            self._checked_start = self.server.start_count

    async def call(self, tool: str, arguments: Mapping[str, Any], timeout: float) -> CallResult:
        await self._verify()
        if tool in self._refused:
            raise McpRefusedError(self._refused[tool])
        return await self.server.call(tool, arguments, timeout)


class McpToolProvider:
    """A `ToolProvider` for the enabled servers in `[mcp]`."""

    def __init__(
        self,
        settings: McpSettings,
        *,
        secrets: Secrets,
        root: str | Path | None = None,
        server_options: Mapping[str, Any] | None = None,
    ) -> None:
        self._settings = settings
        root = settings.path if root is None else root  # tests inject a temporary root
        lock = LockFile.load(root)  # LockError propagates: a bad lock stops startup
        self.servers: dict[str, McpServer] = {}
        self.callers: dict[str, PinnedCaller] = {}
        self._tools: list[McpTool] = []
        self._guidance: list[str] = []
        for name in settings.enabled:
            server_settings = settings.servers[name]
            server = McpServer(
                name, server_settings, secrets=secrets, root=root, **dict(server_options or {})
            )
            pinned = lock.servers.get(name)
            caller = PinnedCaller(server, pinned)
            self.servers[name] = server
            self.callers[name] = caller
            remote = approved_tools(pinned, server_settings)
            tools, skipped = build_mcp_tools(name, server_settings, remote, caller)
            approved = {tool.remote_name for tool in tools}
            for skip in skipped:
                if skip.tool in {r.name for r in remote}:  # approved but unusable
                    logger.warning("MCP server %s: tool %s not offered: %s", name, *skip)
            waiting = [t for t in server_settings.tools if t not in approved]
            if waiting:
                logger.warning(
                    "MCP server %s: %d tool%s waiting for approval (%s); run `%s` to review",
                    name,
                    len(waiting),
                    "" if len(waiting) == 1 else "s",
                    ", ".join(waiting),
                    approve_command(name),
                )
            for tool in tools:
                text = server_settings.tools[tool.remote_name].guidance
                if text:
                    self._guidance.append(guidance_sentence(tool.name, text))
            self._tools.extend(tools)
        if len(self._tools) > settings.max_tools:
            logger.warning(
                "MCP: %d approved tools, offering the first %d (max_tools)",
                len(self._tools),
                settings.max_tools,
            )
            kept = {t.name for t in self._tools[: settings.max_tools]}
            self._tools = self._tools[: settings.max_tools]
            self._guidance = [g for g in self._guidance if any(f"the {n} tool" in g for n in kept)]

    def tools(self) -> list[Tool]:
        return list(self._tools)

    def guidance(self) -> list[str]:
        """Prompt sentences for the exposed tools, each naming "the NAME tool"."""
        return list(self._guidance)

    async def start(self) -> None:
        """Cheap: servers start on their first tool call."""

    async def aclose(self) -> None:
        servers = list(self.servers.values())
        if not servers:
            return
        try:
            async with asyncio.timeout(CLOSE_TIMEOUT):
                results = await asyncio.gather(
                    *(s.aclose() for s in servers), return_exceptions=True
                )
        except TimeoutError:
            logger.warning("MCP: servers did not all close within %.0f s", CLOSE_TIMEOUT)
            return
        for server, result in zip(servers, results, strict=True):
            if isinstance(result, BaseException) and not isinstance(result, McpError):
                logger.warning("MCP server %s: close failed: %s", server.name, result)

    def diagnostics(self) -> dict[str, dict[str, Any]]:
        """Per server: lifecycle state, last error, starts, refused tools."""
        return {
            name: {
                "state": server.state,
                "last_error": server.last_error,
                "starts": server.start_count,
                "refused": sorted(self.callers[name].refused),
            }
            for name, server in self.servers.items()
        }
