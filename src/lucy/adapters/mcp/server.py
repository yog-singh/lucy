"""One MCP server's lifecycle: lazy start, an owner task, restart, shutdown (ADR 0022).

`McpServer` implements `tool.McpCaller` for one configured server (stdio or HTTP).

- **Owner task.** The SDK requires a connection to be opened and closed in the same
  task, so each connection has one owner task that opens it, serves requests from a
  queue and closes it. Callers in any task put a request on the queue and await its
  future. Requests to one server run **one at a time** (the owner awaits each before
  taking the next): simpler and safe, and a voice assistant rarely needs two calls to
  one server at once.
- **Restart.** A dropped connection or exited process marks the server `down`. The
  next call starts it again unless the last start attempt was less than
  `restart_backoff` seconds ago; inside that window calls fail fast with
  `McpConnectError`. A call in flight when the server died is not re-sent here: its
  error has `request_sent=True` and the tool layer decides (READ tools retry once).
- **Shutdown.** The owner closes the connection; if a stdio server has not exited
  `close_grace` seconds after its stdin closed, its process group gets SIGTERM, then
  SIGKILL. After the close, any member of the group still alive (the server's own
  children) gets the same. A process that leaves the group (`setsid`) escapes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import signal
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from lucy.adapters.mcp import sdk
from lucy.adapters.mcp.sdk import (
    CallResult,
    HttpServerSpec,
    McpConnectError,
    McpConnection,
    McpError,
    McpTimeoutError,
    RemoteTool,
    ServerInfo,
    ServerProcess,
    StdioServerSpec,
)
from lucy.adapters.mcp.secrets import SecretError, Secrets
from lucy.core.config import McpServerSettings
from lucy.core.workspace import Workspace

__all__ = ["STDERR_LINE_CHARS", "McpServer", "State", "child_environment"]

logger = logging.getLogger(__name__)

State = Literal["stopped", "starting", "up", "down"]

RESTART_BACKOFF = 30.0
"""Least seconds between two start attempts of one server."""
CLOSE_GRACE = 2.0
"""Seconds a stdio server has to exit after its stdin closes before SIGTERM."""
KILL_GRACE = 1.0
"""Seconds between SIGTERM and SIGKILL to a process group."""
BACKSTOP = 2.0
"""Extra seconds a caller waits beyond its timeout before giving up on the owner."""
STDERR_LINE_CHARS = 300
STDERR_LINES_PER_WINDOW = 50
STDERR_WINDOW = 10.0

_CONTROL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f-\x9f​-‏ -‮⁠-⁤﻿]")


def child_environment(
    name: str,
    settings: McpServerSettings,
    environ: Mapping[str, str],
    secrets: Mapping[str, str],
) -> dict[str, str]:
    """The variables Lucy passes to a stdio server: `env`, `env_from` copied from
    `environ`, and the Keychain secrets. The SDK adds HOME, LOGNAME, PATH, SHELL,
    TERM and USER from Lucy's environment underneath (a configured value wins)."""
    missing = [key for key in settings.env_from if key not in environ]
    if missing:
        raise McpConnectError(
            f"MCP server {name!r}: env_from names {', '.join(missing)}, "
            "which is not set in Lucy's environment"
        )
    env = dict(settings.env)
    env.update({key: environ[key] for key in settings.env_from})
    env.update(secrets)
    return env


@dataclass(eq=False)
class _Request:
    kind: Literal["call", "list"]
    tool: str
    arguments: dict[str, Any] = field(repr=False)
    deadline: float
    future: asyncio.Future[Any] = field(repr=False)


_STOP = object()


class McpServer:
    """One configured MCP server, started on first use. Implements `McpCaller`."""

    def __init__(
        self,
        name: str,
        settings: McpServerSettings,
        *,
        secrets: Secrets,
        root: str | Path,
        environ: Mapping[str, str] | None = None,
        restart_backoff: float = RESTART_BACKOFF,
        connect_timeout: float = sdk.DEFAULT_CONNECT_TIMEOUT,
        close_grace: float = CLOSE_GRACE,
        clock: Callable[[], float] = time.monotonic,
        connect_stdio: Callable[..., Awaitable[McpConnection]] = sdk.connect_stdio,
        connect_http: Callable[..., Awaitable[McpConnection]] = sdk.connect_http,
    ) -> None:
        self.name = name
        self.settings = settings
        self._secrets = secrets
        self._root = Path(root)
        self._environ = environ
        self._restart_backoff = restart_backoff
        self._connect_timeout = connect_timeout
        self._close_grace = close_grace
        self._clock = clock
        self._connect_stdio = connect_stdio
        self._connect_http = connect_http

        self.state: State = "stopped"
        self.info: ServerInfo | None = None
        self.last_error: str | None = None
        self.start_count = 0
        self._last_attempt: float | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[Any] | None = None
        self._ready: asyncio.Future[ServerInfo] | None = None
        self._stop: asyncio.Event | None = None
        self._process: ServerProcess | None = None

    def __repr__(self) -> str:
        return f"McpServer({self.name!r}, state={self.state!r}, starts={self.start_count})"

    @property
    def pid(self) -> int | None:
        """The stdio server's process id while it runs."""
        return self._process.pid if self._process is not None else None

    # --- Public API ------------------------------------------------------------

    async def ensure_started(self) -> ServerInfo:
        """Start the server if it is not up (respecting the restart backoff)."""
        await self._ensure()
        assert self.info is not None
        return self.info

    async def list_tools(self, timeout: float = 30.0) -> list[RemoteTool]:
        return await self._submit("list", "", {}, timeout)

    async def call(self, tool: str, arguments: Mapping[str, Any], timeout: float) -> CallResult:
        return await self._submit("call", tool, dict(arguments), timeout)

    async def aclose(self) -> None:
        """Stop the server (and its process group). Idempotent; requests waiting or
        in flight fail with `McpConnectError`."""
        self._closed = True
        owner = self._owner
        if owner is not None and not owner.done():
            assert self._stop is not None and self._ready is not None
            if self._ready.done():
                self._stop.set()
            else:
                owner.cancel()  # still connecting: the owner cleans up the process
            await asyncio.gather(owner, return_exceptions=True)
        ready = self._ready
        if ready is not None and not ready.done():  # cancelled before it ever ran
            ready.set_exception(McpConnectError(f"MCP server {self.name!r} was shut down"))
        self.state = "stopped"

    # --- Requests --------------------------------------------------------------

    async def _submit(self, kind: Literal["call", "list"], tool: str, args: dict, timeout: float):
        await self._ensure()
        queue = self._queue
        assert queue is not None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        queue.put_nowait(_Request(kind, tool, args, loop.time() + timeout, future))
        try:
            async with asyncio.timeout(timeout + BACKSTOP):
                return await future
        except TimeoutError:
            exc = McpTimeoutError(f"MCP server {self.name!r} did not answer in time")
            exc.request_sent = True
            raise exc from None

    async def _ensure(self) -> None:
        if self._closed:
            raise McpConnectError(f"MCP server {self.name!r} is shut down")
        if self.state == "up" and self._process is not None and self._process.exited():
            await self._retire("the server exited")
        if self.state == "up":
            return
        async with self._lock:
            if self._closed:
                raise McpConnectError(f"MCP server {self.name!r} is shut down")
            if self.state == "up":
                return
            now = self._clock()
            if self._last_attempt is not None and now - self._last_attempt < self._restart_backoff:
                wait = self._restart_backoff - (now - self._last_attempt)
                raise McpConnectError(
                    f"MCP server {self.name!r} is down ({self.last_error or 'not started'}); "
                    f"next start attempt in {wait:.0f} s"
                )
            self._last_attempt = now
            self.start_count += 1
            self.state = "starting"
            loop = asyncio.get_running_loop()
            self._queue = asyncio.Queue()
            self._stop = asyncio.Event()
            self._ready = ready = loop.create_future()
            self._owner = asyncio.create_task(
                self._run(self._queue, self._stop, ready), name=f"lucy-mcp-{self.name}"
            )
            # Shielded: a cancelled caller does not cancel the start other callers share.
            await asyncio.shield(ready)

    async def _retire(self, reason: str) -> None:
        """Stop the current owner (its server died) and mark the server down."""
        owner = self._owner
        self.last_error = reason
        if owner is not None and not owner.done() and self._stop is not None:
            self._stop.set()
            await asyncio.gather(owner, return_exceptions=True)
        if not self._closed:
            self.state = "down"

    # --- Owner task ------------------------------------------------------------

    async def _run(
        self, queue: asyncio.Queue[Any], stop: asyncio.Event, ready: asyncio.Future[ServerInfo]
    ) -> None:
        conn: McpConnection | None = None
        processes: list[ServerProcess] = []
        stderr: _StderrPump | None = None
        current: _Request | None = None
        try:
            try:
                if self.settings.url is not None:
                    conn = await self._open_http()
                else:
                    stderr = _StderrPump(self.name)

                    def spawned(process: ServerProcess) -> None:
                        processes.append(process)
                        self._process = process
                        stderr.close_writer()

                    conn = await self._open_stdio(stderr, spawned)
            except asyncio.CancelledError:
                if not ready.done():
                    ready.set_exception(McpConnectError(f"MCP server {self.name!r} was shut down"))
                raise
            except McpError as exc:
                self._fail(str(exc))
                ready.set_exception(exc)
                return
            except Exception as exc:  # a broken SDK must not leave callers waiting
                self._fail(f"could not start: {type(exc).__name__}")
                ready.set_exception(McpConnectError(f"MCP server {self.name!r} could not start"))
                return
            self.info = conn.info
            self.state = "up"
            self.last_error = None
            logger.info(
                "MCP server %s started (%s, %s)",
                self.name,
                conn.info.era,
                conn.info.protocol_version,
            )
            ready.set_result(conn.info)
            stopping = asyncio.ensure_future(stop.wait())
            try:
                while True:
                    getter = asyncio.ensure_future(queue.get())
                    await asyncio.wait({getter, stopping}, return_when=asyncio.FIRST_COMPLETED)
                    if not getter.done():
                        getter.cancel()
                        break
                    current = getter.result()
                    if current.future.done():  # the caller gave up
                        current = None
                        continue
                    work = asyncio.ensure_future(self._serve(conn, current))
                    await asyncio.wait({work, stopping}, return_when=asyncio.FIRST_COMPLETED)
                    if not work.done():
                        work.cancel()
                        await asyncio.gather(work, return_exceptions=True)
                        _settle(current, _gone(self.name, "was shut down", sent=True))
                        break
                    alive = work.result()
                    current = None
                    if not alive:
                        break
            finally:
                stopping.cancel()
        finally:
            if current is not None:
                _settle(current, _gone(self.name, "stopped", sent=True))
            # In this task, not a shielded copy: the SDK must close where it opened.
            await self._shutdown(conn, processes, stderr)
            while not queue.empty():
                item = queue.get_nowait()
                if isinstance(item, _Request):
                    _settle(item, _gone(self.name, "stopped", sent=False))
            self._process = None
            if self._closed:
                self.state = "stopped"
            elif self.state in ("up", "starting"):
                self.state = "down"
                self.last_error = self.last_error or "the server stopped"

    async def _serve(self, conn: McpConnection, req: _Request) -> bool:
        """Run one request; False if the connection is gone."""
        remaining = req.deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            _settle(req, McpTimeoutError(f"MCP server {self.name!r} did not answer in time"))
            return True
        try:
            async with asyncio.timeout(remaining + 1.0):  # backstop over the SDK's own
                if req.kind == "list":
                    result: Any = await conn.list_tools(timeout=remaining)
                else:
                    result = await conn.call_tool(req.tool, req.arguments, timeout=remaining)
        except TimeoutError:
            exc: McpError = McpTimeoutError(f"MCP server {self.name!r} did not answer in time")
            exc.request_sent = True
            _settle(req, exc)
            return True
        except McpConnectError as exc:
            exc.request_sent = True
            self._fail(f"the connection dropped: {exc}")
            _settle(req, exc)
            return False
        except McpError as exc:
            exc.request_sent = True
            _settle(req, exc)
            return True
        except Exception as exc:
            self._fail(f"unexpected error: {type(exc).__name__}")
            _settle(req, _gone(self.name, "failed", sent=True))
            return False
        if not req.future.done():
            req.future.set_result(result)
        return True

    def _fail(self, reason: str) -> None:
        self.last_error = reason
        if not self._closed:
            self.state = "down"
        logger.warning("MCP server %s: %s", self.name, reason)

    async def _open_http(self) -> McpConnection:
        token = None
        if self.settings.token_keychain is not None:
            token = await self._secret(self.settings.token_keychain)
        spec = HttpServerSpec(url=self.settings.url or "", bearer_token=token)
        return await self._connect_http(spec, connect_timeout=self._connect_timeout)

    async def _open_stdio(
        self, stderr: _StderrPump, spawned: Callable[[ServerProcess], None]
    ) -> McpConnection:
        secrets = {
            var: await self._secret(service) for var, service in self.settings.env_keychain.items()
        }
        environ = os.environ if self._environ is None else self._environ
        env = child_environment(self.name, self.settings, environ, secrets)
        workdir = Workspace(self._root).dir(self.name)
        spec = StdioServerSpec(
            command=self.settings.command or "",
            args=tuple(self.settings.args),
            env=env,
            cwd=workdir,
        )
        return await self._connect_stdio(
            spec,
            connect_timeout=self._connect_timeout,
            errlog=stderr.writer,
            on_spawn=spawned,
        )

    async def _secret(self, service: str) -> str:
        try:
            return await self._secrets.get(service)
        except SecretError as exc:
            raise McpConnectError(f"MCP server {self.name!r}: {exc}") from None

    async def _shutdown(
        self,
        conn: McpConnection | None,
        processes: list[ServerProcess],
        stderr: _StderrPump | None,
    ) -> None:
        process = processes[0] if processes else None
        pgid = _group_of(process)
        if conn is not None:
            killer = None
            if process is not None and pgid is not None:
                killer = asyncio.ensure_future(self._kill_after(process, pgid))
            try:
                await conn.aclose()
            except Exception as exc:
                logger.debug("MCP server %s: close failed: %s", self.name, type(exc).__name__)
            finally:
                if killer is not None:
                    killer.cancel()
                    await asyncio.gather(killer, return_exceptions=True)
        if pgid is not None:
            await _kill_group(pgid)
        if stderr is not None:
            stderr.close()

    async def _kill_after(self, process: ServerProcess, pgid: int) -> None:
        await asyncio.sleep(self._close_grace)
        if not process.exited():
            logger.debug("MCP server %s did not exit after stdin closed; stopping it", self.name)
            await _kill_group(pgid)


def _settle(req: _Request, exc: BaseException) -> None:
    if not req.future.done():
        req.future.set_exception(exc)


def _gone(name: str, what: str, *, sent: bool) -> McpConnectError:
    exc = McpConnectError(f"MCP server {name!r} {what}")
    exc.request_sent = sent
    return exc


# --- Process groups --------------------------------------------------------------


def _group_of(process: ServerProcess | None) -> int | None:
    """The server's own process group (never Lucy's)."""
    if process is None:
        return None
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        pgid = process.pid  # already gone: the SDK starts it as a session leader
    except OSError:
        return None
    return pgid if pgid != os.getpgrp() else None


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _kill_group(pgid: int) -> None:
    """SIGTERM every live member of `pgid`, then SIGKILL after `KILL_GRACE`."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not _group_alive(pgid):
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)
        deadline = time.monotonic() + KILL_GRACE
        while _group_alive(pgid) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)


# --- stderr ----------------------------------------------------------------------


class _StderrPump:
    """A pipe for the server's stderr, read by a thread into Lucy's debug log:
    prefixed with the server name, control characters removed, lines capped at
    `STDERR_LINE_CHARS`, at most `STDERR_LINES_PER_WINDOW` lines per
    `STDERR_WINDOW` seconds (a count of dropped lines is logged)."""

    def __init__(self, name: str) -> None:
        self._name = name
        read_fd, write_fd = os.pipe()
        self.writer = os.fdopen(write_fd, "w")
        self._reader = os.fdopen(read_fd, "rb")
        self._thread = threading.Thread(
            target=self._pump, name=f"lucy-mcp-stderr-{name}", daemon=True
        )
        self._thread.start()

    def close_writer(self) -> None:
        """Drop Lucy's copy of the write end (the child has its own)."""
        with contextlib.suppress(OSError):
            self.writer.close()

    def close(self) -> None:
        self.close_writer()
        self._thread.join(timeout=1.0)

    def _pump(self) -> None:
        window_start = time.monotonic()
        lines = dropped = 0
        try:
            for raw in self._reader:
                now = time.monotonic()
                if now - window_start >= STDERR_WINDOW:
                    if dropped:
                        logger.debug("[mcp %s stderr] %d lines dropped", self._name, dropped)
                    window_start, lines, dropped = now, 0, 0
                if lines >= STDERR_LINES_PER_WINDOW:
                    dropped += 1
                    continue
                lines += 1
                text = _CONTROL.sub("", raw[: STDERR_LINE_CHARS * 4].decode("utf-8", "replace"))
                text = text.strip()[:STDERR_LINE_CHARS]
                if text:
                    logger.debug("[mcp %s stderr] %s", self._name, text)
            if dropped:
                logger.debug("[mcp %s stderr] %d lines dropped", self._name, dropped)
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError):
                self._reader.close()
