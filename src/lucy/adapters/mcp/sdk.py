"""The one module that imports the `mcp` SDK (ADR 0022).

Everything public here is Lucy-owned: dataclasses, exceptions and a small async
connection API. No SDK type or exception crosses this boundary, so the rest of
Lucy works without importing `mcp`, and an SDK upgrade touches only this file
(checked by the contract tests in `tests/mcp/`).

This module is deliberately thin: it does not decide risk, naming, truncation or
prompt text. Protocol versions are negotiated by the SDK; Lucy never names one.

A connection must be opened and closed in the same asyncio task (the SDK holds
anyio task groups and cancel scopes for its lifetime).
"""

from __future__ import annotations

import contextvars
import os
from collections import Counter
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import anyio
import httpx2
import mcp.client.stdio as _sdk_stdio
import mcp_types
from mcp import Client, InputRequiredRoundsExceededError
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import ValidationError

__all__ = [
    "CallResult",
    "Era",
    "HttpServerSpec",
    "McpConnectError",
    "McpConnection",
    "McpError",
    "McpProtocolError",
    "McpTimeoutError",
    "RemoteTool",
    "ServerInfo",
    "ServerProcess",
    "StdioServerSpec",
    "connect_http",
    "connect_in_process",
    "connect_stdio",
    "sdk_version",
]

Era = Literal["auto", "legacy"]
"""How to negotiate: "auto" (SDK default: modern `server/discover`, falling back
to the legacy `initialize` handshake) or "legacy" (handshake only)."""

DEFAULT_CONNECT_TIMEOUT = 30.0
_MAX_TOOL_PAGES = 100
_CLIENT_NAME = "lucy"


# --- Lucy-owned types ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServerInfo:
    protocol_version: str
    era: Literal["modern", "legacy"]
    name: str | None
    version: str | None
    capabilities: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RemoteTool:
    name: str
    title: str | None
    description: str | None
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    annotations: dict[str, Any]
    """Wire-named hints (`readOnlyHint`, ...) as the server sent them: untrusted."""


@dataclass(frozen=True, slots=True)
class CallResult:
    texts: tuple[str, ...]
    structured: dict[str, Any] | None
    is_error: bool
    omitted: dict[str, int] = field(default_factory=dict)
    """Non-text content blocks by kind: image, audio, resource_link, resource, other."""


@dataclass(frozen=True, slots=True)
class StdioServerSpec:
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Merged by the SDK over its small default set (PATH, HOME, USER, ...)."""
    cwd: str | Path | None = None


@dataclass(frozen=True, slots=True)
class HttpServerSpec:
    url: str
    bearer_token: str | None = field(default=None, repr=False)


class McpError(Exception):
    """Base class for every error this module raises."""

    request_sent: bool = False
    """True when the request may have reached the server (so re-sending it could
    repeat its effect). Set by the lifecycle code (`server.McpServer`)."""


class McpConnectError(McpError):
    """The server could not be reached or started, or it went away (exited)."""


class McpProtocolError(McpError):
    """The server answered with a protocol error or a malformed message."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class McpTimeoutError(McpError):
    """A request (or connecting) took longer than allowed."""


def sdk_version() -> str:
    return _dist_version("mcp")


# --- Error translation ---------------------------------------------------------


_GONE = (OSError, httpx2.TransportError, anyio.BrokenResourceError, anyio.ClosedResourceError)


def _leaf(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


def _translate(exc: BaseException) -> McpError | None:
    exc = _leaf(exc)
    if isinstance(exc, McpError):
        return exc
    if isinstance(exc, MCPError):
        if exc.code == mcp_types.CONNECTION_CLOSED:
            return McpConnectError(f"MCP server connection closed: {exc.error.message}")
        if exc.code == mcp_types.REQUEST_TIMEOUT:
            return McpTimeoutError(exc.error.message)
        return McpProtocolError(exc.error.message, exc.code)
    if isinstance(exc, InputRequiredRoundsExceededError):
        return McpProtocolError(
            "the MCP server kept asking for more input; Lucy stopped after the allowed rounds"
        )
    if isinstance(exc, TimeoutError):
        return McpTimeoutError(str(exc) or "timed out")
    if isinstance(exc, ValidationError):
        return McpProtocolError(f"malformed message from MCP server: {exc.error_count()} errors")
    if isinstance(exc, _GONE):
        return McpConnectError(f"cannot reach MCP server: {exc}")
    if isinstance(exc, httpx2.HTTPError):
        return McpProtocolError(f"HTTP error from MCP server: {exc}")
    return None


class _translated:
    """Context manager re-raising SDK and transport errors as Lucy's."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, _t: object, exc: BaseException | None, _tb: object) -> bool:
        if exc is None or not isinstance(exc, Exception):
            return False
        mapped = _translate(exc)
        if mapped is None or mapped is exc:
            return False
        raise mapped from exc


# --- Conversion ----------------------------------------------------------------


def _remote_tool(tool: mcp_types.Tool) -> RemoteTool:
    annotations = (
        tool.annotations.model_dump(by_alias=True, exclude_none=True, mode="json")
        if tool.annotations is not None
        else {}
    )
    return RemoteTool(
        name=tool.name,
        title=tool.title or (tool.annotations.title if tool.annotations else None),
        description=tool.description,
        input_schema=dict(tool.input_schema),
        output_schema=dict(tool.output_schema) if tool.output_schema is not None else None,
        annotations=annotations,
    )


_KINDS: tuple[tuple[type, str], ...] = (
    (mcp_types.ImageContent, "image"),
    (mcp_types.AudioContent, "audio"),
    (mcp_types.ResourceLink, "resource_link"),
    (mcp_types.EmbeddedResource, "resource"),
)


def _call_result(result: mcp_types.CallToolResult) -> CallResult:
    texts: list[str] = []
    omitted: Counter[str] = Counter()
    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            texts.append(block.text)
            continue
        omitted[next((k for t, k in _KINDS if isinstance(block, t)), "other")] += 1
    structured = result.structured_content
    return CallResult(
        texts=tuple(texts),
        structured=dict(structured) if isinstance(structured, Mapping) else None,
        is_error=bool(result.is_error),
        omitted=dict(omitted),
    )


# --- Spawned process -----------------------------------------------------------


class ServerProcess:
    """The stdio server's process as the SDK spawned it (POSIX: in its own session,
    so its process group id is its pid). Read-only: the lifecycle code uses it to
    see whether the server exited and to clean up its process group."""

    def __init__(self, process: Any) -> None:
        self._process = process
        self.pid: int = process.pid

    def exited(self) -> bool:
        return self._process.returncode is not None


_on_spawn: contextvars.ContextVar[Callable[[ServerProcess], None] | None] = contextvars.ContextVar(
    "lucy_mcp_on_spawn", default=None
)


def _install_spawn_hook() -> None:
    # The SDK spawns the server inside `stdio_client` and exposes no process
    # handle. Wrap its (private) spawn function so the caller of `connect_stdio`
    # learns the process as soon as it exists, even if connecting then fails or is
    # cancelled. Checked by `tests/mcp/test_sdk_stdio.py`; if the SDK renames the
    # function, `McpConnection.process` stays None and that test fails.
    original = getattr(_sdk_stdio, "_create_platform_compatible_process", None)
    if original is None or getattr(original, "_lucy_hook", False):
        return

    async def spawn(*args: Any, **kwargs: Any) -> Any:
        process = await original(*args, **kwargs)
        callback = _on_spawn.get()
        if callback is not None:
            callback(ServerProcess(process))
        return process

    spawn._lucy_hook = True  # type: ignore[attr-defined]
    _sdk_stdio._create_platform_compatible_process = spawn  # type: ignore[assignment]


_install_spawn_hook()


# --- Connection ----------------------------------------------------------------


class McpConnection:
    """One live session with one MCP server. Use the `connect_*` functions."""

    def __init__(self, client: Client, stack: AsyncExitStack) -> None:
        self._client = client
        self._stack = stack
        self._closed = False
        self.process: ServerProcess | None = None
        """stdio only: the server process (None for HTTP and in-process)."""
        version = client.protocol_version
        server = client.server_info
        self.info = ServerInfo(
            protocol_version=version,
            era="modern" if version in MODERN_PROTOCOL_VERSIONS else "legacy",
            name=server.name if server else None,
            version=server.version if server else None,
            capabilities=client.server_capabilities.model_dump(
                by_alias=True, exclude_none=True, mode="json"
            ),
        )

    async def list_tools(self, timeout: float | None = None) -> list[RemoteTool]:
        """Every tool the server lists, following pagination (bounded)."""
        self._check_open()
        tools: list[RemoteTool] = []
        cursor: str | None = None
        with _translated(), anyio.fail_after(timeout):
            for _ in range(_MAX_TOOL_PAGES):
                page = await self._client.list_tools(cursor=cursor)
                tools.extend(_remote_tool(t) for t in page.tools)
                cursor = page.next_cursor
                if not cursor:
                    return tools
        raise McpProtocolError(f"tool listing did not end after {_MAX_TOOL_PAGES} pages")

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None, timeout: float
    ) -> CallResult:
        """Call a tool. A tool-level failure is `CallResult.is_error`, not an exception."""
        self._check_open()
        with _translated():
            result = await self._client.call_tool(
                name, dict(arguments or {}), read_timeout_seconds=timeout
            )
        return _call_result(result)

    def _check_open(self) -> None:
        if self._closed:
            raise McpConnectError("MCP connection is closed")

    async def aclose(self) -> None:
        """Close the session and (stdio) stop the server process. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._stack.aclose()
        except Exception as exc:  # closing a dead connection must not fail shutdown
            if _translate(exc) is None:
                raise


async def _open(
    server: Any,
    *,
    era: Era,
    connect_timeout: float | None,
    setup: Callable[[AsyncExitStack], Awaitable[Any]] | None = None,
) -> McpConnection:
    if era not in ("auto", "legacy"):
        raise ValueError(f"era must be 'auto' or 'legacy', got {era!r}")
    stack = AsyncExitStack()
    try:
        # Not wrapped in a cancel scope: the SDK opens task groups that outlive
        # this call. The handshake is bounded by the SDK's per-request timeout.
        with _translated():
            if setup is not None:
                server = await setup(stack)
            client = Client(
                server,
                mode=era,
                read_timeout_seconds=connect_timeout,  # default for requests without their own
                cache=None,  # Lucy decides when to re-list; no hidden response cache
                client_info=mcp_types.Implementation(name=_CLIENT_NAME, version=_lucy_version()),
            )
            await stack.enter_async_context(client)
        return McpConnection(client, stack)
    except BaseException:
        with anyio.CancelScope(shield=True):
            try:
                await stack.aclose()
            except Exception:
                pass
        raise


def _lucy_version() -> str:
    try:
        return _dist_version("lucy")
    except Exception:
        return "0"


async def connect_stdio(
    spec: StdioServerSpec,
    *,
    era: Era = "auto",
    connect_timeout: float | None = DEFAULT_CONNECT_TIMEOUT,
    errlog: Any = None,
    on_spawn: Callable[[ServerProcess], None] | None = None,
) -> McpConnection:
    """Start `spec.command` as a child process and talk MCP over its stdin/stdout.

    `errlog` is a text file with a real file descriptor for the server's stderr
    (default: /dev/null). `on_spawn` is called with the process as soon as it is
    started (before the handshake), so a caller can clean up after a failed or
    cancelled connect; the process is also `McpConnection.process`.
    """
    params = StdioServerParameters(
        command=spec.command,
        args=list(spec.args),
        env=dict(spec.env) if spec.env is not None else None,
        cwd=spec.cwd,
    )

    async def setup(stack: AsyncExitStack) -> Any:
        sink = errlog
        if sink is None:
            sink = stack.enter_context(open(os.devnull, "w"))  # noqa: SIM115 - closed by the stack
        return stdio_client(params, errlog=sink)

    spawned: list[ServerProcess] = []

    def record(process: ServerProcess) -> None:
        spawned.append(process)
        if on_spawn is not None:
            on_spawn(process)

    token = _on_spawn.set(record)
    try:
        conn = await _open(None, era=era, connect_timeout=connect_timeout, setup=setup)
    finally:
        _on_spawn.reset(token)
    conn.process = spawned[0] if spawned else None
    return conn


class _OriginBearerAuth(httpx2.Auth):
    """Adds the bearer token only to requests for the configured origin."""

    def __init__(self, url: str, token: str) -> None:
        self._origin = _origin(url)
        self._token = token

    def auth_flow(
        self, request: httpx2.Request
    ) -> Generator[httpx2.Request, httpx2.Response, None]:
        if _origin(str(request.url)) == self._origin:
            request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    port = parts.port or {"http": 80, "https": 443}.get(scheme)
    return scheme, (parts.hostname or "").lower(), port


async def connect_http(
    spec: HttpServerSpec,
    *,
    era: Era = "auto",
    connect_timeout: float | None = DEFAULT_CONNECT_TIMEOUT,
) -> McpConnection:
    """Connect over Streamable HTTP.

    The token is attached per request only when the request's origin equals the
    configured URL's origin. Redirects: the SDK follows a redirect only within
    the endpoint's origin, whatever the client's `follow_redirects` says; the
    client here also leaves httpx's own redirect following off.
    """
    scheme = urlsplit(spec.url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"MCP server URL must be http(s): {spec.url!r}")

    async def setup(stack: AsyncExitStack) -> Any:
        auth = _OriginBearerAuth(spec.url, spec.bearer_token) if spec.bearer_token else None
        http = httpx2.AsyncClient(
            auth=auth,
            follow_redirects=False,
            timeout=httpx2.Timeout(30.0, read=300.0),
        )
        await stack.enter_async_context(http)
        return streamable_http_client(spec.url, http_client=http)

    return await _open(None, era=era, connect_timeout=connect_timeout, setup=setup)


async def connect_in_process(
    server: object,
    *,
    era: Era = "auto",
    connect_timeout: float | None = DEFAULT_CONNECT_TIMEOUT,
) -> McpConnection:
    """Connect to an SDK server object in this process (tests and diagnostics).

    In "auto" the SDK dispatches modern requests directly (no JSON-RPC framing);
    "legacy" runs the handshake over in-memory streams.
    """
    return await _open(server, era=era, connect_timeout=connect_timeout)
