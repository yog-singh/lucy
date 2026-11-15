"""`McpServer`: one server's lifecycle against the stdio fixture (ADR 0022, run 4)."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import stat
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from mcp import InputRequiredRoundsExceededError

from lucy.adapters.mcp import sdk
from lucy.adapters.mcp.sdk import (
    CallResult,
    McpConnectError,
    McpProtocolError,
    McpTimeoutError,
    RemoteTool,
    ServerInfo,
)
from lucy.adapters.mcp.secrets import FakeSecrets
from lucy.adapters.mcp.server import STDERR_LINE_CHARS, McpServer
from lucy.adapters.mcp.tool import McpTool
from lucy.core.config import McpServerSettings, McpToolSettings
from lucy.core.types import Risk

SCRIPT = Path(__file__).with_name("fixture_server.py")
BASE = {"HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"}
# Python adds LC_CTYPE itself (PEP 538 locale coercion); macOS may add this one.
PLATFORM = {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}
SECRET = "s3cr3t-value-never-logged"


def settings(*args: str, **kw: Any) -> McpServerSettings:
    return McpServerSettings(command=sys.executable, args=[str(SCRIPT), *args], **kw)


def make(tmp_path: Path, *args: str, secrets: dict | None = None, **kw: Any) -> McpServer:
    conf = {k: kw.pop(k) for k in ("env", "env_from", "env_keychain") if k in kw}
    return McpServer(
        "fx",
        settings(*args, **conf),
        secrets=FakeSecrets(secrets or {}),
        root=tmp_path / "mcp",
        connect_timeout=10,
        **kw,
    )


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def echo(server: McpServer, text: str = "hi") -> str:
    return (await server.call("echo", {"text": text}, 5)).texts[0]


async def test_lazy_start_and_concurrent_first_calls_start_one_process(tmp_path: Path) -> None:
    server = make(tmp_path)
    try:
        assert server.state == "stopped" and server.pid is None and server.info is None
        results = await asyncio.gather(*(echo(server, str(i)) for i in range(5)))
        assert results == [f"echo: {i}" for i in range(5)]
        assert server.start_count == 1 and server.state == "up"
        assert server.info is not None and server.info.name == "lucy-fixture"
        assert server.last_error is None
    finally:
        await server.aclose()
    assert server.state == "stopped"


async def test_timeout_leaves_the_server_usable(tmp_path: Path) -> None:
    server = make(tmp_path)
    try:
        with pytest.raises(McpTimeoutError) as caught:
            await server.call("slow", {"seconds": 5}, 0.3)
        assert caught.value.request_sent
        assert await echo(server) == "echo: hi"
        assert server.start_count == 1
    finally:
        await server.aclose()


async def test_crash_in_flight_is_returned_not_resent_then_next_call_restarts(
    tmp_path: Path,
) -> None:
    server = make(tmp_path, restart_backoff=0)
    try:
        await echo(server)
        first = server.pid
        with pytest.raises(McpConnectError) as caught:
            await server.call("die", {}, 5)
        assert caught.value.request_sent
        assert server.state == "down" and server.last_error
        assert server.start_count == 1  # not re-sent, not restarted by the failing call
        assert await echo(server) == "echo: hi"
        assert server.start_count == 2 and server.pid != first and server.state == "up"
    finally:
        await server.aclose()


async def test_repeated_crashes_respect_the_backoff(tmp_path: Path) -> None:
    now = [100.0]
    server = make(tmp_path, restart_backoff=30, clock=lambda: now[0])
    try:
        with pytest.raises(McpConnectError):
            await server.call("die", {}, 5)
        now[0] = 110.0
        started = time.monotonic()
        with pytest.raises(McpConnectError, match="down") as caught:
            await echo(server)
        assert not caught.value.request_sent
        assert time.monotonic() - started < 0.1  # fails fast, no spawn
        assert server.start_count == 1 and server.state == "down"
        now[0] = 131.0
        assert await echo(server) == "echo: hi"
        assert server.start_count == 2
    finally:
        await server.aclose()


async def test_idle_exit_is_noticed_and_restarted_before_sending(tmp_path: Path) -> None:
    now = [100.0]
    server = make(tmp_path, clock=lambda: now[0])
    try:
        await echo(server)
        pid = server.pid
        assert pid is not None
        kill(pid)
        deadline = time.monotonic() + 3
        while server._process is not None and not server._process.exited():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        now[0] = 200.0
        assert await echo(server) == "echo: hi"  # not an error: nothing had been sent
        assert server.start_count == 2
    finally:
        await server.aclose()


async def test_aclose_is_safe_when_never_started_and_twice(tmp_path: Path) -> None:
    server = make(tmp_path)
    await server.aclose()
    await server.aclose()
    assert server.state == "stopped" and server.start_count == 0
    with pytest.raises(McpConnectError, match="shut down"):
        await echo(server)


async def test_aclose_during_start(tmp_path: Path) -> None:
    server = make(tmp_path)
    starting = asyncio.create_task(server.ensure_started())
    await asyncio.sleep(0)  # the owner task is created and spawning
    assert server.state == "starting"
    await server.aclose()
    with pytest.raises(McpConnectError):
        await starting
    assert server.state == "stopped" and server.pid is None


async def test_aclose_with_calls_in_flight_and_waiting(tmp_path: Path) -> None:
    server = make(tmp_path)
    await server.ensure_started()
    pid = server.pid
    assert pid is not None
    calls = [asyncio.create_task(server.call("slow", {"seconds": 10}, 20)) for _ in range(2)]
    await asyncio.sleep(0.2)  # the first is in flight, the second is queued
    started = time.monotonic()
    await server.aclose()
    assert time.monotonic() - started < 3
    results = await asyncio.gather(*calls, return_exceptions=True)
    assert all(isinstance(r, McpConnectError) for r in results)
    assert [r.request_sent for r in results] == [True, False]
    assert not alive(pid)


async def test_no_grandchild_survives_aclose(tmp_path: Path) -> None:
    server = make(tmp_path)
    grandchild = None
    try:
        text = (await server.call("spawn_child", {}, 5)).texts[0]
        _, grandchild = (int(p) for p in text.split())
        assert alive(grandchild)
        await server.aclose()
        assert not alive(grandchild)  # the group is gone when aclose returns
    finally:
        if grandchild:
            kill(grandchild)


async def test_close_is_bounded_for_a_server_ignoring_eof(tmp_path: Path) -> None:
    # Bound: close_grace (0.3 s) + SIGTERM wait (<= KILL_GRACE 1 s) + overhead.
    server = make(tmp_path, "ignore-eof", close_grace=0.3)
    await server.ensure_started()
    pid = server.pid
    assert pid is not None
    started = time.monotonic()
    await server.aclose()
    assert time.monotonic() - started < 1.5
    assert not alive(pid)


async def test_child_environment_is_minimal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUCY_TEST_UNRELATED", "leak")
    monkeypatch.setenv("LUCY_TEST_PASS", "passed")
    server = make(
        tmp_path,
        env={"PLAIN": "1"},
        env_from=["LUCY_TEST_PASS"],
        env_keychain={"API_TOKEN": "svc-token"},
        secrets={"svc-token": SECRET},
    )
    try:
        names = set((await server.call("env_names", {}, 5)).texts[0].split())
        configured = {"PLAIN", "LUCY_TEST_PASS", "API_TOKEN"}
        assert configured <= names
        assert names - PLATFORM <= BASE | configured
        assert "LUCY_TEST_UNRELATED" not in names
        get = server.call
        assert (await get("getenv", {"name": "API_TOKEN"}, 5)).texts == (SECRET,)
        assert (await get("getenv", {"name": "LUCY_TEST_PASS"}, 5)).texts == ("passed",)
        cwd = Path((await get("cwd", {}, 5)).texts[0])
        assert cwd == (tmp_path / "mcp" / "fx").resolve()
        assert stat.S_IMODE(cwd.stat().st_mode) == 0o700
    finally:
        await server.aclose()
    assert SECRET not in repr(server)


async def test_missing_env_from_is_a_clear_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LUCY_TEST_MISSING", raising=False)
    server = make(tmp_path, env_from=["LUCY_TEST_MISSING"])
    with pytest.raises(McpConnectError, match="LUCY_TEST_MISSING"):
        await server.ensure_started()
    assert server.state == "down" and server.pid is None
    assert "LUCY_TEST_MISSING" in (server.last_error or "")
    await server.aclose()


async def test_missing_secret_is_a_clear_error_without_values(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    server = make(
        tmp_path, env_keychain={"A": "present", "B": "absent"}, secrets={"present": SECRET}
    )
    with pytest.raises(McpConnectError, match="'absent' not found") as caught:
        await server.ensure_started()
    assert SECRET not in str(caught.value) and SECRET not in caplog.text
    await server.aclose()


async def test_secrets_never_logged_and_reread_on_restart(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    secrets = FakeSecrets({"svc": SECRET})
    server = McpServer(
        "fx",
        settings(env_keychain={"API_TOKEN": "svc"}),
        secrets=secrets,
        root=tmp_path / "mcp",
        restart_backoff=0,
    )
    try:
        await echo(server)
        with pytest.raises(McpConnectError) as caught:
            await server.call("die", {}, 5)
        await echo(server)
        assert secrets.reads == ["svc", "svc"]
        assert SECRET not in str(caught.value)
    finally:
        await server.aclose()
    assert SECRET not in caplog.text and SECRET not in repr(server)


async def test_stderr_is_logged_cleaned_and_capped(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="lucy.adapters.mcp.server")
    server = make(tmp_path)
    try:
        await server.call("shout", {"text": "\x1b[31mred\x07 alert"}, 5)
        await server.call("shout", {"text": "x" * 1000}, 5)
        await server.call("shout", {"text": "flood", "times": 80}, 5)
    finally:
        await server.aclose()  # joins the stderr reader
    lines = [r.getMessage() for r in caplog.records if "[mcp fx stderr]" in r.getMessage()]
    assert "[mcp fx stderr] [31mred alert" in lines
    assert all("\x1b" not in line and "\x07" not in line for line in lines)
    long = next(line for line in lines if "xxx" in line)
    assert len(long.removeprefix("[mcp fx stderr] ")) == STDERR_LINE_CHARS
    assert sum("flood" in line for line in lines) < 50
    assert any("lines dropped" in line for line in lines)


# --- HTTP (fake at the connect_http seam) ----------------------------------------


class FakeConnection:
    def __init__(self) -> None:
        self.info = ServerInfo("2026-01-01", "modern", "remote", "1", {})
        self.process = None
        self.fail_next = False
        self.closed = False

    async def list_tools(self, timeout: float | None = None) -> list[RemoteTool]:
        return [RemoteTool("echo", None, "Echo.", {"type": "object"}, None, {})]

    async def call_tool(self, name: str, arguments: Any, timeout: float) -> CallResult:
        if self.fail_next:
            raise McpConnectError("cannot reach MCP server")
        return CallResult(texts=(f"{name}:{arguments}",), structured=None, is_error=False)

    async def aclose(self) -> None:
        self.closed = True


async def test_http_server_lazy_connect_token_and_reconnect(tmp_path: Path) -> None:
    specs: list[sdk.HttpServerSpec] = []
    conns: list[FakeConnection] = []

    async def connect_http(spec: sdk.HttpServerSpec, **_: Any) -> FakeConnection:
        specs.append(spec)
        conns.append(FakeConnection())
        return conns[-1]

    server = McpServer(
        "remote",
        McpServerSettings(url="https://mcp.example.test/mcp", token_keychain="tok"),
        secrets=FakeSecrets({"tok": SECRET}),
        root=tmp_path / "mcp",
        restart_backoff=0,
        connect_http=connect_http,
    )
    assert not specs
    assert [t.name for t in await server.list_tools()] == ["echo"]
    assert specs[0].bearer_token == SECRET and SECRET not in repr(specs[0])
    conns[0].fail_next = True
    with pytest.raises(McpConnectError):
        await server.call("echo", {}, 5)
    assert server.state == "down"
    assert (await server.call("echo", {"a": 1}, 5)).texts == ("echo:{'a': 1}",)
    assert conns[0].closed
    assert len(specs) == 2 and server.start_count == 2
    await server.aclose()
    assert conns[1].closed and server.state == "stopped"


# --- READ-only retry in McpTool and the SDK error translation --------------------


class FlakyCaller:
    def __init__(self) -> None:
        self.calls = 0

    async def call(self, tool: str, arguments: Any, timeout: float) -> CallResult:
        self.calls += 1
        if self.calls == 1:
            exc = McpConnectError("gone")
            exc.request_sent = True
            raise exc
        return CallResult(texts=("ok",), structured=None, is_error=False)


@pytest.mark.parametrize(("risk", "calls", "error"), [(Risk.READ, 2, False), (Risk.ACT, 1, True)])
async def test_only_read_tools_are_retried_once(risk: Risk, calls: int, error: bool) -> None:
    caller = FlakyCaller()
    remote = RemoteTool("echo", None, "Echo.", {"type": "object"}, None, {})
    srv = McpServerSettings(command="/bin/true", tools={"echo": McpToolSettings(risk=risk)})
    tool = McpTool("fx", srv, srv.tools["echo"], remote, caller)
    out = await tool.run({})
    assert caller.calls == calls and out.is_error is error


def test_input_rounds_exceeded_is_a_protocol_error() -> None:
    mapped = sdk._translate(InputRequiredRoundsExceededError(3))
    assert isinstance(mapped, McpProtocolError)
    assert "more input" in str(mapped)
