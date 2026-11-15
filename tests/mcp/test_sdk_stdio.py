"""Contract tests: the fixture as a stdio subprocess, era fallback, and process death."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from lucy.adapters.mcp import sdk

SCRIPT = Path(__file__).with_name("fixture_server.py")


def spec(*args: str) -> sdk.StdioServerSpec:
    return sdk.StdioServerSpec(command=sys.executable, args=(str(SCRIPT), *args))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def test_modern_server_negotiates_the_modern_era() -> None:
    conn = await sdk.connect_stdio(spec(), connect_timeout=10)
    try:
        assert conn.info.era == "modern"
        assert conn.info.name == "lucy-fixture"
        assert (await conn.call_tool("echo", {"text": "hi"}, timeout=5)).texts == ("echo: hi",)
        assert "echo" in {t.name for t in await conn.list_tools()}
    finally:
        await conn.aclose()


async def test_legacy_only_server_falls_back_to_the_handshake() -> None:
    # The server refuses modern requests, so "auto" must fall back to initialize.
    conn = await sdk.connect_stdio(spec("legacy"), era="auto", connect_timeout=10)
    try:
        assert conn.info.era == "legacy"
        assert conn.info.protocol_version < "2026"  # a handshake-era date version
        assert (await conn.call_tool("echo", {"text": "hi"}, timeout=5)).texts == ("echo: hi",)
    finally:
        await conn.aclose()


async def test_forced_legacy_era_against_a_dual_era_server() -> None:
    conn = await sdk.connect_stdio(spec(), era="legacy", connect_timeout=10)
    try:
        assert conn.info.era == "legacy"
    finally:
        await conn.aclose()


async def test_server_dying_mid_session_is_a_connect_error_not_a_hang() -> None:
    conn = await sdk.connect_stdio(spec(), connect_timeout=10)
    try:
        started = time.monotonic()
        with pytest.raises(sdk.McpConnectError):
            await conn.call_tool("die", {}, timeout=10)
        assert time.monotonic() - started < 5
        with pytest.raises(sdk.McpConnectError):
            await conn.call_tool("echo", {"text": "x"}, timeout=5)
    finally:
        await conn.aclose()


async def _spawn_and_close() -> tuple[int, int]:
    conn = await sdk.connect_stdio(spec(), connect_timeout=10)
    try:
        text = (await conn.call_tool("spawn_child", {}, timeout=5)).texts[0]
        server_pid, grandchild = (int(p) for p in text.split())
        assert alive(server_pid) and alive(grandchild)
    finally:
        await conn.aclose()
    return server_pid, grandchild


def wait_gone(pid: int, seconds: float = 3) -> bool:
    deadline = time.monotonic() + seconds
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    return not alive(pid)


async def test_aclose_stops_the_server_process() -> None:
    server_pid, grandchild = await _spawn_and_close()
    try:
        assert wait_gone(server_pid)
    finally:
        _kill(grandchild)


@pytest.mark.xfail(
    strict=True,
    reason="SDK 2.3 closes stdin and kills the process group only if the server has "
    "not exited within 2 s; a server that exits on EOF orphans its children. "
    "The lifecycle run must handle this (ADR 0022). Strict: an SDK fix shows up here.",
)
async def test_aclose_also_stops_the_servers_children() -> None:
    _, grandchild = await _spawn_and_close()
    try:
        assert wait_gone(grandchild, seconds=0.5)
    finally:
        _kill(grandchild)


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def test_missing_command_is_a_connect_error() -> None:
    with pytest.raises(sdk.McpConnectError):
        await sdk.connect_stdio(sdk.StdioServerSpec(command="/nonexistent/lucy-mcp-server"))


async def test_connection_exposes_the_spawned_process() -> None:
    seen: list[sdk.ServerProcess] = []
    conn = await sdk.connect_stdio(spec(), connect_timeout=10, on_spawn=seen.append)
    try:
        pid = int((await conn.call_tool("pid", {}, timeout=5)).texts[0])
        assert conn.process is not None and conn.process.pid == pid
        assert [p.pid for p in seen] == [pid] and not conn.process.exited()
        assert os.getpgid(pid) == pid  # its own process group
    finally:
        await conn.aclose()
    assert conn.process.exited()
