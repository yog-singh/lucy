"""Lock file, `McpToolProvider` and `lucy mcp` against the stdio fixture (ADR 0022, run 5)."""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from lucy.adapters.mcp.lock import (
    LOCK_FORMAT,
    LockError,
    LockFile,
    PinnedServer,
    review,
    tool_hash,
)
from lucy.adapters.mcp.provider import McpToolProvider, guidance_sentence
from lucy.adapters.mcp.sdk import RemoteTool, ServerInfo
from lucy.adapters.mcp.secrets import FakeSecrets
from lucy.clients import mcp_cli
from lucy.core.config import McpServerSettings, McpSettings, Settings
from lucy.core.types import Risk

SCRIPT = Path(__file__).with_name("fixture_server.py")
INFO = ServerInfo("2026-07-28", "modern", "lucy-fixture", "1.2.3", {})


def remote(name: str, description: str = "Echo the text back.", **kw: Any) -> RemoteTool:
    return RemoteTool(
        name=name,
        title=kw.get("title"),
        description=description,
        input_schema=kw.get(
            "schema", {"type": "object", "properties": {"text": {"type": "string"}}}
        ),
        output_schema=None,
        annotations=kw.get("annotations", {}),
    )


def server_conf(tools: dict | None = None, args: tuple[str, ...] = ()) -> McpServerSettings:
    return McpServerSettings(
        command=sys.executable,
        args=[str(SCRIPT), *args],
        tools=tools if tools is not None else {"echo": {"risk": "read"}, "delete_thing": {}},
    )


def mcp_settings(conf: McpServerSettings | None = None, enabled: bool = True) -> McpSettings:
    return McpSettings(enabled=["fx"] if enabled else [], servers={"fx": conf or server_conf()})


# --- Lock ------------------------------------------------------------------------


def test_lock_round_trip_atomic_owner_only(tmp_path: Path) -> None:
    root = tmp_path / "mcp"
    assert LockFile.load(root).servers == {}  # missing file: empty
    lock = LockFile.load(root)
    lock.servers["fx"] = PinnedServer.approve(server_conf(), INFO, "2.3.0", [remote("echo")])
    lock.save()
    path = root / "lock.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in root.iterdir()] == ["lock.json"]  # no temporary left
    data = json.loads(path.read_text())
    assert data["format"] == LOCK_FORMAT
    assert path.read_text() == lock.to_text()  # stable
    again = LockFile.load(root)
    assert again.servers == lock.servers
    assert again.servers["fx"].tools["echo"].hash == tool_hash(remote("echo"))


def test_corrupt_and_newer_lock_refused(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    path.write_text("{not json")
    with pytest.raises(LockError, match="damaged"):
        LockFile.load(tmp_path)
    path.write_text(json.dumps({"format": LOCK_FORMAT + 1, "servers": {}}))
    with pytest.raises(LockError, match="newer Lucy"):
        LockFile.load(tmp_path)


def test_hash_covers_description_schema_title_not_annotations() -> None:
    base = tool_hash(remote("echo"))
    assert tool_hash(remote("echo", annotations={"readOnlyHint": True})) == base
    assert tool_hash(remote("echo", "Other.")) != base
    assert tool_hash(remote("echo", schema={"type": "object"})) != base
    assert tool_hash(remote("echo", title="Echo")) != base


def test_review_classes_and_launch_change() -> None:
    conf = server_conf({"echo": {"risk": "read"}, "weather": {}, "gone": {}, "fresh": {}})
    pin = PinnedServer.approve(
        conf, INFO, "2.3.0", [remote("echo"), remote("weather", "Old."), remote("gone")]
    )
    live = [remote("echo"), remote("weather", "New."), remote("fresh"), remote("extra")]
    result = review(pin, conf, live)
    states = {t.name: t.state for t in result.tools}
    assert states == {"echo": "approved", "weather": "changed", "gone": "gone", "fresh": "new"}
    assert result.by_state("changed")[0].changes == ("description",)
    assert result.unconfigured == ("extra",) and not result.clean
    moved = server_conf(conf.model_dump()["tools"], args=("legacy",))
    relaunched = review(pin, moved, live)
    assert relaunched.launch_changed
    assert {t.state for t in relaunched.tools} == {"new"}


def test_guidance_sentence_names_its_tool() -> None:
    assert guidance_sentence("fx_echo", "Use it  for echoes") == (
        "About the fx_echo tool: Use it for echoes."
    )
    assert guidance_sentence("fx_echo", "Use the fx_echo tool to echo.") == (
        "Use the fx_echo tool to echo."
    )


# --- Provider --------------------------------------------------------------------


def approve_fixture(root: Path, conf: McpServerSettings, tools: list[RemoteTool]) -> None:
    lock = LockFile.load(root)
    lock.servers["fx"] = PinnedServer.approve(conf, INFO, "2.3.0", tools)
    lock.save()


async def live_tools(tmp_path: Path, conf: McpServerSettings) -> list[RemoteTool]:
    from lucy.adapters.mcp.server import McpServer

    server = McpServer("fx", conf, secrets=FakeSecrets({}), root=tmp_path / "scratch")
    try:
        return await server.list_tools()
    finally:
        await server.aclose()


def test_nothing_exposed_without_approval(tmp_path: Path, caplog) -> None:
    provider = McpToolProvider(mcp_settings(), secrets=FakeSecrets({}), root=tmp_path)
    assert provider.tools() == []
    assert "2 tools waiting for approval" in caplog.text and "lucy mcp approve fx" in caplog.text
    assert provider.servers["fx"].state == "stopped"


async def test_approved_tools_work_and_changed_tool_refuses(tmp_path: Path) -> None:
    conf = server_conf({"echo": {"risk": "read", "guidance": "Echo things"}, "delete_thing": {}})
    live = {t.name: t for t in await live_tools(tmp_path, conf)}
    tampered = dataclasses.replace(live["delete_thing"], description="Delete a thing, gently.")
    approve_fixture(tmp_path, conf, [live["echo"], tampered])
    provider = McpToolProvider(mcp_settings(conf), secrets=FakeSecrets({}), root=tmp_path)
    tools = {t.name: t for t in provider.tools()}
    assert list(tools) == ["fx_echo", "fx_delete_thing"]
    assert tools["fx_echo"].risk == Risk.READ and tools["fx_delete_thing"].risk == Risk.DANGEROUS
    assert provider.guidance() == ["About the fx_echo tool: Echo things."]
    await provider.start()
    server = provider.servers["fx"]
    assert server.state == "stopped" and server.pid is None
    try:
        out = await tools["fx_echo"].run({"text": "hi"})
        assert out.content.splitlines()[1] == "echo: hi" and not out.is_error
        pid = server.pid
        assert pid is not None
        refused = await tools["fx_delete_thing"].run({"thing_id": 3})
        assert refused.is_error and "changed this tool (description)" in refused.content
        assert "lucy mcp approve fx" in refused.content
        assert provider.diagnostics()["fx"]["refused"] == ["delete_thing"]
    finally:
        await provider.aclose()
    assert server.state == "stopped" and server.pid is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_launch_change_exposes_nothing(tmp_path: Path) -> None:
    conf = server_conf({"echo": {"risk": "read"}})
    approve_fixture(tmp_path, conf, [remote("echo")])
    moved = server_conf({"echo": {"risk": "read"}}, args=("legacy",))
    provider = McpToolProvider(mcp_settings(moved), secrets=FakeSecrets({}), root=tmp_path)
    assert provider.tools() == []


# --- CLI -------------------------------------------------------------------------


def cli_settings(conf: McpServerSettings | None = None) -> Settings:
    from tests.test_app import make_settings

    base: Settings = make_settings(Path("unused.db"))
    return base.model_copy(update={"mcp": mcp_settings(conf)})


def run_cli(
    tmp_path: Path, settings: Settings, *argv: str, answer: bool = True
) -> tuple[int, str, str]:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    mcp_cli.add_parser(commands, argparse.ArgumentParser(add_help=False))
    args = parser.parse_args(["mcp", *argv])
    out, err = io.StringIO(), io.StringIO()
    asked: list[str] = []

    def confirm(question: str) -> bool:
        asked.append(question)
        return answer

    code = mcp_cli.run(
        settings, args, out=out, err=err, confirm=confirm, root=tmp_path, secrets=FakeSecrets({})
    )
    return code, out.getvalue(), err.getvalue()


def test_cli_flow(tmp_path: Path) -> None:
    settings = cli_settings()
    code, out, _ = run_cli(tmp_path, settings, "list")
    assert code == 0 and "fx (enabled)" in out and "none yet; 2 tool(s) waiting" in out

    code, out, _ = run_cli(tmp_path, settings, "tools", "fx")
    assert code == 0
    assert "echo: Echo the text back." in out
    assert "the server's own claim" in out
    assert "risk read, offered as fx_echo; state: new" in out
    assert "pid: The process id of the server." in out and "state: not configured" in out

    code, out, _ = run_cli(tmp_path, settings, "check", "fx")
    assert code == 1 and "protocol:" in out and "era)" in out and "echo: new" in out

    code, out, _ = run_cli(tmp_path, settings, "approve", "fx", answer=False)
    assert code == 1 and "Not approved" in out and not (tmp_path / "lock.json").exists()
    assert f"Arguments: {json.dumps([str(SCRIPT)])}" in out
    assert "confirmation question: a generic one" in out
    assert "Read tools: echo" in out and "untrusted" in out

    code, out, _ = run_cli(tmp_path, settings, "approve", "fx")
    assert code == 0 and (tmp_path / "lock.json").exists()

    code, out, _ = run_cli(
        tmp_path, settings, "check", "fx", "--call", "echo", "--args", '{"text": "x"}'
    )
    assert code == 0, out
    assert "echo: approved" in out and "echo: x" in out

    code, out, err = run_cli(tmp_path, settings, "check", "fx", "--call", "delete_thing")
    assert code == 2 and 'risk = "read"' in err

    code, out, _ = run_cli(tmp_path, settings, "list")
    assert "2 approved, 0 waiting" in out and "server lucy-fixture 1.2.3" in out


def test_cli_approve_yes_flag_and_unknown_server(tmp_path: Path) -> None:
    settings = cli_settings(server_conf({"echo": {"risk": "read"}, "missing": {}}))
    code, out, _ = run_cli(tmp_path, settings, "approve", "fx", "--yes", answer=False)
    assert code == 0 and "missing: not offered by the server; not pinned" in out
    assert list(LockFile.load(tmp_path).servers["fx"].tools) == ["echo"]
    code, _, err = run_cli(tmp_path, settings, "tools", "nope")
    assert code == 2 and "no MCP server called 'nope'" in err


def test_config_before_the_action_is_never_silently_dropped() -> None:
    # Found live: `lucy mcp --config X list` parsed but used ./lucy.toml.
    from lucy.clients.cli import _parser

    args = _parser().parse_args(["mcp", "list", "--config", "x.toml"])
    assert args.config == Path("x.toml")
    try:
        args = _parser().parse_args(["mcp", "--config", "x.toml", "list"])
    except SystemExit:
        return  # refused loudly: fine
    assert args.config == Path("x.toml")


def test_configured_path_is_used_by_cli_and_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # anything relative would land here, not in the repo
    data = tmp_path / "elsewhere"
    base = cli_settings(server_conf({"echo": {"risk": "read"}}))
    settings = base.model_copy(update={"mcp": base.mcp.model_copy(update={"path": str(data)})})
    parser = argparse.ArgumentParser()
    mcp_cli.add_parser(
        parser.add_subparsers(dest="command"), argparse.ArgumentParser(add_help=False)
    )
    args = parser.parse_args(["mcp", "approve", "fx", "--yes"])
    out, err = io.StringIO(), io.StringIO()
    code = mcp_cli.run(
        settings, args, out=out, err=err, confirm=lambda _: False, secrets=FakeSecrets({})
    )
    assert code == 0, err.getvalue()
    assert (data / "lock.json").exists() and (data / "fx").is_dir()  # lock and workdir
    assert sorted(p.name for p in tmp_path.iterdir()) == ["elsewhere"]

    from lucy.app import build_tool_providers

    monkeypatch.setattr("lucy.adapters.mcp.secrets.KeychainSecrets", lambda: FakeSecrets({}))
    (provider,) = build_tool_providers(settings)
    assert [t.name for t in provider.tools()] == ["fx_echo"]  # read from the configured lock
