"""`lucy mcp`: list servers, inspect and check one, approve its tools (ADR 0022).

No chat model is needed. `list` reads the settings and the lock file only; `tools`,
`check` and `approve` start the server, do their work and stop it. Output goes to
an injectable stream; confirmation goes through a `Confirm` callable (as in
`memory_cli`). Secrets are read only to start a server and never printed.

The MCP adapter (and so the SDK) is imported inside the commands, so registering
the subcommand costs nothing for the other `lucy` commands.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

from lucy.core.config import McpServerSettings, Settings, mcp_tool_name
from lucy.core.types import Risk

if TYPE_CHECKING:
    from lucy.adapters.mcp.secrets import Secrets
    from lucy.adapters.mcp.server import McpServer

Confirm = Callable[[str], bool]

UNTRUSTED_CLAIM: Final = "the server's own claim"


def add_parser(commands: Any, common: argparse.ArgumentParser) -> None:
    # No `parents=[common]` here: the action parsers' defaults would silently
    # overwrite a `--config` given before the action (`lucy mcp --config X list`).
    mcp = commands.add_parser("mcp", help="MCP servers: list, inspect, check and approve tools")
    actions = mcp.add_subparsers(dest="mcp_action", required=True, metavar="ACTION")
    actions.add_parser("list", parents=[common], help="configured servers and their approvals")
    tools = actions.add_parser("tools", parents=[common], help="start NAME and list its tools")
    tools.add_argument("name")
    check = actions.add_parser(
        "check", parents=[common], help="start NAME, show its protocol, compare with approvals"
    )
    check.add_argument("name")
    check.add_argument("--call", metavar="TOOL", help="also call one READ tool")
    check.add_argument("--args", default="{}", metavar="JSON", help="arguments for --call")
    approve = actions.add_parser(
        "approve", parents=[common], help="review NAME's configured tools and pin them"
    )
    approve.add_argument("name")
    approve.add_argument("--yes", action="store_true", help="approve without asking")


def _write(out: TextIO, line: str = "") -> None:
    out.write(line + "\n")


def _first_sentence(text: str | None) -> str:
    from lucy.adapters.mcp.tool import clean_description

    return clean_description(text) or "(no description)"


def _launch_text(launch: dict[str, Any]) -> str:
    if "url" in launch:
        return str(launch["url"])
    return shlex.join([str(launch.get("command") or ""), *map(str, launch.get("args") or [])])


def _settings_launch(settings: McpServerSettings) -> str:
    from lucy.adapters.mcp.lock import launch_of

    return _launch_text(launch_of(settings))


def _exposed_name(server: str, tool: str, settings: McpServerSettings) -> str:
    conf = settings.tools[tool]
    return conf.alias or mcp_tool_name(server, tool)


def _annotations(annotations: dict[str, Any]) -> str:
    if not annotations:
        return "none"
    shown = ", ".join(f"{k}={json.dumps(v)}" for k, v in sorted(annotations.items()))
    return f"{shown} ({UNTRUSTED_CLAIM})"


class _Usage(Exception):
    """A plain-sentence error for the user (exit code 2)."""


def _server_settings(settings: Settings, name: str) -> McpServerSettings:
    server = settings.mcp.servers.get(name)
    if server is None:
        known = ", ".join(settings.mcp.servers) or "none"
        raise _Usage(
            f"There is no MCP server called {name!r} in the settings (configured: {known})."
        )
    return server


def _load_lock(root: str | Path):
    from lucy.adapters.mcp.lock import LockFile

    return LockFile.load(root)


def _make_server(
    name: str, settings: McpServerSettings, secrets: Secrets | None, root
) -> McpServer:
    from lucy.adapters.mcp.secrets import KeychainSecrets
    from lucy.adapters.mcp.server import McpServer

    return McpServer(name, settings, secrets=secrets or KeychainSecrets(), root=root)


# --- list ------------------------------------------------------------------------


def list_servers(settings: Settings, out: TextIO, *, root: str | Path) -> int:
    from lucy.adapters.mcp.lock import review

    lock = _load_lock(root)
    if not settings.mcp.servers:
        _write(out, "No MCP servers are configured (add [mcp.servers.NAME] to the settings).")
        return 0
    for name, server in settings.mcp.servers.items():
        enabled = "enabled" if name in settings.mcp.enabled else "not enabled"
        pinned = lock.servers.get(name)
        _write(out, f"{name} ({enabled})")
        _write(out, f"  configured: {_settings_launch(server)}")
        if pinned is None:
            _write(out, f"  approval: none yet; {len(server.tools)} tool(s) waiting")
            continue
        result = review(pinned, server, None)
        if result.launch_changed:
            _write(out, f"  approved:   {_launch_text(pinned.launch)}")
            _write(out, "  approval: the launch command changed since approval; approve again")
            continue
        approved = len(result.by_state("approved"))
        waiting = len(result.by_state("new"))
        _write(out, f"  approval: {approved} approved, {waiting} waiting (on {pinned.approved_at})")
        _write(
            out,
            f"  at approval: server {pinned.server_name or '?'} {pinned.server_version or '?'}, "
            f"protocol {pinned.protocol_version or '?'} ({pinned.era or '?'}), "
            f"SDK {pinned.sdk_version or '?'}",
        )
    return 0


# --- tools -----------------------------------------------------------------------


async def _live(name: str, conf: McpServerSettings, secrets: Secrets | None, root):
    from lucy.adapters.mcp.sdk import McpError

    server = _make_server(name, conf, secrets, root)
    try:
        info = await server.ensure_started()
        live = await server.list_tools()
    except McpError as exc:
        await server.aclose()
        raise _Usage(f"Lucy couldn't start or list the {name} server: {exc}") from None
    return server, info, live


async def tools(
    settings: Settings,
    name: str,
    out: TextIO,
    *,
    root: str | Path,
    secrets: Secrets | None = None,
) -> int:
    from lucy.adapters.mcp.lock import review

    conf = _server_settings(settings, name)
    pinned = _load_lock(root).servers.get(name)
    server, _info, live = await _live(name, conf, secrets, root)
    try:
        result = review(pinned, conf, live)
        states = {t.name: t for t in result.tools}
        for tool in live:
            item = states.get(tool.name)
            _write(out, f"{tool.name}: {_first_sentence(tool.description)}")
            _write(out, f"  annotations: {_annotations(tool.annotations)}")
            if item is None:
                _write(out, "  configured: no; state: not configured")
                continue
            risk = conf.tools[tool.name].risk
            _write(
                out,
                f"  configured: yes, risk {risk.value}, offered as "
                f"{_exposed_name(name, tool.name, conf)}; state: {item.state}",
            )
            if fixed := conf.tools[tool.name].fixed:
                _write(out, f"  fixed (hidden from the model): {', '.join(sorted(fixed))}")
            if item.state == "changed":
                _changed(out, item)
        for item in result.tools:
            if item.live is None and item.state in ("gone", "new"):
                _write(out, f"{item.name}: configured but not offered by the server ({item.state})")
        if result.launch_changed:
            _write(out, "The launch command changed since approval: every tool needs approval.")
    finally:
        await server.aclose()
    return 0


def _changed(out: TextIO, item: Any) -> None:
    _write(out, f"  changed: {', '.join(item.changes) or 'definition'}")
    if "description" in item.changes:
        _write(out, f"    approved description: {item.pinned.description or ''}")
        _write(out, f"    current description:  {item.live.description or ''}")
    _write(out, f"    schema changed: {'yes' if 'input schema' in item.changes else 'no'}")


# --- check -----------------------------------------------------------------------


async def check(
    settings: Settings,
    name: str,
    out: TextIO,
    *,
    call: str | None = None,
    arguments: str = "{}",
    root: str | Path,
    secrets: Secrets | None = None,
) -> int:
    from lucy.adapters.mcp.lock import review
    from lucy.adapters.mcp.sdk import sdk_version
    from lucy.adapters.mcp.tool import McpTool

    conf = _server_settings(settings, name)
    if call is not None:
        tool_conf = conf.tools.get(call)
        if tool_conf is None or tool_conf.risk != Risk.READ:
            raise _Usage(
                f'`--call` only runs a tool configured with risk = "read"; {call!r} is '
                + ("not configured." if tool_conf is None else f"{tool_conf.risk.value}.")
            )
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise _Usage(f"--args is not valid JSON: {exc.msg}.") from None
        if not isinstance(parsed, dict):
            raise _Usage('--args must be a JSON object, such as \'{"city": "Paris"}\'.')
    pinned = _load_lock(root).servers.get(name)
    server, info, live = await _live(name, conf, secrets, root)
    try:
        _write(out, f"server: {info.name or '?'} {info.version or '?'}")
        _write(out, f"protocol: {info.protocol_version} ({info.era} era)")
        _write(out, f"SDK: mcp {sdk_version()}")
        _write(out, f"capabilities: {', '.join(sorted(info.capabilities)) or 'none'}")
        result = review(pinned, conf, live)
        if pinned is None:
            _write(out, "approval: none yet")
        elif result.launch_changed:
            _write(out, "approval: the launch command changed since approval")
        for item in result.tools:
            extra = f" ({', '.join(item.changes)})" if item.changes else ""
            _write(out, f"  {item.name}: {item.state}{extra}")
        if result.unconfigured:
            _write(out, f"not configured: {', '.join(result.unconfigured)}")
        if call is not None:
            remote = next((t for t in live if t.name == call), None)
            if remote is None:
                _write(out, f"call: the server does not offer {call!r}")
            else:
                tool = McpTool(name, conf, conf.tools[call], remote, server)
                output = await tool.run(parsed)
                _write(out, f"call {call}{' (error)' if output.is_error else ''}:")
                _write(out, output.content)
    finally:
        await server.aclose()
    return 0 if result.clean else 1


# --- approve ---------------------------------------------------------------------


async def approve(
    settings: Settings,
    name: str,
    out: TextIO,
    *,
    confirm: Confirm,
    root: str | Path,
    secrets: Secrets | None = None,
) -> int:
    from lucy.adapters.mcp.lock import PinnedServer
    from lucy.adapters.mcp.sdk import sdk_version

    conf = _server_settings(settings, name)
    lock = _load_lock(root)  # refuse a damaged lock before starting anything
    server, info, live = await _live(name, conf, secrets, root)
    await server.aclose()
    offered = {t.name: t for t in live}
    _write(out, f"Server {name}: {info.name or '?'} {info.version or '?'}")
    if conf.url is not None:
        _write(out, f"URL: {conf.url}")
    else:
        _write(out, f"Command: {conf.command}")
        _write(out, f"Arguments: {json.dumps(list(conf.args), ensure_ascii=False)}")
    _write(out, f"Protocol {info.protocol_version} ({info.era} era), SDK mcp {sdk_version()}")
    _write(out)
    pinned_tools = []
    reads: list[str] = []
    others: list[str] = []
    for tool_name, tool_conf in conf.tools.items():
        remote = offered.get(tool_name)
        if remote is None:
            _write(out, f"{tool_name}: not offered by the server; not pinned")
            continue
        pinned_tools.append(remote)
        (reads if tool_conf.risk == Risk.READ else others).append(tool_name)
        _write(
            out,
            f"{tool_name} (risk {tool_conf.risk.value}, offered as "
            f"{_exposed_name(name, tool_name, conf)})",
        )
        _write(out, f"  description: {remote.description or '(none)'}")
        _write(out, f"  annotations: {_annotations(remote.annotations)}")
        if tool_conf.risk == Risk.DANGEROUS:
            if tool_conf.preview:
                _write(out, f"  confirmation question: {tool_conf.preview}")
            else:
                _write(out, "  confirmation question: a generic one (no preview configured)")
    if reads and others and not conf.private:
        _write(out)
        _write(out, f"Read tools: {', '.join(reads)}")
        _write(out, f"Tools that send or change things: {', '.join(others)}")
        _write(out, "Results from this server are untrusted; it is not marked private.")
    if not pinned_tools:
        _write(out, "Nothing to approve: the server offers none of the configured tools.")
        return 1
    _write(out)
    if not confirm(f"Approve {len(pinned_tools)} tool(s) of {name} as shown? [y/N] "):
        _write(out, "Not approved; nothing changed.")
        return 1
    lock.servers[name] = PinnedServer.approve(conf, info, sdk_version(), pinned_tools)
    lock.save()
    _write(out, f"Approved {len(pinned_tools)} tool(s) of {name}; written to {lock.path}.")
    return 0


# --- entry -----------------------------------------------------------------------


def run(
    settings: Settings,
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    confirm: Confirm,
    root: str | Path | None = None,
    secrets: Secrets | None = None,
) -> int:
    """Run one `lucy mcp` action; returns the exit code. `root` defaults to
    `[mcp] path` (tests inject a temporary one)."""
    from lucy.adapters.mcp.lock import LockError

    root = settings.mcp.path if root is None else root
    action = args.mcp_action
    try:
        if action == "list":
            return list_servers(settings, out, root=root)
        if action == "tools":
            return asyncio.run(tools(settings, args.name, out, root=root, secrets=secrets))
        if action == "check":
            return asyncio.run(
                check(
                    settings,
                    args.name,
                    out,
                    call=args.call,
                    arguments=args.args,
                    root=root,
                    secrets=secrets,
                )
            )
        if action == "approve":
            ask = (lambda _: True) if args.yes else confirm
            return asyncio.run(
                approve(settings, args.name, out, confirm=ask, root=root, secrets=secrets)
            )
    except (_Usage, LockError) as exc:
        print(f"lucy mcp: {exc}", file=err)
        return 2
    raise AssertionError(f"unknown action {action!r}")
