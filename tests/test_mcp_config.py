"""The `[mcp]` settings (ADR 0022): each validation rule accepts and refuses, and the
exposed-name helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lucy.core.config import (
    ConfigError,
    McpServerSettings,
    McpSettings,
    McpToolSettings,
    load_settings,
    mcp_tool_name,
)
from lucy.core.types import Risk

TIME = {"command": "uvx", "args": ["mcp-server-time==2026.1.0"]}


def refused(model: type[Any], data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    assert message in str(caught.value)


def mcp(**data: Any) -> McpSettings:
    return McpSettings.model_validate(data)


# --- Names --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("server", "tool", "name"),
    [
        ("time", "get_current_time", "time_get_current_time"),
        ("gmail", "Send-Message", "gmail_send_message"),
        ("files", "read.file v2", "files_read_file_v2"),
        ("x", "__weird__", "x_weird"),
        ("x", "ünïcode", "x_n_code"),
        ("", "!!!", ""),
    ],
)
def test_tool_name_normalisation(server: str, tool: str, name: str) -> None:
    assert mcp_tool_name(server, tool) == name


def test_exposed_tools_use_aliases_and_only_enabled_servers() -> None:
    settings = mcp(
        enabled=["time", "other"],
        servers={
            "time": {**TIME, "tools": {"get_current_time": {}, "convert_time": {"as": "convert"}}},
            "other": {**TIME, "tools": {"Look Up": {"risk": "read"}}},
            "off": {**TIME, "tools": {"x": {}}},
        },
    )
    assert settings.exposed_tool_names() == ["time_get_current_time", "convert", "other_look_up"]
    assert settings.exposed_tools()[2].server == "other"
    assert settings.exposed_tools()[2].tool == "Look Up"


def test_defaults_enable_nothing() -> None:
    settings = McpSettings()
    assert settings.enabled == [] and settings.max_tools == 8 and settings.servers == {}
    assert settings.exposed_tool_names() == []


# --- Tool settings ------------------------------------------------------------


def test_tool_defaults_to_dangerous_and_reads_as() -> None:
    tool = McpToolSettings.model_validate({"as": "send_mail"})
    assert tool.risk is Risk.DANGEROUS and tool.alias == "send_mail"
    assert McpToolSettings.model_validate({"risk": "read"}).risk is Risk.READ


def test_alias_must_be_plain() -> None:
    refused(McpToolSettings, {"as": "Send-Mail"}, "must be 1 to 64 characters of a-z, 0-9 and _")


def test_unknown_tool_option_is_refused() -> None:
    refused(McpToolSettings, {"rsk": "read"}, "Extra inputs are not permitted")


def test_preview_accepts_named_placeholders() -> None:
    tool = McpToolSettings(preview="Send an email to {to} about {subject}? {{literal}}")
    assert tool.preview is not None


@pytest.mark.parametrize(
    ("preview", "message"),
    [
        ("Send to {}?", "placeholders must be argument names"),
        ("Send to {0}?", "placeholders must be argument names"),
        ("Send to {to.__class__}?", "placeholders must be argument names"),
        ("Send to {to[0]}?", "placeholders must be argument names"),
        ("Send to {to!r}?", "use plain {name} placeholders"),
        ("Send to {to:>10}?", "use plain {name} placeholders"),
        ("Send to {to?", "is not a valid template"),
        ("Send to to}?", "is not a valid template"),
    ],
)
def test_preview_refuses_other_placeholders(preview: str, message: str) -> None:
    refused(McpToolSettings, {"preview": preview}, message)


# --- Server transport ---------------------------------------------------------


def test_local_and_remote_servers_are_accepted() -> None:
    local = McpServerSettings.model_validate(
        {**TIME, "env": {"TZ": "UTC"}, "env_keychain": {"API_KEY": "lucy-x"}, "env_from": ["HOME"]}
    )
    assert local.command == "uvx" and local.timeout == 30 and local.private is False
    remote = McpServerSettings.model_validate(
        {"url": "https://mcp.example.com/mcp", "token_keychain": "lucy-example"}
    )
    assert remote.url is not None and remote.max_result_chars == 6000
    assert McpServerSettings(url="http://127.0.0.1:8000/mcp").url
    assert McpServerSettings(url="http://localhost/mcp").url


def test_exactly_one_of_command_and_url() -> None:
    message = "set exactly one of command (a local server) or url (a remote one)"
    refused(McpServerSettings, {}, message)
    refused(McpServerSettings, {**TIME, "url": "https://x.example"}, message)


def test_remote_url_must_be_https_off_this_machine() -> None:
    refused(McpServerSettings, {"url": "http://mcp.example.com/mcp"}, "must use https")


def test_token_keychain_only_with_url() -> None:
    refused(
        McpServerSettings,
        {**TIME, "token_keychain": "x"},
        "token_keychain applies only to a remote server (url)",
    )


@pytest.mark.parametrize(
    "extra",
    [{"env_keychain": {"K": "item"}}, {"env_from": ["HOME"]}, {"args": ["a"]}, {"env": {"A": "b"}}],
)
def test_local_options_only_with_command(extra: dict[str, Any]) -> None:
    refused(
        McpServerSettings,
        {"url": "https://x.example", **extra},
        "apply only to a local server (command), not url",
    )


def test_environment_names_are_checked() -> None:
    refused(McpServerSettings, {**TIME, "env_from": ["NOT-A-NAME"]}, "is not an environment")
    refused(
        McpServerSettings,
        {**TIME, "env": {"KEY": "a"}, "env_keychain": {"KEY": "item"}},
        "environment variable 'KEY' is set more than once",
    )


# --- Version pinning ----------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("uvx", ["mcp-server-time==2026.1.0"]),
        ("uvx", ["mcp-server-time@2026.1.0", "--local-timezone=Europe/London"]),
        ("/opt/homebrew/bin/uvx", ["--python=3.12", "mcp-server-time==1.2.3rc1"]),
        ("uvx", ["--from", "some-pkg==1.0", "some-command"]),
        ("uvx", ["--with", "extra==2.0", "pkg[cli]==1.0"]),
        ("uv", ["tool", "run", "pkg==1.0"]),
        ("pipx", ["run", "--spec", "pkg==1.0", "cmd"]),
        ("npx", ["-y", "@modelcontextprotocol/server-everything@2026.7.1"]),
        ("npx", ["-y", "-p", "pkg@1.2.3", "cmd"]),
        ("bunx", ["pkg@1.2.3-beta.1"]),
        ("pnpm", ["dlx", "pkg@1.0.0"]),
        ("node", ["server.js"]),  # not a package launcher: not checked
        ("uv", ["run", "server.py"]),  # running a local script, not a package
    ],
)
def test_pinned_launches_are_accepted(command: str, args: list[str]) -> None:
    assert McpServerSettings(command=command, args=args).command == command


@pytest.mark.parametrize(
    ("command", "args", "message"),
    [
        ("uvx", ["mcp-server-time"], "must pin an exact version"),
        ("uvx", ["mcp-server-time>=1.0"], "must pin an exact version"),
        ("uvx", ["mcp-server-time@latest"], "is not a pinned version"),
        ("uvx", ["--python", "3.12", "pkg==1.0"], "options before the package must be"),
        ("uvx", ["--from", "pkg", "cmd"], "must pin an exact version"),
        ("uvx", ["--with", "extra", "pkg==1.0"], "must pin an exact version"),
        ("uvx", [], "is given no package to run"),
        ("npx", ["-y", "@modelcontextprotocol/server-everything"], "must pin an exact version"),
        ("npx", ["pkg@^1.2.0"], "must pin an exact version"),
        ("npx", ["pkg@1.2"], "must pin an exact version"),
        ("npx", ["pkg@latest"], "is not a pinned version"),
        ("npx", ["pkg", "latest"], "is not a pinned version"),
        ("pnpm", ["dlx", "pkg"], "must pin an exact version"),
        ("docker", ["run", "-i", "mcp/time:latest"], "is not a pinned version"),
    ],
)
def test_unpinned_launches_are_refused(command: str, args: list[str], message: str) -> None:
    refused(McpServerSettings, {"command": command, "args": args}, message)


# --- The whole section --------------------------------------------------------


def test_server_names_must_be_plain() -> None:
    refused(McpSettings, {"servers": {"My-Server": TIME}}, "must use only a-z, 0-9 and _")


def test_enabled_servers_must_exist_and_be_listed_once() -> None:
    refused(McpSettings, {"enabled": ["time"]}, "has no [mcp.servers.NAME] table")
    refused(McpSettings, {"enabled": ["time", "time"], "servers": {"time": TIME}}, "twice")


def test_names_colliding_with_a_built_in_are_refused() -> None:
    data = {"enabled": ["t"], "servers": {"t": {**TIME, "tools": {"x": {"as": "weather"}}}}}
    refused(McpSettings, data, "would be named 'weather', a built-in tool")
    data["servers"]["t"]["tools"]["x"] = {"as": "mcp_weather"}  # type: ignore[index]
    assert mcp(**data).exposed_tool_names() == ["mcp_weather"]


def test_names_colliding_across_servers_are_refused() -> None:
    servers = {"a": {**TIME, "tools": {"b_c": {}}}, "a_b": {**TIME, "tools": {"c": {}}}}
    refused(
        McpSettings, {"enabled": ["a", "a_b"], "servers": servers}, "would both be named 'a_b_c'"
    )
    # Only enabled servers count.
    assert mcp(enabled=["a"], servers=servers).exposed_tool_names() == ["a_b_c"]


def test_unusable_names_must_be_aliased() -> None:
    long = "x" * 70
    refused(
        McpSettings,
        {"enabled": ["t"], "servers": {"t": {**TIME, "tools": {long: {}}}}},
        "give it a name",
    )
    ok = mcp(enabled=["t"], servers={"t": {**TIME, "tools": {long: {"as": "short"}}}})
    assert ok.exposed_tool_names() == ["short"]


def test_max_tools_caps_the_enabled_servers_tools() -> None:
    tools = {f"t{i}": {} for i in range(3)}
    servers = {"s": {**TIME, "tools": tools}}
    assert len(mcp(enabled=["s"], max_tools=3, servers=servers).exposed_tool_names()) == 3
    refused(
        McpSettings,
        {"enabled": ["s"], "max_tools": 2, "servers": servers},
        "list 3 tools, more than max_tools (2)",
    )


def test_unknown_keys_are_refused_everywhere() -> None:
    refused(McpSettings, {"enable": []}, "Extra inputs are not permitted")
    refused(McpServerSettings, {**TIME, "cwd": "/"}, "Extra inputs are not permitted")


def test_mcp_section_loads_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(
        """
[llm.roles.chat]
adapter = "fake"

[mcp]
enabled = ["time"]

[mcp.servers.time]
command = "uvx"
args = ["mcp-server-time==2026.1.0"]

[mcp.servers.time.tools]
get_current_time = { risk = "read", as = "clock" }
"""
    )
    settings = load_settings(path)
    assert settings.mcp.exposed_tool_names() == ["clock"]
    path.write_text(path.read_text().replace('"mcp-server-time==2026.1.0"', '"mcp-server-time"'))
    with pytest.raises(ConfigError, match="must pin an exact version"):
        load_settings(path)
