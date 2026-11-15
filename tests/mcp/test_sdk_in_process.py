"""Contract tests: Lucy's SDK boundary against an in-process SDK server, both eras."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from lucy.adapters.mcp import sdk

from .fixture_server import PAGED_NAMES, build_paged_server, build_server

ERAS = ["auto", "legacy"]


@asynccontextmanager
async def connected(era: sdk.Era, server: object | None = None) -> AsyncIterator[sdk.McpConnection]:
    conn = await sdk.connect_in_process(server or build_server(), era=era)
    try:
        yield conn
    finally:
        await conn.aclose()


@pytest.mark.parametrize("era", ERAS)
async def test_server_info_reports_the_negotiated_era(era: sdk.Era) -> None:
    async with connected(era) as conn:
        info = conn.info
    assert info.era == ("modern" if era == "auto" else "legacy")
    assert info.protocol_version  # whatever the SDK negotiated; Lucy names none
    assert (info.name, info.version) == ("lucy-fixture", "1.2.3")
    assert "tools" in info.capabilities


@pytest.mark.parametrize("era", ERAS)
async def test_list_tools_arrive_in_lucy_types(era: sdk.Era) -> None:
    async with connected(era) as conn:
        tools = {t.name: t for t in await conn.list_tools()}
    assert {"echo", "weather", "picture", "broken", "slow", "delete_thing"} <= set(tools)
    echo = tools["echo"]
    assert isinstance(echo, sdk.RemoteTool)
    assert echo.description == "Echo the text back. Second sentence."
    assert echo.input_schema["properties"]["text"]["type"] == "string"
    assert echo.input_schema["required"] == ["text"]
    assert tools["weather"].title == "Weather"
    assert tools["weather"].output_schema is not None
    assert set(tools["weather"].output_schema["properties"]) == {"city", "celsius"}
    assert tools["picture"].output_schema is None
    assert tools["delete_thing"].annotations == {
        "title": "Delete thing",
        "readOnlyHint": False,
        "destructiveHint": True,
    }
    assert tools["echo"].annotations == {}


@pytest.mark.parametrize("era", ERAS)
async def test_list_tools_follows_pagination(era: sdk.Era) -> None:
    async with connected(era, build_paged_server(page_size=3)) as conn:
        names = [t.name for t in await conn.list_tools()]
    assert names == PAGED_NAMES


@pytest.mark.parametrize("era", ERAS)
async def test_call_result_kinds(era: sdk.Era) -> None:
    async with connected(era) as conn:
        text = await conn.call_tool("echo", {"text": "hi"}, timeout=5)
        structured = await conn.call_tool("weather", {"city": "Pune"}, timeout=5)
        picture = await conn.call_tool("picture", {}, timeout=5)
        broken = await conn.call_tool("broken", None, timeout=5)
    assert isinstance(text, sdk.CallResult)
    assert text.texts == ("echo: hi",)
    assert not text.is_error and text.omitted == {}
    assert structured.structured == {"city": "Pune", "celsius": 21.5}
    assert picture.texts == ("a picture:",)
    assert picture.omitted == {"image": 1}
    assert broken.is_error
    assert "the service is down" in broken.texts[0]


@pytest.mark.parametrize("era", ERAS)
async def test_timeout_raises_lucy_error_and_connection_stays_usable(era: sdk.Era) -> None:
    async with connected(era) as conn:
        with pytest.raises(sdk.McpTimeoutError):
            await conn.call_tool("slow", {"seconds": 5}, timeout=0.1)
        after = await conn.call_tool("echo", {"text": "still here"}, timeout=5)
    assert after.texts == ("echo: still here",)


@pytest.mark.parametrize("era", ERAS)
async def test_unknown_tool_and_bad_arguments_are_tool_errors(era: sdk.Era) -> None:
    # The SDK's MCPServer reports both as an isError result, not a protocol
    # error; another server may answer with a JSON-RPC error (McpProtocolError).
    async with connected(era) as conn:
        unknown = await conn.call_tool("no_such_tool", {}, timeout=5)
        bad = await conn.call_tool("delete_thing", {"thing_id": "not a number"}, timeout=5)
    assert unknown.is_error and "no_such_tool" in unknown.texts[0]
    assert bad.is_error and "thing_id" in bad.texts[0]


async def test_aclose_is_idempotent_and_calls_after_close_fail_as_lucy_errors() -> None:
    conn = await sdk.connect_in_process(build_server())
    await conn.aclose()
    await conn.aclose()
    with pytest.raises(sdk.McpError):
        await conn.call_tool("echo", {"text": "x"}, timeout=1)


async def test_era_must_be_auto_or_legacy() -> None:
    with pytest.raises(ValueError):
        await sdk.connect_in_process(build_server(), era="2026-07-28")  # type: ignore[arg-type]
