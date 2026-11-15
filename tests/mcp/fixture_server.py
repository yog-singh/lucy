"""The MCP fixture server for the contract tests (test code: may import the SDK).

Run as a script to serve it over stdio:

    python fixture_server.py            # dual era (the SDK default: modern or legacy)
    python fixture_server.py legacy     # legacy only: refuses modern requests
    python fixture_server.py ignore-eof # keeps running after stdin closes (until signalled)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import mcp_types as types
from mcp.server import Server
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

# 1x1 transparent PNG.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


class Weather(BaseModel):
    city: str
    celsius: float


def build_server() -> MCPServer:
    server = MCPServer("lucy-fixture", version="1.2.3")

    @server.tool(description="Echo the text back. Second sentence.")
    def echo(text: str) -> str:
        return f"echo: {text}"

    @server.tool(title="Weather", description="Structured weather.", structured_output=True)
    def weather(city: str) -> Weather:
        return Weather(city=city, celsius=21.5)

    @server.tool(description="Return a tiny picture.")
    def picture() -> list:
        return ["a picture:", Image(data=PNG, format="png")]

    @server.tool(description="Always fails.")
    def broken() -> str:
        raise ToolError("the service is down")

    @server.tool(description="Sleep, then answer.")
    async def slow(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "done"

    @server.tool(
        description="Delete a thing.",
        annotations=types.ToolAnnotations(
            title="Delete thing", read_only_hint=False, destructive_hint=True
        ),
    )
    def delete_thing(thing_id: int) -> str:
        return f"deleted {thing_id}"

    @server.tool(description="Exit the server process abruptly.")
    def die() -> str:
        os._exit(3)

    @server.tool(description="Start a sleeping grandchild; return 'server_pid child_pid'.")
    def spawn_child() -> str:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"{os.getpid()} {child.pid}"

    @server.tool(description="The process id of the server.")
    def pid() -> str:
        return str(os.getpid())

    @server.tool(description="The names of the server's environment variables.")
    def env_names() -> str:
        return " ".join(sorted(os.environ))

    @server.tool(description="One environment variable's value, or <unset>.")
    def getenv(name: str) -> str:
        return os.environ.get(name, "<unset>")

    @server.tool(description="The server's working directory.")
    def cwd() -> str:
        return os.getcwd()

    @server.tool(description="Write text to stderr, `times` times.")
    def shout(text: str, times: int = 1) -> str:
        for _ in range(times):
            print(text, file=sys.stderr, flush=True)
        return "shouted"

    return server


PAGED_NAMES = [f"tool_{i}" for i in range(7)]


def build_paged_server(page_size: int = 3) -> Server:
    """A low-level server whose tools/list is paginated (MCPServer does not page)."""

    async def list_tools(
        _ctx, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        start = int(params.cursor) if params and params.cursor else 0
        names = PAGED_NAMES[start : start + page_size]
        end = start + page_size
        return types.ListToolsResult(
            tools=[
                types.Tool(name=n, description=n, input_schema={"type": "object"}) for n in names
            ],
            next_cursor=str(end) if end < len(PAGED_NAMES) else None,
        )

    async def call_tool(_ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        return types.CallToolResult(content=[types.TextContent(type="text", text=params.name)])

    return Server("lucy-paged", on_list_tools=list_tools, on_call_tool=call_tool)


async def _serve_legacy_only(server: MCPServer) -> None:
    # The SDK's servers serve both eras; its legacy serving loop (private API)
    # refuses enveloped (modern) requests, so the client must fall back.
    from mcp.server.runner import _serve_legacy_stream
    from mcp.server.stdio import stdio_server

    low = server._lowlevel_server
    async with stdio_server() as (read, write), low.lifespan(low) as state:
        await _serve_legacy_stream(
            low,
            read,
            write,
            lifespan_state=state,
            session_id=None,
            init_options=low.create_initialization_options(),
            raise_exceptions=False,
        )


if __name__ == "__main__":
    fixture = build_server()
    if sys.argv[1:] == ["legacy"]:
        asyncio.run(_serve_legacy_only(fixture))
    elif sys.argv[1:] == ["ignore-eof"]:
        fixture.run("stdio")
        time.sleep(60)  # stdin closed: keep running until signalled
    else:
        fixture.run("stdio")
