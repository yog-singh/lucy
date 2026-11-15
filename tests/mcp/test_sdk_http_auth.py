"""The bearer token goes only to the configured origin (no network)."""

from __future__ import annotations

import httpx2
import pytest

from lucy.adapters.mcp import sdk
from lucy.adapters.mcp.sdk import _OriginBearerAuth


@pytest.mark.parametrize(
    ("url", "sent"),
    [
        ("https://mcp.example.com/mcp", True),
        ("https://mcp.example.com:443/other", True),
        ("https://MCP.example.com/mcp", True),
        ("http://mcp.example.com/mcp", False),
        ("https://mcp.example.com:8443/mcp", False),
        ("https://evil.example.net/mcp", False),
    ],
)
def test_token_only_for_the_configured_origin(url: str, sent: bool) -> None:
    auth = _OriginBearerAuth("https://mcp.example.com/mcp", "s3cret")
    request = next(auth.auth_flow(httpx2.Request("POST", url)))
    assert (request.headers.get("Authorization") == "Bearer s3cret") is sent


async def test_non_http_url_is_refused() -> None:
    with pytest.raises(ValueError):
        await sdk.connect_http(sdk.HttpServerSpec(url="file:///etc/passwd"))
