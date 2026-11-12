"""The `fetch_page` tool: results, failures and every safety rule.

No network is used: an `httpx.MockTransport` stands in for the web and a fake
resolver for DNS. Requests reach the transport addressed to the checked IP,
which is how the tests see that the tool connects to the address it checked.
"""

from __future__ import annotations

import asyncio
import gzip
import ipaddress
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from lucy.core.config import FetchPageSettings
from lucy.tools.common import TRUNCATION_NOTE, USER_AGENT
from lucy.tools.fetch_page import UNTRUSTED_NOTE, FetchPageTool
from lucy.tools.netguard import AllowList, BlockedAddress, address_problem, check_url

DATA = Path(__file__).parent / "data"
PUBLIC = "93.184.215.14"
PUBLIC6 = "2606:2800:21f:cb07:6820:80da:af6b:8b2c"
PAGE = b"<html><head><title>A page</title></head><body><p>Hello &amp; welcome.</p></body></html>"

Handler = Callable[[httpx.Request], httpx.Response]


class Resolver:
    """Fake DNS: a name maps to a list of addresses, or to a sequence of answers."""

    def __init__(self, table: Mapping[str, list[str] | list[list[str]]] | None = None) -> None:
        self.table = dict(table or {"example.com": [PUBLIC]})
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append((host, port))
        if host not in self.table:
            raise OSError("Name or service not known")
        answer = self.table[host]
        if answer and isinstance(answer[0], list):  # successive answers (rebinding)
            index = sum(1 for h, _ in self.calls if h == host) - 1
            return list(answer[min(index, len(answer) - 1)])
        return list(answer)  # type: ignore[arg-type]


async def stream(body: bytes, chunk: int = 4096) -> AsyncIterator[bytes]:
    """`body` as a streamed response body (a bytes body would be read eagerly by httpx)."""
    for start in range(0, len(body), chunk):
        yield body[start : start + chunk]


def html(
    body: bytes = PAGE, *, content_type: str = "text/html; charset=utf-8", **headers: str
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        headers_ = {"content-type": content_type, **headers}
        return httpx.Response(200, content=stream(body), headers=headers_)

    return handler


def make_tool(
    handler: Handler,
    *,
    resolver: Resolver | None = None,
    seen: list[httpx.Request] | None = None,
    **options: Any,
) -> FetchPageTool:
    def record(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return handler(request)

    return FetchPageTool(
        FetchPageSettings(**options),
        transport=httpx.MockTransport(record),
        resolver=resolver or Resolver(),
    )


def never(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request expected, got {request.url}")


# --- Normal results -------------------------------------------------------------


async def test_fetches_page_text_from_the_checked_address() -> None:
    seen: list[httpx.Request] = []
    tool = make_tool(html(), seen=seen)
    out = await tool.run({"url": "https://example.com/a/b?x=1#frag"})
    await tool.aclose()
    assert not out.is_error
    assert out.content == (
        "Web page: A page\nAddress: https://example.com/a/b?x=1#frag\n"
        f"{UNTRUSTED_NOTE}\n\nHello & welcome."
    )
    (request,) = seen
    assert str(request.url) == f"https://{PUBLIC}/a/b?x=1"  # pinned to the checked IP
    assert request.headers["host"] == "example.com"
    assert request.extensions["sni_hostname"] == "example.com"  # TLS checks the real name
    assert request.headers["user-agent"] == USER_AGENT


async def test_ipv6_address_and_custom_port_are_pinned() -> None:
    seen: list[httpx.Request] = []
    resolver = Resolver({"v6.example": [PUBLIC6]})
    out = await make_tool(html(), resolver=resolver, seen=seen).run(
        {"url": "http://v6.example:8081/"}
    )
    assert not out.is_error
    assert str(seen[0].url) == f"http://[{PUBLIC6}]:8081/"
    assert seen[0].headers["host"] == "v6.example:8081"
    assert "sni_hostname" not in seen[0].extensions
    assert resolver.calls == [("v6.example", 8081)]


async def test_address_without_scheme_assumes_https() -> None:
    seen: list[httpx.Request] = []
    out = await make_tool(html(), seen=seen).run({"url": "example.com/page"})
    assert not out.is_error and seen[0].url.scheme == "https"


async def test_plain_text_and_json_are_read() -> None:
    out = await make_tool(html(b"line one\n\n\n\nline two", content_type="text/plain")).run(
        {"url": "http://example.com/notes.txt"}
    )
    assert out.content.endswith("\n\nline one\n\nline two")
    assert out.content.startswith("Web page\nAddress: http://example.com/notes.txt")
    out = await make_tool(html(b'{"a": 1}', content_type="application/json")).run(
        {"url": "http://example.com/x.json"}
    )
    assert out.content.endswith('{"a": 1}')


async def test_recorded_real_page() -> None:
    body = (DATA / "example_com.html").read_bytes()
    out = await make_tool(html(body)).run({"url": "https://example.com/"})
    assert out.content.startswith("Web page: Example Domain\n")
    assert "This domain is for use in documentation examples" in out.content


async def test_long_text_is_capped_with_a_note() -> None:
    body = b"<p>" + b"word " * 5000 + b"</p>"
    out = await make_tool(html(body), max_chars=1000).run({"url": "http://example.com/"})
    assert len(out.content) <= 1000
    assert out.content.endswith(TRUNCATION_NOTE)
    assert out.content.startswith("Web page\nAddress: http://example.com/")


async def test_redirect_is_followed_and_checked_again() -> None:
    seen: list[httpx.Request] = []
    resolver = Resolver({"example.com": [PUBLIC], "www.example.org": ["93.184.215.15"]})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "example.com":
            return httpx.Response(301, headers={"location": "https://www.example.org/final"})
        return html()(request)

    out = await make_tool(handler, resolver=resolver, seen=seen).run({"url": "http://example.com/"})
    assert not out.is_error
    assert "Address: https://www.example.org/final" in out.content
    assert [str(r.url) for r in seen] == [f"http://{PUBLIC}/", "https://93.184.215.15/final"]
    assert [h for h, _ in resolver.calls] == ["example.com", "www.example.org"]


async def test_relative_redirect() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "/new"})
        return html()(request)

    out = await make_tool(handler, seen=seen).run({"url": "https://example.com/old"})
    assert "Address: https://example.com/new" in out.content
    assert seen[-1].headers["host"] == "example.com"


# --- Failures -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (lambda r: httpx.Response(404), "The page answered with HTTP 404 (Not Found)."),
        (lambda r: httpx.Response(302), "The page redirected without saying where to."),
        (html(b"<script>app()</script>"), "The page has no readable text"),
        (html(b"x", **{"content-encoding": "br"}), "unsupported compression (br)"),
    ],
)
async def test_page_failures_are_error_results(handler: Handler, message: str) -> None:
    out = await make_tool(handler).run({"url": "https://example.com/"})
    assert out.is_error and message in out.content


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (httpx.ConnectError("refused"), "The web page is not reachable."),
        (httpx.ReadTimeout("slow"), "The web page did not answer in time."),
    ],
)
async def test_transport_failures_are_error_results(exc: Exception, message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    out = await make_tool(handler).run({"url": "https://example.com/"})
    assert out.is_error and out.content.startswith(message)


async def test_total_time_is_bounded() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    tool = FetchPageTool(
        FetchPageSettings(timeout=0.05), transport=httpx.MockTransport(slow), resolver=Resolver()
    )
    out = await tool.run({"url": "https://example.com/"})
    assert out.is_error and "did not answer in time" in out.content


async def test_slow_dns_counts_against_the_time_limit() -> None:
    async def slow_resolver(host: str, port: int) -> list[str]:
        await asyncio.sleep(5)
        return [PUBLIC]

    tool = FetchPageTool(
        FetchPageSettings(timeout=0.05),
        transport=httpx.MockTransport(never),
        resolver=slow_resolver,
    )
    out = await tool.run({"url": "https://example.com/"})
    assert out.is_error and "did not answer in time" in out.content


async def test_unknown_host() -> None:
    out = await make_tool(never).run({"url": "https://no-such-host.example/"})
    assert out.is_error and out.content.startswith("The host no-such-host.example could not be")


async def test_needs_a_url() -> None:
    for args in ({}, {"url": ""}, {"url": None}):
        out = await make_tool(never).run(args)
        assert out.is_error and "needs a 'url'" in out.content


# --- Safety: schemes, credentials, names ----------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "gopher://example.com/",
        "data://text/plain,hello",
        "ws://example.com/socket",
    ],
)
async def test_only_http_and_https(url: str) -> None:
    out = await make_tool(never).run({"url": url})
    assert out.is_error and out.content == "Only http and https addresses can be fetched."


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@example.com/",
        "https://user@example.com/",
        "https://:pw@example.com/",
        "https://example.com@127.0.0.1/",
    ],
)
async def test_credentials_are_refused(url: str) -> None:
    out = await make_tool(never).run({"url": url})
    assert out.is_error and "user name or password" in out.content


async def test_malformed_addresses() -> None:
    out = await make_tool(never).run({"url": "http://example.com:99999/"})
    assert out.is_error and "not a valid web address" in out.content
    out = await make_tool(never).run({"url": "http:///path"})
    assert out.is_error and "no host name" in out.content


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LOCALHOST.",
        "api.localhost",
        "router.local",
        "db.internal",
        "nas.lan",
        "x.home.arpa",
    ],
)
async def test_local_names_are_refused_without_resolving(host: str) -> None:
    resolver = Resolver({host.lower().rstrip("."): [PUBLIC]})
    out = await make_tool(never, resolver=resolver).run({"url": f"http://{host}:11434/api/tags"})
    assert out.is_error and "is a local name" in out.content
    assert resolver.calls == []


# --- Safety: address classes ----------------------------------------------------

BLOCKED_IPV4 = {
    "127.0.0.1": "a loopback address",
    "127.255.0.9": "a loopback address",
    "10.1.2.3": "a private-network address",
    "172.16.5.4": "a private-network address",
    "192.168.1.1": "a private-network address",
    "169.254.169.254": "a link-local address",
    "224.0.0.251": "a multicast address",
    "240.0.0.1": "a private-network address",  # reserved; Python also counts it private
    "255.255.255.255": "a private-network address",
    "0.0.0.0": "an unspecified address",
    "100.64.0.1": "a non-public address",  # carrier-grade NAT
    "192.0.2.10": "a private-network address",  # documentation
}

BLOCKED_IPV6 = {
    "::1": "a loopback address",
    "::": "an unspecified address",
    "fe80::1": "a link-local address",
    "fc00::1": "a private-network address",
    "fd12:3456::1": "a private-network address",
    "ff02::1": "a multicast address",
    "fec0::1": "a site-local address",
    "2001:db8::1": "a private-network address",  # documentation
    "::ffff:127.0.0.1": "a loopback address",  # IPv4-mapped
    "::ffff:192.168.0.1": "a private-network address",
    "64:ff9b::a00:1": "a private-network address",  # NAT64 of 10.0.0.1
    "2002:7f00:1::": "a loopback address",  # 6to4 of 127.0.0.1
    "4000::1": "a reserved address",
}


@pytest.mark.parametrize(("address", "label"), sorted({**BLOCKED_IPV4, **BLOCKED_IPV6}.items()))
def test_address_classes(address: str, label: str) -> None:
    assert address_problem(ipaddress.ip_address(address)) == label


@pytest.mark.parametrize("address", [PUBLIC, "8.8.8.8", "1.1.1.1", PUBLIC6, "2a00:1450:4001::1"])
def test_public_addresses_pass(address: str) -> None:
    assert address_problem(ipaddress.ip_address(address)) is None


@pytest.mark.parametrize("address", sorted(BLOCKED_IPV4))
async def test_ipv4_literals_are_refused(address: str) -> None:
    out = await make_tool(never).run({"url": f"http://{address}/"})
    assert out.is_error and out.content.startswith(f"Refused: {address} points to")


@pytest.mark.parametrize("address", sorted(BLOCKED_IPV6))
async def test_ipv6_literals_are_refused(address: str) -> None:
    out = await make_tool(never).run({"url": f"http://[{address}]:8080/"})
    assert out.is_error and out.content.startswith("Refused:")


@pytest.mark.parametrize("answer", [["10.0.0.5"], [PUBLIC, "127.0.0.1"], ["fd00::5"], ["::1"]])
async def test_name_resolving_to_a_private_address_is_refused(answer: list[str]) -> None:
    resolver = Resolver({"sneaky.example": answer})
    out = await make_tool(never, resolver=resolver).run({"url": "https://sneaky.example/"})
    assert out.is_error and out.content.startswith("Refused: sneaky.example points to")


async def test_redirect_to_a_private_address_is_refused() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1:11434/api/tags"})

    out = await make_tool(handler, seen=seen).run({"url": "https://example.com/"})
    assert out.is_error and "Refused: 127.0.0.1 points to a loopback address" in out.content
    assert len(seen) == 1


async def test_redirect_to_a_name_that_resolves_privately_is_refused() -> None:
    resolver = Resolver({"example.com": [PUBLIC], "evil.example": ["192.168.1.1"]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"location": "http://evil.example/admin"})

    out = await make_tool(handler, resolver=resolver).run({"url": "https://example.com/"})
    assert out.is_error and "evil.example points to a private-network address" in out.content


async def test_rebinding_between_hops_is_caught() -> None:
    """The name resolves publicly first and privately on the next hop."""
    resolver = Resolver({"rebind.example": [[PUBLIC], ["127.0.0.1"]]})
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    out = await make_tool(handler, resolver=resolver, seen=seen).run(
        {"url": "http://rebind.example/"}
    )
    assert out.is_error and "loopback" in out.content
    assert [str(r.url) for r in seen] == [f"http://{PUBLIC}/"]


async def test_redirect_limit() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": f"/hop{len(seen)}"})

    out = await make_tool(handler, seen=seen, max_redirects=3).run({"url": "https://example.com/"})
    assert out.is_error and out.content == "Gave up after 3 redirects."
    assert len(seen) == 4


# --- Safety: size and content type ----------------------------------------------


async def test_size_limit_stops_reading_the_stream() -> None:
    sent: list[int] = []

    async def body() -> AsyncIterator[bytes]:
        yield b"<html><title>Big</title><body><p>"
        for index in range(1000):
            sent.append(index)
            yield b"x" * 1023 + b" "

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body(), headers={"content-type": "text/html"})

    out = await make_tool(handler, max_bytes=10_000).run({"url": "https://example.com/"})
    assert not out.is_error
    assert "(Only the first 10000 bytes of the page were read.)" in out.content
    assert len(sent) < 20  # the rest of the body was never pulled


async def test_declared_oversize_body_is_marked_cut_short() -> None:
    body = b"<p>" + b"a " * 600 + b"</p>"
    handler = html(body, **{"content-length": str(len(body))})
    out = await make_tool(handler, max_bytes=1024).run({"url": "https://example.com/"})
    assert "(Only the first 1024 bytes" in out.content


async def test_compressed_body_cannot_expand_past_the_limit() -> None:
    bomb = gzip.compress(b"<p>" + b"a" * 50_000_000)
    assert len(bomb) < 100_000
    handler = html(bomb, **{"content-encoding": "gzip"})
    out = await make_tool(handler, max_bytes=5000, max_chars=200_000).run(
        {"url": "https://example.com/"}
    )
    assert not out.is_error
    assert "(Only the first 5000 bytes" in out.content
    assert len(out.content) < 6000


async def test_gzip_page_is_decoded() -> None:
    handler = html(gzip.compress(PAGE), **{"content-encoding": "gzip"})
    out = await make_tool(handler).run({"url": "https://example.com/"})
    assert out.content.endswith("Hello & welcome.")


@pytest.mark.parametrize(
    "content_type", ["image/png", "application/octet-stream", "application/pdf", "video/mp4", ""]
)
async def test_non_text_content_is_refused(content_type: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": content_type} if content_type else {}
        return httpx.Response(200, content=stream(b"\x89PNG\r\n"), headers=headers)

    out = await make_tool(handler).run({"url": "https://example.com/file"})
    assert out.is_error and "not a text page" in out.content


# --- Safety: the allow-list -----------------------------------------------------


async def test_allow_listed_name_may_be_private() -> None:
    resolver = Resolver({"homeassistant.local": ["192.168.1.20"], "other.local": ["192.168.1.21"]})
    tool = make_tool(html(), resolver=resolver, allow_private_hosts=["HomeAssistant.local"])
    out = await tool.run({"url": "http://homeassistant.local:8123/"})
    assert not out.is_error
    out = await tool.run({"url": "http://other.local/"})
    assert out.is_error and "local name" in out.content


async def test_allow_listed_address_or_network() -> None:
    tool = make_tool(html(), allow_private_hosts=["192.168.1.0/24", "127.0.0.1", "[::1]"])
    assert not (await tool.run({"url": "http://192.168.1.77/"})).is_error
    assert not (await tool.run({"url": "http://127.0.0.1:8000/"})).is_error
    assert not (await tool.run({"url": "http://[::1]:8000/"})).is_error
    assert (await tool.run({"url": "http://192.168.2.1/"})).is_error
    assert (await tool.run({"url": "http://127.0.0.2/"})).is_error


async def test_allow_list_does_not_cover_redirect_targets() -> None:
    resolver = Resolver({"ha.lan": ["192.168.1.20"]})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "ha.lan":
            return httpx.Response(302, headers={"location": "http://10.0.0.1/"})
        return html()(request)

    tool = make_tool(handler, resolver=resolver, allow_private_hosts=["ha.lan"])
    out = await tool.run({"url": "http://ha.lan/"})
    assert out.is_error and "10.0.0.1 points to a private-network address" in out.content


async def test_check_url_directly() -> None:
    target = await check_url("https://Example.COM./x", resolver=Resolver())
    assert (target.host, target.port, str(target.address)) == ("example.com", 443, PUBLIC)
    with pytest.raises(BlockedAddress):
        await check_url("http://localhost/", resolver=Resolver(), allow=AllowList(["other"]))
