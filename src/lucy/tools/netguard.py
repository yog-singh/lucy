"""Which web addresses `fetch_page` may connect to.

Page content is untrusted: a page could try to make the model fetch an
internal address (the router, local services, the SearXNG instance, the
Ollama API). So a URL is fetched only if it is http(s) without credentials
and its host resolves exclusively to public addresses, unless the host is on
the configured allow-list. The caller then connects to the checked address
itself, so the name cannot resolve somewhere else between check and connect.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final
from urllib.parse import SplitResult, urlsplit

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str, int], Awaitable[list[str]]]
"""Resolves a host name and port to IP address strings."""

BLOCKED_NAMES: Final = ("localhost",)
BLOCKED_SUFFIXES: Final = (".localhost", ".local", ".internal", ".lan", ".home.arpa")
"""Names refused before resolution (local by convention), unless allow-listed."""

_NAT64: Final = ipaddress.ip_network("64:ff9b::/96")


class BlockedAddress(Exception):
    """The URL may not be fetched; the message says why, for the model."""


@dataclass(frozen=True)
class Target:
    """A checked URL: where to connect, and the host name for the Host header and TLS."""

    url: SplitResult
    host: str
    port: int
    address: IPAddress


async def system_resolver(host: str, port: int) -> list[str]:
    """Resolve with the operating system's resolver (getaddrinfo)."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def _embedded_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    if address in _NAT64:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None


def address_problem(address: IPAddress) -> str | None:
    """Why `address` is not a public address, or None if it is."""
    if isinstance(address, ipaddress.IPv6Address):
        if address.scope_id:
            return "a scoped (link-local) address"
        embedded = _embedded_ipv4(address)
        if embedded is not None:
            problem = address_problem(embedded)
            if problem is not None:
                return problem
    checks = (
        ("is_unspecified", "an unspecified address"),
        ("is_loopback", "a loopback address"),
        ("is_link_local", "a link-local address"),
        ("is_multicast", "a multicast address"),
        ("is_private", "a private-network address"),
        ("is_reserved", "a reserved address"),
    )
    for attribute, label in checks:
        if getattr(address, attribute):
            return label
    if isinstance(address, ipaddress.IPv6Address) and address in ipaddress.ip_network("fec0::/10"):
        return "a site-local address"
    if not address.is_global:
        return "a non-public address"
    return None


class AllowList:
    """Hosts that may be private: exact host names, IP addresses or networks."""

    def __init__(self, entries: Sequence[str] = ()) -> None:
        self.names: set[str] = set()
        self.networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for entry in entries:
            entry = entry.strip().lower().rstrip(".")
            if not entry:
                continue
            try:
                self.networks.append(ipaddress.ip_network(entry.strip("[]"), strict=False))
            except ValueError:
                self.names.add(entry)

    def allows_name(self, host: str) -> bool:
        return host in self.names

    def allows_address(self, address: IPAddress) -> bool:
        return any(address in network for network in self.networks)


def parse_url(url: str) -> SplitResult:
    """Check the URL's form: http(s), a host, no credentials. Raises `BlockedAddress`."""
    try:
        parts = urlsplit(url.strip())
        _ = parts.port  # raises ValueError for a bad port
    except ValueError as exc:
        raise BlockedAddress(f"That is not a valid web address ({exc}).") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise BlockedAddress("Only http and https addresses can be fetched.")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise BlockedAddress("Addresses with a user name or password are not fetched.")
    if not parts.hostname:
        raise BlockedAddress("That web address has no host name.")
    return parts


async def check_url(
    url: str, *, resolver: Resolver = system_resolver, allow: AllowList | None = None
) -> Target:
    """Validate `url` and pick the address to connect to. Raises `BlockedAddress`."""
    allow = allow or AllowList()
    parts = parse_url(url)
    host = (parts.hostname or "").lower().rstrip(".")
    port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    name_allowed = allow.allows_name(host)

    try:
        literal: IPAddress | None = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        addresses: list[IPAddress] = [literal]
    else:
        if not name_allowed and (host in BLOCKED_NAMES or host.endswith(BLOCKED_SUFFIXES)):
            raise BlockedAddress(f"{host} is a local name; only public web pages are fetched.")
        try:
            resolved = await resolver(host, port)
        except (OSError, UnicodeError) as exc:
            raise BlockedAddress(f"The host {host} could not be found ({exc}).") from None
        addresses = []
        for text in resolved:
            try:
                addresses.append(ipaddress.ip_address(text))
            except ValueError:
                continue
        if not addresses:
            raise BlockedAddress(f"The host {host} could not be found.")

    if not name_allowed:
        for address in addresses:
            problem = address_problem(address)
            if problem is not None and not allow.allows_address(address):
                raise BlockedAddress(
                    f"Refused: {host} points to {problem}. Only public web pages are fetched."
                )
    return Target(url=parts, host=host, port=port, address=addresses[0])
