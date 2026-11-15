"""Secrets for MCP servers, read from the macOS Keychain (ADR 0022).

A secret is read when a server starts (and again on each restart), handed to the
server and dropped: it is never written to disk, logged, or put in an error
message or repr. Only the Keychain item's service name is ever shown.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Mapping
from typing import Protocol

__all__ = ["FakeSecrets", "KeychainSecrets", "SecretError", "Secrets"]

_SECURITY = "/usr/bin/security"
_TIMEOUT = 10.0


class SecretError(Exception):
    """A secret could not be read. The message names the item, never a value."""


class Secrets(Protocol):
    async def get(self, service: str) -> str:
        """The secret stored under `service`; raises `SecretError`."""
        ...


class KeychainSecrets:
    """Generic passwords from the login Keychain, by service name
    (`security find-generic-password -s SERVICE -w`), read off the event loop."""

    async def get(self, service: str) -> str:
        return await asyncio.to_thread(self._read, service)

    @staticmethod
    def _read(service: str) -> str:
        try:
            done = subprocess.run(
                [_SECURITY, "find-generic-password", "-s", service, "-w"],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SecretError(
                f"could not read Keychain item {service!r}: {type(exc).__name__}"
            ) from None
        if done.returncode != 0:
            # stderr names the item and the reason; it never holds the secret.
            raise SecretError(
                f"Keychain item {service!r} not found (add it with: security "
                f"add-generic-password -s {service} -a lucy -w)"
            )
        value = done.stdout.rstrip("\n")
        if not value:
            raise SecretError(f"Keychain item {service!r} is empty")
        return value


class FakeSecrets:
    """Secrets from a dict, for tests. Records which items were read."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)
        self.reads: list[str] = []

    async def get(self, service: str) -> str:
        self.reads.append(service)
        if service not in self._values:
            raise SecretError(f"Keychain item {service!r} not found")
        return self._values[service]

    def __repr__(self) -> str:
        return f"FakeSecrets({sorted(self._values)})"
