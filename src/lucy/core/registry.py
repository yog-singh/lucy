"""Adapter registry: adapters register a class under (port, name); config picks one.

The registry stores classes only. It never imports adapter modules statically;
`load_adapters` imports modules by name at startup so their `@register`
decorators run, keeping `core/` free of adapter imports (ADR 0002).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import Any, Final

PORTS: Final[frozenset[str]] = frozenset({"llm", "stt", "tts", "embedder", "memory", "tool"})
"""The valid port names."""


class RegistryError(LookupError):
    """Raised for an unknown port, an unknown adapter name or a conflicting registration."""


class Registry:
    """A mapping of port name to adapter name to adapter class."""

    def __init__(self) -> None:
        self._classes: dict[str, dict[str, type[Any]]] = {port: {} for port in PORTS}

    def _port(self, port: str) -> dict[str, type[Any]]:
        try:
            return self._classes[port]
        except KeyError:
            valid = ", ".join(sorted(PORTS))
            raise RegistryError(f"unknown port {port!r}; valid ports are: {valid}") from None

    def register[C: type[Any]](self, port: str, name: str) -> Callable[[C], C]:
        """Class decorator registering the class as adapter `name` for `port`.

        Registering the same class object again under the same port and name is a
        no-op, so importing an adapter module twice is harmless. A different class
        under an existing name raises `RegistryError`.
        """
        adapters = self._port(port)

        def decorator(cls: C) -> C:
            existing = adapters.get(name)
            if existing is cls:
                return cls
            if existing is not None:
                raise RegistryError(
                    f"{port} adapter {name!r} is already registered "
                    f"({existing.__module__}.{existing.__qualname__})"
                )
            adapters[name] = cls
            return cls

        return decorator

    def get(self, port: str, name: str) -> type[Any]:
        """Return the class registered as `name` for `port`."""
        adapters = self._port(port)
        try:
            return adapters[name]
        except KeyError:
            known = ", ".join(sorted(adapters)) or "none"
            raise RegistryError(f"no {port} adapter named {name!r}; registered: {known}") from None

    def create(self, port: str, name: str, **options: Any) -> Any:
        """Instantiate adapter `name` for `port`, passing `options` to its constructor."""
        return self.get(port, name)(**options)

    def available(self, port: str) -> list[str]:
        """Sorted names of the adapters registered for `port`."""
        return sorted(self._port(port))


_default = Registry()

register = _default.register
get = _default.get
create = _default.create
available = _default.available


def load_adapters(module_names: Iterable[str]) -> None:
    """Import the named modules so their `@register` decorators run.

    Raises `RegistryError` if a module cannot be imported.
    """
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise RegistryError(f"cannot import adapter module {module_name!r}: {exc}") from exc
