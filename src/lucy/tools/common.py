"""Helpers shared by the internet tools: the HTTP client, result capping, error texts."""

from __future__ import annotations

import re
from typing import Any, Final

import httpx

from lucy.core.types import ToolOutput

USER_AGENT: Final = "Lucy/0.1 (local personal assistant)"
"""Sent with every request. Deliberately generic: no user, machine or project details."""

TRUNCATION_NOTE: Final = "[Truncated: the rest was cut to keep the result short.]"


def make_client(
    timeout: float, transport: httpx.AsyncBaseTransport | None = None, **options: Any
) -> httpx.AsyncClient:
    """An HTTP client with the generic User-Agent and no proxy or netrc settings from
    the environment (so requests go exactly where the tool decides)."""
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(timeout),
        transport=transport,
        trust_env=False,
        **options,
    )


def required_text(args: dict[str, Any], name: str) -> str | None:
    """The non-empty string argument `name`, stripped, or None."""
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def squash(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def tidy_paragraphs(text: str) -> str:
    """Collapse spaces within lines and runs of blank lines to one blank line, and drop
    invisible characters (zero-width spaces, stray byte-order marks)."""
    text = re.sub("[\u200b\u200c\u200d\u2060\ufeff\u00ad]", "", text)
    lines = [re.sub(r"[ \t\f\v\r]+", " ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def shorten(text: str, max_chars: int, *, note: str = "…") -> str:
    """`text` cut at a word boundary to at most `max_chars` characters including `note`."""
    if len(text) <= max_chars:
        return text
    room = max(0, max_chars - len(note) - 1)
    cut = text[:room]
    if room < len(text) and not text[room].isspace() and " " in cut:
        cut = cut[: cut.rindex(" ")]
    cut = cut.rstrip()
    joiner = "\n" if note.startswith("[") else ""
    return f"{cut}{joiner}{note}" if cut else note[:max_chars]


def cap(text: str, max_chars: int) -> str:
    """`text` limited to `max_chars`, with a note saying so when it was cut."""
    return shorten(text, max_chars, note=TRUNCATION_NOTE)


def describe_http_error(exc: Exception, service: str) -> str:
    """A short, actionable message for a failed request to `service`."""
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return f"{service} did not answer in time. Try again later or answer without it."
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return f"{service} answered with an error (HTTP {code})."
    if isinstance(exc, httpx.TransportError):
        return f"{service} is not reachable."
    return f"{service} returned something unexpected ({type(exc).__name__})."


def error(message: str) -> ToolOutput:
    """An error result for the model."""
    return ToolOutput(content=message, is_error=True)
