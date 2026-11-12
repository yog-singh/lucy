"""Incremental sentence segmentation of streamed text, for TTS."""

from __future__ import annotations

from typing import Final

_TERMINATORS: Final = ".!?"
_CLOSERS: Final = "\"')]”’"
_OPENERS: Final = "\"'([“‘"
ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "st.",
        "jr.",
        "sr.",
        "vs.",
        "e.g.",
        "i.e.",
        "etc.",
    }
)
"""Lower-case words ending in "." that do not end a sentence. "etc." is treated as
non-terminal even at the end of a sentence, which only delays that sentence."""


class SentenceSplitter:
    """Turns a stream of text deltas into complete sentences.

    A sentence ends at `.`, `!` or `?` (plus any run of further terminators and
    closing quotes or brackets) followed by whitespace, or at a newline. A `.`
    in a number ("3.5") or after a known abbreviation does not end a sentence.
    Because a boundary is only decided once the character after it has arrived,
    the output does not depend on how the text is split into deltas.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add a delta; return the sentences it completed, stripped and non-empty."""
        self._buffer += text
        sentences: list[str] = []
        while (end := self._boundary()) is not None:
            sentence, self._buffer = self._buffer[:end].strip(), self._buffer[end:]
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> list[str]:
        """Return whatever text remains as a final sentence (if not blank) and reset."""
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []

    def _boundary(self) -> int | None:
        """Index just past the first complete sentence in the buffer, if there is one."""
        buffer = self._buffer
        i = 0
        while i < len(buffer):
            char = buffer[i]
            if char == "\n":
                return i + 1
            if char in _TERMINATORS:
                end = i + 1
                while end < len(buffer) and buffer[end] in _TERMINATORS + _CLOSERS:
                    end += 1
                if end == len(buffer):
                    return None  # the next character decides; wait for it
                if buffer[end].isspace() and not self._is_abbreviation(i):
                    return end
                i = end
                continue
            i += 1
        return None

    def _is_abbreviation(self, dot: int) -> bool:
        """Whether the terminator at `dot` is a "." ending a known abbreviation."""
        if self._buffer[dot] != ".":
            return False
        start = dot
        while start > 0 and not self._buffer[start - 1].isspace():
            start -= 1
        word = self._buffer[start : dot + 1].lstrip(_OPENERS).lower()
        return word in ABBREVIATIONS
