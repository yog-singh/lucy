"""Keyword matching for eval checks: simple and transparent on purpose.

A keyword matches when it occurs in the text as whole words, ignoring case and
runs of whitespace, optionally followed by a plural or possessive ending ("s",
"es", "'s"): "cat" matches "Cats" and "cat's" but not "category"; "dark roast"
matches "Dark  roast". A keyword is a phrase if it has spaces. Curly apostrophes
count as straight ones. Nothing else (no stemming, no synonyms): list the
alternatives in `any` instead.

A `Match` passes when every `require` keyword occurs, at least one `any`
keyword occurs (if any are given), and no `forbid` keyword occurs.
"""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field


def normalise(text: str) -> str:
    """Lower case, straight apostrophes, single spaces."""
    return " ".join(text.replace("’", "'").replace("‘", "'").lower().split())


@lru_cache(maxsize=1024)
def _pattern(keyword: str) -> re.Pattern[str]:
    words = [re.escape(w) for w in normalise(keyword).split()]
    if not words:
        raise ValueError("a keyword must not be blank")
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?:s|es|'s)?(?!\w)")


def has_keyword(text: str, keyword: str) -> bool:
    """Whether `keyword` occurs in `text` (see the module docstring)."""
    return _pattern(keyword).search(normalise(text)) is not None


class Match(BaseModel):
    """Keywords a text must contain (`require`, all; `any`, one of) and must not (`forbid`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    require: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list, alias="any")
    forbid: list[str] = Field(default_factory=list)

    def matches(self, text: str) -> bool:
        return (
            all(has_keyword(text, k) for k in self.require)
            and (not self.any_of or any(has_keyword(text, k) for k in self.any_of))
            and not any(has_keyword(text, k) for k in self.forbid)
        )

    def mentions(self, text: str) -> bool:
        """Whether `text` contains any wanted keyword (to show near misses)."""
        return any(has_keyword(text, k) for k in [*self.require, *self.any_of])

    def failures(self, text: str) -> list[str]:
        """Why `text` does not match, one phrase per broken rule (empty if it matches)."""
        problems = [f"missing {k!r}" for k in self.require if not has_keyword(text, k)]
        if self.any_of and not any(has_keyword(text, k) for k in self.any_of):
            problems.append("none of " + ", ".join(repr(k) for k in self.any_of))
        problems += [f"contains {k!r}" for k in self.forbid if has_keyword(text, k)]
        return problems

    def describe(self) -> str:
        parts = []
        if self.require:
            parts.append("with " + " and ".join(repr(k) for k in self.require))
        if self.any_of:
            parts.append("with one of " + ", ".join(repr(k) for k in self.any_of))
        if self.forbid:
            parts.append("without " + " or ".join(repr(k) for k in self.forbid))
        return " ".join(parts) or "(anything)"
