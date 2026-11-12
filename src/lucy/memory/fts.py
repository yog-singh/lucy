"""Turning arbitrary user text into safe FTS5 queries.

FTS5 has its own query syntax (quotes, `AND`/`OR`/`NOT`/`NEAR`, `*`, `^`,
column filters with `:`, parentheses), so user text is never passed through.
Instead it is split into word tokens, and each token becomes a quoted string,
which FTS5 treats as a literal phrase: operators and punctuation in the input
cannot raise or change the query's meaning.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

_WORD: Final = re.compile(r"\w+", re.UNICODE)

#: Common English function words, and verbs typical of questions to an assistant
#: ("what do you know about..."), dropped from keyword search: they match many
#: statements, so they only add noise to the ranking.
STOPWORDS: Final = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    s t d ll m re ve don doesn didn isn wasn aren weren tell know remember think
    """.split()
)

#: At most this many tokens make it into one query.
MAX_TERMS: Final = 32


def normalise(text: str) -> str:
    """Case- and width-insensitive form used for exact name matching."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text: str) -> list[str]:
    """The word tokens of `text`, casefolded, in order (duplicates kept)."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def _quote(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def any_terms_query(text: str, *, drop_stopwords: bool = True) -> str | None:
    """An FTS5 query matching rows containing any word of `text`, or None if there are none.

    Stopwords are dropped (unless that would leave nothing, or `drop_stopwords`
    is false) and duplicates removed; at most `MAX_TERMS` terms are kept.
    """
    words = list(dict.fromkeys(tokens(text)))
    if drop_stopwords:
        content = [w for w in words if w not in STOPWORDS]
        words = content or words
    if not words:
        return None
    return " OR ".join(_quote(w) for w in words[:MAX_TERMS])


def phrase_query(text: str) -> str | None:
    """An FTS5 query matching `text`'s words as one phrase, or None if it has no words."""
    words = tokens(text)
    if not words:
        return None
    return _quote(" ".join(words))
