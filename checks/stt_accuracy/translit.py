"""The romanised-Hindi spelling fold used for scoring, on top of Lucy's `to_latin`.

`to_latin` (Devanagari to Latin) lives in `lucy.voice.translit`, the one copy that
both Lucy's `indic_flex` speech adapter and this check use; it is re-exported here.

`fold(text)` is for scoring only. Romanised Hindi has no standard spelling, so the
same word is written several ways; `fold` maps the common variants to one key,
applied to reference and output alike:
  - lower case, accents stripped, only letters and digits kept;
  - ph f, w v, z j, q k, x ks; chh, chch, cch ch;
  - ee, ii i; oo, uu u; aa a; ai, ei, ay e; au, ou o (hai = he, mein = men);
  - a doubled letter written once (accha = acha);
  - a final h after a vowel dropped (yeh = ye, woh = wo);
  - a final n after a vowel dropped, as nasalisation (nahin = nahi, main = mein = me,
    hain = hai = he).
It merges some distinct words (मैं "main" and में "mein" both become "me"), so it
flatters slightly; character error rate on the folded text is reported alongside.
"""

from __future__ import annotations

import re
import unicodedata

from lucy.voice.translit import to_latin

__all__ = ["fold", "fold_word", "non_latin_share", "to_latin"]

_FOLDS = [
    ("chh", "ch"), ("chch", "ch"), ("cch", "ch"), ("ph", "f"), ("w", "v"), ("z", "j"), ("q", "k"), ("x", "ks"),
    ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("aa", "a"),
    ("ai", "e"), ("ei", "e"), ("au", "o"), ("ou", "o"),
]  # fmt: skip
_DOUBLE = re.compile(r"([a-z])\1+")
_FINAL_NASAL = re.compile(r"([aeiou])n$")


def fold_word(word: str) -> str:
    w = unicodedata.normalize("NFKD", word.lower())
    w = "".join(c for c in w if c.isascii() and c.isalnum())
    for a, b in _FOLDS:
        w = w.replace(a, b)
    w = re.sub(r"ay$", "e", w)
    w = re.sub(r"([aeiou])h$", r"\1", w) if len(w) > 2 else w
    w = _DOUBLE.sub(r"\1", w)
    return _FINAL_NASAL.sub(r"\1", w) if len(w) > 2 else w


def fold(text: str) -> str:
    """The romanised-Hindi spelling fold of `text` (see the module docstring)."""
    words = re.split(r"[^\w]+|_", to_latin(text))
    return " ".join(f for f in (fold_word(w) for w in words) if f)


def non_latin_share(text: str) -> float:
    """Share of the letters and marks in `text` that are not Latin script (0 if none)."""
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    other = sum(not unicodedata.name(c, "").startswith("LATIN") for c in letters)
    return other / len(letters)
