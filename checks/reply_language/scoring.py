"""Scoring for checks/reply_language: script, Latin words in Hindi, a rough language verdict.

Pure functions, no model. Since 2026-11-17 (ADR 0024) Lucy writes Hindi replies in
Devanagari, English loanwords and names included, and English replies in English: a
Hindi reply's script is right when most of its words are Devanagari, an English reply's
when it has none, and `latin_words` counts the Latin words left in a Devanagari reply
(the voice would spell them out). The language heuristic counts Devanagari words and
common romanised Hindi function words as Hindi (`lucy.core.hinglish`, the user's Hindi
reaches Lucy romanised); the thresholds are per reply.
Limits, by design of so simple a measure:

- It says nothing about whether the Hindi is good, natural or correctly spelled; read
  the replies for that.
- Short replies are noisy: "Okay." scores as English, "Theek hai." as Hindi, and a
  one-word "Haan" as Hindi whatever the conversation was.
- English words Hindi speakers use (calendar, meeting, weather) count as English, so
  a mixed reply and a Hindi reply full of such nouns look alike: "mixed" accepts
  anything with some Hindi in it.
- English "me" (Hindi "mein" said short) is left out because it is common in English.
  A few list words are also English words or names ("main", "par", "ho", "jo"); an
  English reply that uses one of them once in a long reply stays under the threshold,
  but a short one may not.
- Spellings outside the list (rare words, unusual spellings) count as English, so
  very literary or oddly spelled Hindi scores lower.
"""

from __future__ import annotations

import re

from lucy.core.hinglish import HINDI_WORDS, hindi_share, words

__all__ = [
    "HINDI_WORDS",
    "devanagari",
    "devanagari_words",
    "hindi_share",
    "language_ok",
    "latin_words",
    "script_ok",
    "verdict",
    "words",
]

DEVANAGARI = re.compile(r"[ऀ-ॿ꣠-ꣿ]")
DEVANAGARI_WORD = re.compile(r"[ऀ-ॣॱ-ॿ]+")  # not the danda or Devanagari digits

HINDI_AT = 0.15  # share of list words from which a reply counts as Hindi
SOME_HINDI_AT = 0.05  # below this a reply counts as English


def devanagari(text: str) -> int:
    """How many Devanagari characters `text` contains."""
    return len(DEVANAGARI.findall(text))


def devanagari_words(text: str) -> int:
    return len(DEVANAGARI_WORD.findall(text))


def latin_words(text: str) -> int:
    """How many Latin-script words `text` contains."""
    return len(words(text))


def script_ok(expected: str, text: str) -> bool:
    """English replies: no Devanagari. Hindi or mixed: most words in Devanagari."""
    if expected == "en":
        return devanagari(text) == 0
    return devanagari_words(text) > latin_words(text)


def verdict(text: str) -> str:
    """'hi', 'mixed' (some Hindi, under the Hindi threshold) or 'en'; '' for no words.
    Devanagari words and romanised Hindi list words both count as Hindi."""
    latin, deva = words(text), devanagari_words(text)
    if not latin and not deva:
        return ""
    share = (deva + sum(w in HINDI_WORDS for w in latin)) / (deva + len(latin))
    if share >= HINDI_AT:
        return "hi"
    return "mixed" if share >= SOME_HINDI_AT else "en"


def language_ok(expected: str, text: str) -> bool:
    """Whether `text` is in the `expected` language ('en', 'hi', 'mixed').
    'mixed' accepts any reply with some Hindi ('hi' or 'mixed'), since a natural mixed
    reply is Hindi sentences with English nouns, which the list cannot tell from Hindi."""
    got = verdict(text)
    if expected == "en":
        return got == "en"
    if expected == "hi":
        return got == "hi"
    return got in ("hi", "mixed")
