"""Devanagari to Latin script, deterministic and pure Python (no dependency).

Used by the `indic_flex` speech adapter (ADR 0023), whose output must be Latin script
only: the model's "mixed" mode writes English words in Latin but Hindi in Devanagari.

`to_latin(text)` writes every Devanagari word in plain ASCII in the informal style
Hindi speakers type ("main ghar ja raha hun", "nahin", "theek hai"): no diacritics,
aa/ee/oo for long vowels inside a word and a/i/u at the end, the inherent vowel
dropped at the end of a word and between a vowel and a consonant that carries a vowel
(schwa deletion, right to left: समझना samajhna, कमरा kamra), nasal marks as n
(में mein, हैं hain), nukta letters as their Urdu sounds (ज़ z, फ़ f, क़ q). Digits become
ASCII, the danda a full stop; everything that is not Devanagari passes unchanged.
English loanwords written in Devanagari (`lucy.core.loanwords`: कैलेंडर, फ्राइडे, टाइम)
are shown in their English spelling (calendar, Friday, time) instead of "kaileindar".
The spelling is less idiomatic than a person's ("ke lie", "chaahie"); see
checks/README.md (stt_accuracy, Latin-only flex) for how it was measured.
"""

from __future__ import annotations

import re
import unicodedata

from lucy.core.loanwords import TO_ENGLISH

_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e", "ॠ": "ri",
}  # fmt: skip
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e",
    "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ॄ": "ri",
}  # fmt: skip
_FINAL = {"aa": "a", "ee": "i", "oo": "u"}  # a long vowel at the end of a word
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
    "ळ": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}  # fmt: skip
_NUKTA_FORMS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
_NUKTA, _VIRAMA, _ANUSVARA, _CHANDRABINDU, _VISARGA = "़", "्", "ं", "ँ", "ः"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]+")
_ENGLISH = {unicodedata.normalize("NFC", k): v for k, v in TO_ENGLISH.items()}
"""Loanwords in Devanagari (NFC, as `to_latin` sees them) to their English spelling."""
_TRAILING_DANDAS = re.compile(r"[।॥]+$")


def _word(word: str) -> str:
    """One run of Devanagari characters to Latin."""
    # Units: [kind, latin, vowel, nasal]; kind "c" consonant, "v" vowel, "x" other.
    units: list[list] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in _CONSONANTS:
            latin = _CONSONANTS[ch]
            if i + 1 < len(word) and word[i + 1] == _NUKTA:
                latin = _NUKTA_FORMS.get(ch, latin)
                i += 1
            units.append(["c", latin, "a", ""])  # inherent vowel until a sign says otherwise
        elif ch in _MATRAS and units and units[-1][0] == "c":
            units[-1][2] = _MATRAS[ch]
        elif ch == _VIRAMA and units and units[-1][0] == "c":
            units[-1][2] = ""
        elif ch in (_ANUSVARA, _CHANDRABINDU):
            if units:
                units[-1][3] = "n"
            else:
                units.append(["x", "n", "", ""])
        elif ch == _VISARGA:
            units.append(["x", "h", "", ""])
        elif ch in _VOWELS:
            units.append(["v", "", _VOWELS[ch], ""])
        elif ch == "ॐ":
            units.append(["x", "om", "", ""])
        elif unicodedata.category(ch) == "Nd":
            units.append(["x", str(unicodedata.digit(ch)), "", ""])
        elif ch in "।॥":
            units.append(["x", ".", "", ""])
        elif ch in (_NUKTA, "ऽ") or unicodedata.category(ch) in ("Mn", "Mc"):
            pass  # a stray sign
        else:
            units.append(["x", ch, "", ""])
        i += 1

    def voiced(k: int) -> bool:
        return 0 <= k < len(units) and units[k][0] in "cv" and units[k][2] != ""

    cons = [k for k, u in enumerate(units) if u[0] == "c"]
    # schwa deletion, right to left
    if len(cons) > 1 or (cons and len(units) > 1):
        last = cons[-1] if cons else -1
        if last == len(units) - 1 and units[last][2] == "a" and not units[last][3]:
            units[last][2] = ""
        for k in reversed(range(1, len(units) - 1)):
            u = units[k]
            if u[0] == "c" and u[2] == "a" and not u[3]:
                nxt = units[k + 1]
                if voiced(k - 1) and nxt[0] == "c" and nxt[2] != "":
                    u[2] = ""
    out = []
    for k, (_kind, latin, vowel, nasal) in enumerate(units):
        if vowel in _FINAL and k == len(units) - 1:
            vowel = _FINAL[vowel]
        if nasal and vowel == "e":
            vowel = "ei"  # में mein
        out.append(latin + vowel + nasal)
    return "".join(out)


def to_latin(text: str) -> str:
    """`text` with every Devanagari run written in Latin (see the module docstring)."""
    return _DEVANAGARI.sub(lambda m: _run(m.group()), unicodedata.normalize("NFC", text))


def _run(run: str) -> str:
    """One Devanagari run: a known loanword in English, anything else through `_word`."""
    dandas = _TRAILING_DANDAS.search(run)
    core = run[: dandas.start()] if dandas else run
    if core in _ENGLISH:
        return _ENGLISH[core] + ("." * len(dandas.group()) if dandas else "")
    return _word(run)


class StreamingLatin:
    """`to_latin` for text arriving in pieces (a streamed reply shown on screen): a
    Devanagari word at the end of a piece is held back until the next piece shows where
    it ends, so a word split across pieces is still written as one. `flush` at the end."""

    def __init__(self) -> None:
        self._held = ""

    def feed(self, text: str) -> str:
        text = self._held + text
        cut = len(text)
        while cut and _is_devanagari(text[cut - 1]):
            cut -= 1
        self._held = text[cut:]
        return to_latin(text[:cut])

    def flush(self) -> str:
        text, self._held = self._held, ""
        return to_latin(text)


def _is_devanagari(char: str) -> bool:
    return "ऀ" <= char <= "ॿ" and char not in "।॥"
