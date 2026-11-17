"""The romanised-Hindi to Devanagari fallback (`lucy.core.devanagari`, ADR 0024)."""

from __future__ import annotations

import re

import pytest

from lucy.core.devanagari import ROMAN_HINDI, roman_word, romanised_to_devanagari

DEVANAGARI_ONLY = re.compile(r"^[ऀ-ॿ]+$")


@pytest.mark.parametrize(
    ("roman", "deva"),
    [
        ("hai", "है"), ("hain", "हैं"), ("mein", "में"), ("nahi", "नहीं"), ("nahin", "नहीं"),
        ("kya", "क्या"), ("aap", "आप"), ("kal", "कल"), ("aaj", "आज"), ("subah", "सुबह"),
        ("baje", "बजे"), ("theek", "ठीक"), ("thik", "ठीक"), ("haan", "हाँ"), ("accha", "अच्छा"),
        ("acha", "अच्छा"), ("bohot", "बहुत"), ("main", "मैं"), ("to", "तो"), ("the", "थे"),
    ],
)  # fmt: skip
def test_table_words(roman: str, deva: str) -> None:
    assert roman_word(roman) == deva
    assert roman_word(roman.capitalize()) == deva


def test_table_is_devanagari() -> None:
    assert len(ROMAN_HINDI) > 400
    for roman, deva in ROMAN_HINDI.items():
        assert roman.isascii() and roman.islower(), roman
        assert DEVANAGARI_ONLY.match(deva), (roman, deva)


def test_loanwords_use_the_english_spelling_list() -> None:
    assert roman_word("meeting") == "मीटिंग"
    assert roman_word("calendar") == "कैलेंडर"
    assert roman_word("Friday") == "फ्राइडे"


@pytest.mark.parametrize(
    ("roman", "deva"),
    [
        ("kabhi", "कभी"),  # in the table
        ("bachpan", "बच्पन"),  # rules: a halant between consonants (sounds the same)
        ("khelna", "खेल्ना"),  # kh, e, final a long
        ("chhutiyaan", "छुतियां"),  # chh, final n after a long vowel as anusvara
        ("ghoomti", "घूम्ती"),  # gh, oo, final i long
        ("dhanda", "धंदा"),  # dh, n before a consonant as anusvara
        ("bhaiya", "भैया"),  # bh, ai
        ("phool", "फूल"),  # ph
        ("shaadi", "शादी"),  # sh, aa
    ],
)
def test_rules(roman: str, deva: str) -> None:
    assert roman_word(roman) == deva


def test_english_looking_words_take_the_english_rules() -> None:
    assert roman_word("exercise").startswith("एक्स")  # x, c: English
    assert roman_word("evaporation").endswith("न")


def test_sentence() -> None:
    assert (
        romanised_to_devanagari("Kal aapke calendar mein teen events hain.")
        == "कल आपके कैलेंडर में तीन इवेंट्स हैं."
    )
    assert romanised_to_devanagari("Main thik hoon, shukriya!") == "मैं ठीक हूँ, शुक्रिया!"
    # Digits, times and punctuation stay as they are.
    assert romanised_to_devanagari("Subah 7:30 baje.") == "सुबह 7:30 बजे."
    assert romanised_to_devanagari("") == ""
