"""Scoring of checks/reply_language: Devanagari count and language verdict (no model)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks" / "reply_language"))
from scoring import (  # noqa: E402
    devanagari,
    hindi_share,
    language_ok,
    latin_words,
    script_ok,
    verdict,
)


def test_devanagari_counts_only_devanagari():
    assert devanagari("Kal aapki meeting 10 baje hai.") == 0
    assert devanagari("कल meeting है") == 4


def test_verdicts():
    assert verdict("The capital of Japan is Tokyo.") == "en"
    assert verdict("Kal aapke calendar mein teen events hain.") == "hi"
    assert verdict("Theek hai.") == "hi"
    assert verdict("") == ""
    assert hindi_share("hai hai the") == 2 / 3


def test_language_ok():
    hindi = "Delhi mein abhi mausam saaf hai aur tapman 24 degree hai."
    assert language_ok("hi", hindi)
    assert language_ok("mixed", hindi)
    assert not language_ok("en", hindi)
    english = "It is sunny in Delhi, with a high of 28 degrees today."
    assert language_ok("en", english)
    assert not language_ok("hi", english)
    assert not language_ok("mixed", english)


def test_devanagari_hindi_and_script():
    hindi = "कल आपकी तीन मीटिंग हैं, पहली 10 बजे।"
    assert verdict(hindi) == "hi" and language_ok("hi", hindi) and script_ok("hi", hindi)
    assert not language_ok("en", hindi) and not script_ok("en", hindi)
    leftover = "कल आपकी तीन meeting हैं।"
    assert script_ok("hi", leftover) and latin_words(leftover) == 1
    romanised = "Kal aapki teen meeting hain."
    assert language_ok("hi", romanised) and not script_ok("hi", romanised)
    assert script_ok("en", "It is sunny.")
