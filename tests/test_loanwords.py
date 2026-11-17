"""The loanword table, used both for speech (English to Devanagari) and on screen."""

from __future__ import annotations

import pytest

from lucy.core.devanagari import EXCEPTIONS, FUNCTION_WORDS, respell_word
from lucy.core.loanwords import CANONICAL, CAPITALISED, LOANWORDS, TO_ENGLISH
from lucy.voice.translit import StreamingLatin, to_latin


@pytest.mark.parametrize(
    ("devanagari", "shown"),
    [("कैलेंडर", "calendar"), ("फ्राइडे", "Friday"), ("टाइम", "time"), ("ओके", "ok"),
     ("मीटिंग्स", "meetings"), ("लूसी", "Lucy"), ("दिसंबर", "December")],
)  # fmt: skip
def test_loanwords_are_shown_in_english(devanagari, shown) -> None:
    assert to_latin(devanagari) == shown


def test_hindi_words_are_transliterated_as_before() -> None:
    assert to_latin("नहीं") == "nahin"
    assert to_latin("बस, थोड़ा रुको") == "bas, thora ruko"  # बस is Hindi here, not "bus"


def test_a_sentence_mixes_both() -> None:
    assert to_latin("कल की मीटिंग का टाइम क्या है?") == "kal ki meeting ka time kya hai?"
    assert to_latin("आपके कैलेंडर में फ्राइडे को डेंटिस्ट अपॉइंटमेंट है।") == (
        "aapke calendar mein Friday ko dentist appointment hai."
    )


def test_streaming_shows_a_loanword_split_across_pieces() -> None:
    s = StreamingLatin()
    assert s.feed("कल कैलें") + s.feed("डर देखो") + s.flush() == "kal calendar dekho"


def test_every_entry_round_trips() -> None:
    for english, devanagari in LOANWORDS.items():
        assert respell_word(english) == devanagari
        shown = TO_ENGLISH[devanagari]
        assert shown.lower() == CANONICAL.get(devanagari, english)
        assert to_latin(devanagari) == shown
        assert respell_word(shown) == devanagari


def test_one_table_in_both_directions() -> None:
    assert EXCEPTIONS == {**LOANWORDS, **FUNCTION_WORDS}
    assert not set(FUNCTION_WORDS.values()) & set(TO_ENGLISH)  # "the" is never shown back
    assert CAPITALISED <= set(LOANWORDS)
    shared = {d for d in LOANWORDS.values() if list(LOANWORDS.values()).count(d) > 1}
    assert shared <= set(CANONICAL)
