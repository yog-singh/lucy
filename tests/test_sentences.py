import random

import pytest

from lucy.core.sentences import SentenceSplitter

TEXT = (
    "Hello there! It is 3.5 degrees outside. Dr. Smith said so, e.g. on the radio. "
    'Really?! "Yes." Mrs. Jones agreed\nNew line here... and more. Last bit'
)
EXPECTED = [
    "Hello there!",
    "It is 3.5 degrees outside.",
    "Dr. Smith said so, e.g. on the radio.",
    "Really?!",
    '"Yes."',
    "Mrs. Jones agreed",
    "New line here...",
    "and more.",
    "Last bit",
]


def split(chunks: list[str]) -> list[str]:
    splitter = SentenceSplitter()
    out = [s for chunk in chunks for s in splitter.feed(chunk)]
    return out + splitter.flush()


def test_whole_text() -> None:
    assert split([TEXT]) == EXPECTED


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13])
def test_fixed_chunkings_give_same_sentences(size: int) -> None:
    chunks = [TEXT[i : i + size] for i in range(0, len(TEXT), size)]
    assert split(chunks) == EXPECTED


@pytest.mark.parametrize("seed", range(20))
def test_random_chunkings_give_same_sentences(seed: int) -> None:
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, len(TEXT)), rng.randint(1, 40)))
    chunks = [TEXT[a:b] for a, b in zip([0, *cuts], [*cuts, len(TEXT)], strict=True)]
    assert split(chunks) == EXPECTED


def test_decimal_is_not_a_boundary() -> None:
    assert split(["Pi is 3.14159 roughly."]) == ["Pi is 3.14159 roughly."]


@pytest.mark.parametrize("abbr", ["Dr.", "Mr.", "Mrs.", "e.g.", "i.e.", "etc."])
def test_abbreviations_do_not_end_sentences(abbr: str) -> None:
    assert split([f"See {abbr} it works. Done"]) == [f"See {abbr} it works.", "Done"]


def test_newlines_split_and_blank_lines_vanish() -> None:
    assert split(["one\n\n  \ntwo\n"]) == ["one", "two"]


def test_terminator_waits_for_next_character() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Done.") == []
    assert splitter.feed(" Next") == ["Done."]
    assert splitter.flush() == ["Next"]
    assert splitter.flush() == []


def test_flush_of_whitespace_is_empty() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("   ") == []
    assert splitter.flush() == []
