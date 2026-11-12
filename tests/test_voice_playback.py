"""SpeechPlayer: ordering, look-ahead synthesis, stopping and the spoken-text estimate."""

import asyncio

import pytest

from lucy.core.types import OUTPUT_SAMPLE_RATE
from lucy.voice.audio import FileAudioIO
from lucy.voice.playback import SpeechPlayer, cut_at_words, speakable
from tests.voice_fakes import FakeTTS

TIMEOUT = 5


async def until(condition, timeout: float = TIMEOUT) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0)


def played_texts(io: FileAudioIO) -> int:
    return io.played_samples


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**Sure!** Here it is.", "Sure! Here it is."),
        ("- first item", "first item"),
        ("2. second item", "second item"),
        ("## Heading", "Heading"),
        ("See [the docs](https://example.com/x) now.", "See the docs now."),
        ("Use `ls -la` here", "Use ls -la here"),
        ("Great job 🎉👍🏽!", "Great job !"),
        ("I ❤️ tea", "I tea"),
        ("> quoted line", "quoted line"),
        ("| a | b |", "a b"),
        ("|---|---|", ""),
        ("🎉🎉", ""),
        ("***", ""),
        ("It costs 3.5 € or £4.", "It costs 3.5 € or £4."),
        ("snake_case stays", "snake_case stays"),
    ],
)
def test_speakable(text: str, expected: str) -> None:
    assert speakable(text) == expected


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.0, ""),
        (0.1, ""),
        (0.3, "The"),
        (0.4, "The quick"),
        (0.5, "The quick"),
        (0.62, "The quick brown"),
        (0.99, "The quick brown fox"),
        (1.0, "The quick brown fox jumps."),
    ],
)
def test_cut_at_words(fraction: float, expected: str) -> None:
    assert cut_at_words("The quick brown fox jumps.", fraction) == expected


async def test_speaks_all_sentences_in_order() -> None:
    io = FileAudioIO([])
    tts = FakeTTS()
    firsts: list[str] = []
    player = SpeechPlayer(
        tts,
        io,
        on_first_audio=lambda: firsts.append("audio"),
        on_first_played=lambda: firsts.append("played"),
    )
    sentences = ["One two three.", "Four five.", "Six seven eight nine."]
    for s in sentences:
        player.add(s)
    async with asyncio.timeout(TIMEOUT):
        await player.wait()
    assert tts.started == sentences
    assert sorted(tts.finished) == sorted(sentences)
    expected = sum(tts.samples_for(s) for s in sentences)
    assert io.played_samples == expected
    assert len(io.plays) == 1 and not io.plays[0].stopped
    assert firsts == ["audio", "played"]
    assert player.spoken_text() == " ".join(sentences)
    assert player.sentences_played == 3


async def test_never_plays_out_of_order_when_a_later_sentence_is_ready_first() -> None:
    io = FileAudioIO([])
    hold_first = asyncio.Event()
    tts = FakeTTS(gate=lambda text: hold_first if text.startswith("Slow") else None)
    player = SpeechPlayer(tts, io, lookahead=1)
    player.add("Slow first sentence.")
    player.add("Fast second.")
    await until(lambda: "Fast second." in tts.finished)
    assert io.played_samples == 0  # the second is ready but waits for the first
    hold_first.set()
    async with asyncio.timeout(TIMEOUT):
        await player.wait()
    first = tts.samples_for("Slow first sentence.")
    # The output starts with the first sentence's audio: its length precedes the second's.
    assert io.played_samples == first + tts.samples_for("Fast second.")
    assert player.spoken_text() == "Slow first sentence. Fast second."


async def test_next_sentence_is_synthesised_while_the_current_one_plays() -> None:
    io = FileAudioIO([])
    tts = FakeTTS()
    player = SpeechPlayer(tts, io, lookahead=1)
    for s in ["Alpha sentence here.", "Beta sentence here.", "Gamma sentence here."]:
        player.add(s)
    await until(lambda: io.played_samples > 0)
    # While the first plays, the second has started, the third not yet.
    assert tts.started[:2] == ["Alpha sentence here.", "Beta sentence here."]
    assert "Gamma sentence here." not in tts.started
    async with asyncio.timeout(TIMEOUT):
        await player.wait()
    assert tts.max_active <= 2


async def test_sentences_added_while_playing_follow_without_a_new_play() -> None:
    io = FileAudioIO([])
    tts = FakeTTS()
    player = SpeechPlayer(tts, io)
    player.add("First one.")
    await until(lambda: io.played_samples > 0)
    player.add("Second one.")
    async with asyncio.timeout(TIMEOUT):
        await player.wait()
    assert len(io.plays) == 1
    assert player.spoken_text() == "First one. Second one."


async def test_stop_mid_sentence_returns_what_was_heard() -> None:
    io = FileAudioIO([])
    tts = FakeTTS(seconds_per_char=0.05)
    player = SpeechPlayer(tts, io)
    first, second = "Hello there.", "The quick brown fox jumps over the dog."
    player.add(first)
    player.add(second)
    player.add("Never heard at all.")
    first_len = tts.samples_for(first)
    second_len = tts.samples_for(second)
    # Stop half way through the second sentence.
    await until(lambda: io.played_samples >= first_len + second_len // 2)
    spoken = player.stop()
    assert spoken.startswith(first + " The quick brown")
    assert not spoken.endswith("dog.")
    heard = io.played_samples
    await player.aclose()
    assert io.plays[0].stopped
    assert io.played_samples == heard  # nothing more played after stop
    assert player.stop() == spoken  # idempotent
    assert player.spoken_text(start=1) == spoken[len(first) + 1 :]


async def test_stop_before_any_audio_is_empty_and_closes_synthesis() -> None:
    io = FileAudioIO([])
    hold = asyncio.Event()
    tts = FakeTTS(gate=lambda text: hold)
    player = SpeechPlayer(tts, io)
    player.add("Waiting sentence.")
    await until(lambda: tts.active == 1)
    assert player.stop() == ""
    await player.aclose()
    assert tts.active == 0
    assert io.played_samples == 0


async def test_stop_closes_an_in_flight_tts_stream() -> None:
    io = FileAudioIO([])
    tts = FakeTTS(seconds_per_char=0.05, chunk_seconds=0.05)
    player = SpeechPlayer(tts, io, lookahead=0)
    player.add("A fairly long sentence that takes a while to synthesise and play.")
    await until(lambda: io.played_samples > 0)
    player.stop()
    await player.aclose()
    assert tts.active == 0


async def test_a_failed_sentence_is_skipped_and_reported() -> None:
    io = FileAudioIO([])
    tts = FakeTTS(fail_on="broken", fail_after=0)
    errors: list[str] = []
    player = SpeechPlayer(tts, io, on_error=lambda text, exc: errors.append(text))
    player.add("Good one.")
    player.add("This is broken.")
    player.add("Good again.")
    async with asyncio.timeout(TIMEOUT):
        await player.wait()
    assert errors == ["This is broken."]
    assert io.played_samples == tts.samples_for("Good one.") + tts.samples_for("Good again.")
    assert player.spoken_text() == "Good one. Good again."


async def test_a_sentence_failing_mid_stream_keeps_its_played_part() -> None:
    io = FileAudioIO([])
    tts = FakeTTS(fail_on="broken", fail_after=2, chunk_seconds=0.1)
    player = SpeechPlayer(tts, io)
    player.add("This sentence is broken somewhere in the middle of it.")
    player.add("Next.")
    async with asyncio.timeout(TIMEOUT):
        await player.wait()
    assert io.played_samples == int(0.2 * OUTPUT_SAMPLE_RATE) + tts.samples_for("Next.")


async def test_unspeakable_sentences_are_skipped_without_synthesis() -> None:
    io = FileAudioIO([])
    tts = FakeTTS()
    player = SpeechPlayer(tts, io)
    player.add("🎉🎉")
    player.add("**Done.**")
    async with asyncio.timeout(TIMEOUT):
        await player.wait()
    assert tts.started == ["Done."]
    assert player.spoken_text() == "**Done.**"  # the text as shown, not as cleaned
    assert player.sentences_played == 2


async def test_wait_without_sentences_returns_at_once() -> None:
    player = SpeechPlayer(FakeTTS(), FileAudioIO([]))
    async with asyncio.timeout(TIMEOUT):
        await player.wait()
    assert player.spoken_text() == ""


async def test_adding_after_stop_is_ignored() -> None:
    io = FileAudioIO([])
    tts = FakeTTS()
    player = SpeechPlayer(tts, io)
    player.stop()
    player.add("Too late.")
    await player.aclose()
    assert tts.started == []
