"""The endpointer, fed synthetic probability sequences: no audio, no model."""

import pytest

from lucy.voice.endpoint import Discarded, Endpointer, SpeechStarted, UtteranceComplete

FRAME = 0.032  # seconds per frame


def run(endpointer: Endpointer, probabilities: list[float], *, flush: bool = False) -> list:
    """Feed one frame per probability; frame i's audio is bytes([i % 256]) * 2, ending at
    (i + 1) * FRAME. Returns (frame index, event) pairs."""
    events = []
    for i, p in enumerate(probabilities):
        for event in endpointer.feed(bytes([i % 256]) * 2, p, (i + 1) * FRAME):
            events.append((i, event))
    if flush:
        for event in endpointer.flush(len(probabilities) * FRAME):
            events.append((len(probabilities), event))
    return events


def frame_ids(utterance: UtteranceComplete) -> list[int]:
    return list(utterance.audio.data[::2])


def speech(n: int, p: float = 0.9) -> list[float]:
    return [p] * n


def quiet(n: int, p: float = 0.05) -> list[float]:
    return [p] * n


def test_frame_counts_follow_the_milliseconds() -> None:
    e = Endpointer(min_speech_ms=160, endpoint_silence_ms=600, preroll_ms=300, postroll_ms=200)
    assert e.min_speech_frames == 5
    assert e.silence_frames == 19  # 19 x 32 ms = 608 ms
    assert e.preroll_frames == 10
    assert e.postroll_frames == 7


def test_a_short_click_does_not_start_speech() -> None:
    e = Endpointer(min_speech_ms=160)
    assert run(e, quiet(5) + speech(4) + quiet(40)) == []
    assert not e.in_speech


def test_speech_start_end_and_preroll() -> None:
    e = Endpointer(min_speech_ms=160, endpoint_silence_ms=600, preroll_ms=300, postroll_ms=200)
    probs = quiet(20) + speech(30) + quiet(25)
    events = run(e, probs)
    (i_start, started), (i_end, done) = events
    assert isinstance(started, SpeechStarted) and isinstance(done, UtteranceComplete)
    assert i_start == 24  # the 5th speech frame (frames 20..24)
    assert started.time == pytest.approx(20 * FRAME)
    assert done.speech_start == pytest.approx(20 * FRAME)
    assert done.speech_end == pytest.approx(50 * FRAME)  # end of frame 49
    assert i_end == 49 + 19  # 19 silent frames after the last speech frame
    assert done.endpoint_time == pytest.approx((i_end + 1) * FRAME)
    ids = frame_ids(done)
    assert ids[0] == 10  # 10 frames of pre-roll before frame 20
    assert ids[-1] == 49 + 7  # post-roll of 7 frames after the last speech frame
    assert ids == list(range(10, 57))
    assert not done.forced
    assert done.duration == pytest.approx(len(ids) / 16000)


def test_preroll_is_limited_by_what_was_heard() -> None:
    e = Endpointer(preroll_ms=300)
    events = run(e, quiet(2) + speech(20) + quiet(25))
    done = events[-1][1]
    assert frame_ids(done)[0] == 0


def test_a_short_pause_does_not_end_the_utterance() -> None:
    e = Endpointer(endpoint_silence_ms=600)
    events = run(e, speech(10) + quiet(15) + speech(10) + quiet(25))
    completes = [ev for _, ev in events if isinstance(ev, UtteranceComplete)]
    assert len(completes) == 1
    assert completes[0].speech_end == pytest.approx(35 * FRAME)


def test_hysteresis_keeps_soft_frames_as_speech() -> None:
    e = Endpointer(threshold=0.5, endpoint_silence_ms=200)
    # 0.4 is below the start threshold but above 0.5 - 0.15: it continues speech.
    events = run(e, speech(10) + [0.4] * 20 + quiet(10))
    done = [ev for _, ev in events if isinstance(ev, UtteranceComplete)]
    assert len(done) == 1
    assert done[0].speech_end == pytest.approx(30 * FRAME)


def test_soft_frames_do_not_start_speech() -> None:
    e = Endpointer(threshold=0.5)
    assert run(e, [0.45] * 50) == []


def test_too_short_an_utterance_is_discarded() -> None:
    e = Endpointer(min_speech_ms=96, min_utterance_ms=250)
    events = run(e, quiet(5) + speech(4) + quiet(25))  # 128 ms of speech
    kinds = [type(ev) for _, ev in events]
    assert kinds == [SpeechStarted, Discarded]
    assert "too short" in events[-1][1].reason
    assert not e.in_speech


def test_max_utterance_forces_an_endpoint_and_restarts() -> None:
    e = Endpointer(max_utterance_s=1.0, min_speech_ms=160)  # 32 frames
    events = run(e, speech(80) + quiet(25))
    completes = [ev for _, ev in events if isinstance(ev, UtteranceComplete)]
    assert completes[0].forced
    assert completes[0].speech_end - completes[0].speech_start == pytest.approx(32 * FRAME)
    # Speech goes on, so a new utterance starts after another minimum run.
    assert len(completes) >= 2
    assert not completes[-1].forced


def test_flush_completes_speech_in_progress() -> None:
    e = Endpointer()
    events = run(e, quiet(3) + speech(20), flush=True)
    assert isinstance(events[-1][1], UtteranceComplete)
    assert events[-1][1].speech_end == pytest.approx(23 * FRAME)


def test_flush_without_speech_returns_nothing() -> None:
    e = Endpointer()
    assert run(e, quiet(10) + speech(2), flush=True) == []


def test_reset_forgets_speech_in_progress() -> None:
    e = Endpointer()
    run(e, speech(20))
    assert e.in_speech
    e.reset()
    assert not e.in_speech
    assert run(e, quiet(30)) == []


def test_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        Endpointer(threshold=1.2)
