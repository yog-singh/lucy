"""Pure logic of checks/tts_listen: timing summary, failure flags and the listening page."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks" / "tts_listen"))
from report import compare_loaded, flags, render_page, repeated_ngram, rtf, summarise  # noqa: E402


def _row(column, id_, first, total, duration, phase="idle", **extra):
    return {
        "column": column,
        "id": id_,
        "phase": phase,
        "first_s": first,
        "total_s": total,
        "duration_s": duration,
        "rtf": rtf(total, duration),
        "wav": f"{column}/{id_}.wav",
        **extra,
    }


def test_rtf_handles_empty_audio():
    assert rtf(1.0, 2.0) == 0.5
    assert rtf(1.0, 0.0) is None


def test_summarise_takes_medians_per_column_and_counts_errors():
    rows = [
        _row("a", "s1", 0.1, 1.0, 2.0),
        _row("a", "s2", 0.3, 3.0, 2.0),
        _row("a", "s3", 0.2, 2.0, 4.0),
        {"column": "a", "id": "s4", "phase": "idle", "error": "boom"},
        _row("a", "s1", 9.0, 9.0, 1.0, phase="loaded"),
    ]
    s = summarise(rows)["a"]
    assert s["n"] == 3 and s["errors"] == 1
    assert s["first_median_s"] == pytest.approx(0.2)
    assert s["first_max_s"] == pytest.approx(0.3)
    assert s["rtf_median"] == pytest.approx(0.5)


def test_compare_loaded_pairs_the_same_sentences():
    rows = [
        _row("a", "s1", 0.1, 1.0, 2.0),
        _row("a", "s2", 0.1, 5.0, 2.0),  # not timed loaded: left out
        _row("a", "s1", 0.2, 2.0, 2.0, phase="loaded"),
    ]
    c = compare_loaded(rows)["a"]
    assert c["n"] == 1
    assert c["total_s_idle"] == 1.0 and c["total_s_loaded"] == 2.0
    assert c["total_ratio"] == pytest.approx(2.0)


def test_flags_catch_drops_stutters_and_garbage_only():
    ref = "it should stay dry today".split()
    assert flags(ref, ref, 0.0, None) == []
    assert flags(ref, ["it"], 0.8, None) == ["dropped", "garbled"]
    assert "repeated" in flags(ref, "it should should stay stay dry today".split(), 0.4, None)
    assert repeated_ngram("a b a b c".split()) == "a b"
    # a high word error with a low character error (spelling variants) is not garbled
    assert flags(ref, ref, 0.6, 0.1) == []


def test_page_is_self_contained_with_relative_audio_and_escaped_text():
    sentences = [
        {"id": "s1", "lang": "hi", "kind": "hindi", "text": "ठीक है <b>"},
        {"id": "s2", "lang": "en", "kind": "english", "text": "Sure."},
    ]
    columns = [{"column": "a", "title": "Model A", "facts": {"Licence": "MIT"}}]
    rows = [
        _row("a", "s1", 0.25, 1.0, 2.0, asr="theek hai", flags=["dropped"]),
        {"column": "a", "id": "s2", "phase": "idle", "error": "no audio"},
    ]
    page = render_page("Test", sentences, columns, rows, ["a note"])
    assert page.startswith("<!doctype html>")
    assert '<audio controls preload="none" src="a/s1.wav">' in page
    assert "ठीक है &lt;b&gt;" in page
    assert "first 0.25 s" in page and "0.25 s</td>" in page  # cell and header median
    assert "heard: theek hai" in page and "dropped" in page and "no audio" in page
    assert "http" not in page and "<script" not in page and "<link" not in page


def test_playback_simulation_counts_waits_between_sentences():
    from report import simulate_playback

    rows = [
        {"ready_s": 1.0, "duration_s": 2.0},  # plays 1 to 3
        {"ready_s": 2.5, "duration_s": 1.0},  # ready in time: plays 3 to 4
        {"ready_s": 5.0, "duration_s": 1.0},  # 1 s late
        {"error": "boom"},
        {"ready_s": 5.5, "duration_s": 1.0},  # plays 6 to 7
    ]
    sim = simulate_playback(rows)
    assert sim["underruns"] == 1
    assert sim["underrun_s"] == pytest.approx(1.0)
    assert simulate_playback([]) == {"underruns": 0, "underrun_s": 0}
