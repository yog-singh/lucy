"""Scoring, normalisation and enrolment matching of checks/stt_accuracy (no model)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks" / "stt_accuracy"))
from stt_scoring import (  # noqa: E402
    Truth,
    append_truth,
    corpus_wer,
    exact,
    indic_normalise,
    indic_score,
    length_bucket,
    list_clips,
    load_truth,
    match_enrolment,
    native_lattice,
    normal_text,
    percentile,
    side_by_side,
    text_lattice,
    wer,
)

from lucy.voice.enrol import SENTENCES  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, Lucy!", "hello lucy"),
        ("What’s the Nifty at?", "what's the nifty at"),
        ("Lucy's", "lucy's"),
        ("twenty five", "25"),
        ("twenty-five", "25"),
        ("two hundred", "200"),
        ("three thousand four hundred and five", "3400 and 5"),
        ("seven thirty", "7 30"),
        ("7:30", "7 30"),
        ("one two three", "1 2 3"),
        ("the fifth of May", "the 5th of may"),
        ("twenty first", "21st"),
        ("1,500 rupees", "1500 rupees"),
        ("50%", "50 percent"),
        ("fifty percent", "50 percent"),
        ("Nifty fifty", "nifty 50"),
        ("Can you enroll my voice?", "can you enrol my voice"),
        ("my neighbor", "my neighbour"),
        ("oh no", "oh no"),
        ("", ""),
    ],
)
def test_normalise(text, expected):
    assert normal_text(text) == expected


def test_wer_ignores_case_punctuation_and_number_form():
    assert wer("Set a timer for twenty five minutes.", "set a timer for 25 minutes") == 0
    assert exact("It's 7:30.", "it's seven thirty")


def test_wer_counts_word_edits():
    assert wer("call yogesh now", "call yoga now") == pytest.approx(1 / 3)
    assert wer("a b", "") == 1.0
    assert wer("", "anything") == 1.0  # one truth word at least, so never divides by zero


def test_corpus_wer_weights_by_truth_words():
    pairs = [("one two three four", "one two three four"), ("hello", "yellow")]
    assert corpus_wer(pairs) == pytest.approx(1 / 5)  # not the mean of 0 and 1
    assert corpus_wer([("a b c d", "a b c x"), ("e", "e")]) == pytest.approx(1 / 5)


def test_length_bucket_and_percentile():
    assert [length_bucket(s) for s in (1.9, 2.0, 5.0, 5.1)] == ["<2s", "2-5s", "2-5s", ">5s"]
    assert percentile([0.1, 0.2, 0.3, 0.4], 50) == 0.2
    assert percentile(list(range(1, 21)), 95) == 19
    assert percentile([1.0], 95) == 1.0


def test_truth_file_round_trip_last_line_wins(tmp_path):
    f = tmp_path / "truth.jsonl"
    append_truth(f, Truth("clips/a.wav", "hello", heard="yellow"))
    append_truth(f, Truth("clips/b.wav", None, skipped=True, reason="TV"))
    append_truth(f, Truth("clips/a.wav", "hello there"))
    got = load_truth(f)
    assert got["clips/a.wav"].truth == "hello there"
    assert got["clips/b.wav"].skipped and got["clips/b.wav"].reason == "TV"
    assert load_truth(tmp_path / "missing.jsonl") == {}


def test_list_clips_joins_heard_by_message_id(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "20261115T1_aaa.wav").write_bytes(b"")
    (tmp_path / "clips" / "20261115T2_bbb.wav").write_bytes(b"")
    (tmp_path / "enrol_1.wav").write_bytes(b"")
    (tmp_path / "heard.jsonl").write_text('{"id": "aaa", "heard": "hi Lucy"}\n')
    clips = list_clips(tmp_path)
    assert [(c.clip, c.heard) for c in clips] == [
        ("clips/20261115T1_aaa.wav", "hi Lucy"),
        ("clips/20261115T2_bbb.wav", None),
        ("enrol_1.wav", None),
    ]


def test_enrolment_matches_closest_sentence_despite_small_errors():
    heard = {
        "enrol_1.wav": "My neighbor bakes fresh bread every Thursday and shares it with the street",
        "enrol_2.wav": "the quick brown fox jumps over the lazy dog near the quiet riverbank",
    }
    got = {m.clip: m for m in match_enrolment(heard, SENTENCES)}
    assert got["enrol_1.wav"].sentence == SENTENCES[4] and got["enrol_1.wav"].score == 0
    assert got["enrol_2.wav"].sentence == SENTENCES[0]
    assert got["enrol_2.wav"].score == pytest.approx(2 / 14)
    assert got["enrol_2.wav"].margin > 0.5


def test_enrolment_leaves_unclear_clips_unlabelled():
    pool = ["the cat sat on the mat", "the cat sat on the hat", "a b c d e f"]
    got = {m.clip: m for m in match_enrolment({"x": "the cat sat on the", "y": "zzz"}, pool)}
    assert got["x"].sentence is None and "margin" in got["x"].why
    assert got["y"].sentence is None and "best WER" in got["y"].why


def test_enrolment_refuses_a_sentence_matched_twice():
    heard = {"a": SENTENCES[1], "b": SENTENCES[1]}
    got = match_enrolment(heard, SENTENCES)
    assert all(m.sentence is None and m.why == "sentence matched twice" for m in got)


def test_side_by_side_marks_clips_where_configurations_disagree():
    rows = [
        {"clip": "clips/a.wav", "audio_s": 1.24, "truth": None, "heard": "It's yoga."},
        {"clip": "clips/b.wav", "audio_s": 3.0, "truth": "No, thank you.", "heard": None},
    ]
    outputs = {
        "parakeet": {"clips/a.wav": "It's yoga.", "clips/b.wav": "No, thank you."},
        "indic_flex": {"clips/a.wav": "It's Yogesh.", "clips/b.wav": "no thank you"},
    }
    text = side_by_side(rows, outputs)
    assert text.startswith("2 clips; configurations differ (after normalisation) on 1\n")
    assert "clips/a.wav  (1.2 s)  DIFFERS\n" in text
    assert "clips/b.wav  (3.0 s)\n" in text  # same after normalisation
    assert "  heard       It's yoga.\n" in text
    assert "  indic_flex  It's Yogesh.\n" in text
    assert "  truth       No, thank you.\n" in text


def test_worker_configurations_build_prompt_from_words():
    from recognisers import CONFIGS, MLX_WORKER, prompt_sentence, read_words

    assert prompt_sentence(["Lucy", "Hey Jarvis", "Burn"]) == (
        "Names and words that come up: Lucy, Hey Jarvis and Burn."
    )
    assert {"Jarvis", "Hey Jarvis", "Daniela Amodei", "Burn"} <= set(read_words())
    rec = CONFIGS["qwen3_asr_context"]()  # builds the command only; starts nothing
    assert rec._script == MLX_WORKER
    assert rec._args == ["--model=qwen3", f"--prompt={', '.join(read_words())}"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("हाँ, ठीक है।", ["हां", "ठीक", "है"]),  # chandrabindu as anusvara, danda dropped
        ("ज़रूर", ["जरूर"]),  # precomposed nukta letter
        ("ज\u093cरूर", ["जरूर"]),  # nukta as a combining sign
        ("१५ मिनट", ["15", "मिनट"]),  # Devanagari digits
        ("Bus स्टॉप?", ["bus", "स्टॉप"]),  # Latin lower-cased, kept
        ("क\u200dष", ["कष"]),  # zero-width joiner dropped, not a word break
    ],
)
def test_indic_normalise(text, expected):
    assert indic_normalise(text) == expected


LATTICE = [["वेलकम", "welcome", "वेल कम"], ["होता", "hota"], ["है", "hai"]]


def test_lattice_accepts_any_form_including_multi_word_ones():
    assert indic_score(LATTICE, "वेल कम होता है").word_edits == 0
    assert indic_score(LATTICE, "welcome hota hai").word_edits == 0


def test_native_lattice_drops_latin_forms_only_where_a_native_one_exists():
    assert native_lattice([["है", "hai"], ["ok", "okay"]]) == [["है"], ["ok", "okay"]]
    s = indic_score(native_lattice(LATTICE), "welcome hota hai")
    assert (s.word_edits, s.ref_words) == (3, 3)


def test_indic_score_counts_words_and_characters():
    s = indic_score(text_lattice("मैं घर जा रहा हूं"), "मैं घर जा रही हूं")
    assert (s.word_edits, s.ref_words) == (1, 5)
    assert s.char_edits == 1  # ा -> ी
    assert not s.exact
    # joining a compound costs words but no characters (spaces are not counted)
    s = indic_score(text_lattice("मोल भाव"), "मोलभाव")
    assert (s.word_edits, s.char_edits) == (2, 0)


def test_indic_score_of_empty_output_and_insertions():
    s = indic_score(text_lattice("ठीक है"), "")
    assert (s.word_edits, s.ref_words, s.char_edits) == (2, 2, s.ref_chars)
    assert indic_score(text_lattice("ठीक है"), "ठीक है ठीक").word_edits == 1


# --- romanised Hindi: transliteration, spelling fold, scoring -----------------

from stt_scoring import roman_score  # noqa: E402
from translit import fold, non_latin_share, to_latin  # noqa: E402


def test_to_latin_writes_devanagari_in_informal_latin():
    assert to_latin("मैं घर जा रहा हूँ") == "main ghar ja raha hun"
    assert to_latin("समझना कमरा नहीं है") == "samajhna kamra nahin hai"  # schwa deletion
    assert to_latin("ठीक है, में ज़रूर") == "theek hai, mein zaroor"
    assert to_latin("उस section में १५") == "us section mein 15"  # Latin passes, digits ASCII
    assert to_latin("Hello, 7:30") == "Hello, 7:30"


def test_fold_merges_common_romanised_spellings():
    for variants in (
        ["hai", "he", "है"],
        ["mein", "me", "main", "में"],
        ["nahin", "nahi", "nahee", "नहीं"],
        ["accha", "acha", "achha", "अच्छा"],
        ["theek", "thik", "ठीक"],
        ["zaroor", "jarur", "ज़रूर"],
        ["yeh", "ye", "ये"],
        ["woh", "wo", "vo", "वो"],
    ):
        assert len({fold(v) for v in variants}) == 1, variants
    assert fold("kya") != fold("kab")
    assert fold("Hello, World!") == "helo vorld"


def test_roman_score_accepts_devanagari_and_romanised_references():
    lattice = [["मैं"], ["घर"], ["जा"], ["रहा"], ["हूँ"]]
    assert roman_score(lattice, "main ghar ja raha hoon").word_edits == 0
    assert roman_score(lattice, "mai ghar jaa rahaa hu").word_edits == 0
    s = roman_score(lattice, "main ghar ja rahi hun")
    assert (s.word_edits, s.ref_words) == (1, 5)
    # unfolded: transliteration only, so spelling variants count as errors
    assert roman_score(lattice, "mai ghar ja raha hun", fold=False).word_edits == 1
    # Latin English forms in the lattice still count
    assert roman_score([["वेलकम", "welcome"]], "welcome").word_edits == 0


def test_non_latin_share():
    assert non_latin_share("hello 123") == 0.0
    assert non_latin_share("") == 0.0
    assert non_latin_share("ab हा") == 0.5


from guard import collapse_repeats, junk_reason, repeated  # noqa: E402


def test_collapse_repeats_keeps_one_copy_of_a_loop():
    assert collapse_repeats("so install it install it install it install it") == "so install it"
    loop = "Next question is: " * 5
    assert collapse_repeats(loop) == "Next question is:"
    assert collapse_repeats("ji ji ji Jharkhand") == "ji ji ji Jharkhand"  # three is speech
    assert collapse_repeats("haan haan haan haan haan") == "haan"
    assert repeated("a b a b a b", 3) and not repeated("a b a b", 3)


def test_junk_reason_keeps_short_real_answers_and_drops_loops_and_quiet():
    real = {"tokens": 3, "max_new_tokens": 512, "mean_logprob": -0.1}
    assert junk_reason("Yes", 0.8, real, -30.0) == ""
    assert junk_reason("Haan", 0.6, real, -30.0) == ""
    assert junk_reason("Thank you", 2.0, real, -70.0) == "quiet"
    loop = {"tokens": 512, "max_new_tokens": 512, "mean_logprob": -0.1}
    assert junk_reason("thank you sir " * 100, 2.0, loop, -30.0) == "cap"
