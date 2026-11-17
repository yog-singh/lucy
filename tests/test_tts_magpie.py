"""The magpie TTS adapter, with a fake worker script (no model or runtime)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import textwrap
import time
import tomllib
from pathlib import Path

import pytest

import lucy.adapters.tts_magpie as module
from lucy.adapters.tts_magpie import MagpieTTS, TTSError, prepare
from lucy.adapters.worker_process import stop_all
from lucy.core import registry
from lucy.core.config import AdapterConfig
from lucy.core.types import OUTPUT_SAMPLE_RATE

# The fake runs at the core rate (no resampling) unless `rate` says otherwise.
# Each request is answered with `chunks` audio messages; chunk k of request N is
# `samples` int16 samples all equal to N * 100 + k, so tests can tell whose audio it is.
FAKE_WORKER = textwrap.dedent(
    """
    import base64, json, os, queue, subprocess, sys, threading, time
    cfg = json.load(open(sys.argv[1]))
    state = cfg["state"]
    with open(state + ".argv", "w") as f:
        json.dump(sys.argv[2:], f)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(state + ".pids", "a") as f:
        f.write(f"{os.getpid()} {child.pid}\\n")
    print("loading the fake model", file=sys.stderr, flush=True)
    if cfg.get("fatal"):
        print(json.dumps({"fatal": cfg["fatal"]}), flush=True)
        sys.exit(2)
    print(json.dumps({"ready": True, "load_s": 0.01, "sample_rate": cfg.get("rate", 24000),
                      "voices": ["Aria", "Jason", "John", "Leo", "Sofia"]}), flush=True)
    requests, cancelled = queue.Queue(), set()
    def read():
        for line in sys.stdin:
            msg = json.loads(line)
            if "cancel" in msg:
                cancelled.add(msg["cancel"])
                with open(state + ".cancels", "a") as f:
                    f.write(f"{msg['cancel']}\\n")
            else:
                requests.put(msg)
        requests.put(None)
    threading.Thread(target=read, daemon=True).start()
    def out(obj):
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    while (req := requests.get()) is not None:
        rid = req["id"]
        with open(state + ".requests", "a") as f:
            f.write(json.dumps(req, ensure_ascii=False) + "\\n")
        n = 0
        if os.path.exists(state + ".count"):
            n = int(open(state + ".count").read())
        open(state + ".count", "w").write(str(n + 1))
        out({"id": rid, "start": True})
        if n < cfg.get("die_before", 0):
            os._exit(3)
        if n < cfg.get("silent", 0):
            time.sleep(10)
        stopped = False
        for k in range(cfg.get("chunks", 3)):
            if rid in cancelled:
                stopped = True
                break
            value = (rid * 100 + k).to_bytes(2, "little", signed=True)
            pcm = value * cfg.get("samples", 240)
            out({"id": rid, "audio": base64.b64encode(pcm).decode()})
            if n < cfg.get("die_after", 0):
                os._exit(3)
            time.sleep(cfg.get("gap", 0.0))
        out({"id": rid, "done": True, "cancelled": stopped, "first_s": 0.01, "total_s": 0.02})
    """
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = os.popen(f"ps -o stat= -p {pid}").read().strip()
    return bool(status) and not status.startswith("Z")


def _gone(pids: list[int], within: float = 3.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.05)
    return False


class Fake:
    def __init__(self, tmp_path: Path, **cfg: object) -> None:
        self.state = str(tmp_path / "state")
        script = tmp_path / "fake_worker.py"
        script.write_text(FAKE_WORKER)
        config = tmp_path / "cfg.json"
        config.write_text(json.dumps({"state": self.state, **cfg}))
        self.command = [sys.executable, str(script), str(config)]

    def tts(self, **options: object) -> MagpieTTS:
        return MagpieTTS(command=self.command, **options)  # type: ignore[arg-type]

    def _lines(self, suffix: str) -> list[str]:
        path = Path(self.state + suffix)
        return path.read_text().split("\n")[:-1] if path.exists() else []

    def pids(self) -> list[int]:
        return [int(p) for line in self._lines(".pids") for p in line.split()]

    def requests(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self._lines(".requests")]

    def cancels(self) -> list[int]:
        return [int(line) for line in self._lines(".cancels")]


def _values(data: bytes) -> list[int]:
    """The distinct sample values in `data`, in order of appearance."""
    seen: list[int] = []
    for i in range(0, len(data), 2):
        v = int.from_bytes(data[i : i + 2], "little", signed=True)
        if not seen or seen[-1] != v:
            seen.append(v)
    return seen


async def _collect(tts: MagpieTTS, text: str) -> bytes:
    data = b""
    async for chunk in tts.synthesize(text):
        assert chunk.sample_rate == OUTPUT_SAMPLE_RATE
        data += chunk.data
    return data


async def test_streams_chunks_in_order_resampled_to_the_core_format(tmp_path) -> None:
    fake = Fake(tmp_path, rate=22050, chunks=3, samples=2205)  # 0.1 s each at 22.05 kHz
    tts = fake.tts()
    assert not tts.loaded
    data = await _collect(tts, "Hello there.")
    assert abs(len(data) // 2 - 3 * 2400) < 50  # 0.3 s at 24 kHz
    assert tts.sample_rate == 22050
    request = fake.requests()[0]
    assert request["text"] == "Hello there." and request["voice"] == "Sofia"
    assert request["lang"] == "en-US" and request["seed"] == 0
    argv = json.loads(Path(fake.state + ".argv").read_text())
    assert "--threads=4" in argv and "--seed=0" in argv and "--voice=Sofia" in argv
    await tts.aclose()
    assert _gone(fake.pids())


async def test_core_rate_is_passed_through(tmp_path) -> None:
    fake = Fake(tmp_path, chunks=3, samples=240)
    tts = fake.tts()
    data = await _collect(tts, "Hello.")
    assert len(data) == 3 * 240 * 2 and _values(data) == [100, 101, 102]
    await tts.aclose()


async def test_chunks_are_yielded_as_they_arrive(tmp_path) -> None:
    fake = Fake(tmp_path, chunks=3, gap=0.3, samples=2205)  # 0.1 s chunks: soxr yields at once
    tts = fake.tts()
    await tts.warm_up()
    started = time.monotonic()
    stream = tts.synthesize("Hello.")
    await anext(stream)
    assert time.monotonic() - started < 0.25  # not after all three chunks
    await stream.aclose()
    await tts.aclose()


async def test_language_follows_the_script(tmp_path) -> None:
    fake = Fake(tmp_path, chunks=1)
    tts = fake.tts()
    texts = (
        "ठीक है।",
        "आपकी meeting 7:30 बजे है।",
        "Theek hai.",
        "Sure, the Taj Mahal is in Agra (आगरा).",
        "Ok.",
    )
    for text in texts:
        await _collect(tts, text)
    requests = fake.requests()
    # "Theek hai." is romanised Hindi: converted and sent as hi-IN.
    assert [r["lang"] for r in requests] == ["hi-IN", "hi-IN", "hi-IN", "en-US", "en-US"]
    assert requests[1]["text"] == "आपकी मीटिंग 7:30 बजे है।"
    assert requests[2]["text"] == "ठीक है."
    assert requests[3]["text"] == "Sure, the Taj Mahal is in Agra (आगरा)."
    await tts.aclose()
    (tmp_path / "forced").mkdir()
    forced = Fake(tmp_path / "forced", chunks=1)
    tts = forced.tts(language="hi-IN")
    await _collect(tts, "Theek hai.")
    assert forced.requests()[0]["lang"] == "hi-IN"
    assert not re.search("[A-Za-z]", forced.requests()[0]["text"])  # never letter by letter
    await tts.aclose()


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Your meeting is at 7:30 pm on 18 November.", "en-US"),
        ("That comes to 1,250 rupees, 40% off.", "en-US"),
        ("आपकी मीटिंग 18 नवंबर को शाम 7:30 बजे है।", "hi-IN"),
        ("कुल 1250 रुपये, 40% छूट के साथ, तापमान 32 डिग्री।", "hi-IN"),
        ("₹1,250 का बिल 3/12 को आया।", "hi-IN"),
    ],
)
def test_digits_are_left_to_the_normaliser(text, lang) -> None:
    assert prepare(text) == (text, lang)


def test_prepare() -> None:
    assert prepare("Sure.") == ("Sure.", "en-US")
    assert prepare("कल आपकी doctor के साथ meeting है।") == ("कल आपकी डॉक्टर के साथ मीटिंग है।", "hi-IN")
    # Mostly Latin words with a Devanagari name: read as English.
    assert prepare("Call Rahul (राहुल) at nine tomorrow.")[1] == "en-US"
    text, lang = prepare("Sure.", "hi-IN")
    assert lang == "hi-IN" and not re.search("[A-Za-z]", text)
    assert prepare("ठीक है।", "en-US") == ("ठीक है।", "en-US")


async def test_early_close_cancels_and_the_next_request_is_clean(tmp_path) -> None:
    fake = Fake(tmp_path, chunks=6, gap=0.1)
    tts = fake.tts()
    stream = tts.synthesize("A long sentence.")
    first = await anext(stream)
    assert _values(first.data) == [100]
    await stream.aclose()
    data = await _collect(tts, "Next.")
    assert _values(data) == [200, 201, 202, 203, 204, 205]  # nothing of request 1
    assert fake.cancels() == [1]
    await tts.aclose()


async def test_lookahead_overlap_read_in_any_order(tmp_path) -> None:
    fake = Fake(tmp_path, chunks=3, gap=0.05)
    tts = fake.tts()
    first, second = tts.synthesize("One."), tts.synthesize("Two.")
    one = (await anext(first)).data  # sentence one is being synthesised...
    two = b""
    async for chunk in second:  # ...and sentence two is read to the end before it
        two += chunk.data
    async for chunk in first:
        one += chunk.data
    assert _values(one) == [100, 101, 102] and _values(two) == [200, 201, 202]
    await tts.aclose()


async def test_concurrent_consumers(tmp_path) -> None:
    fake = Fake(tmp_path, chunks=3, gap=0.02)
    tts = fake.tts()
    one, two = await asyncio.gather(_collect(tts, "One."), _collect(tts, "Two."))
    assert _values(one) == [100, 101, 102] and _values(two) == [200, 201, 202]
    await tts.aclose()


async def test_timeout_to_first_audio_stops_the_worker(tmp_path) -> None:
    fake = Fake(tmp_path, silent=1)
    tts = fake.tts(first_audio_timeout=0.5)
    with pytest.raises(TTSError, match="took too long for the first audio"):
        await _collect(tts, "Hello.")
    assert _gone(fake.pids()[:2])
    assert _values(await _collect(tts, "Again.")) == [200, 201, 202]  # restarted
    await tts.aclose()


async def test_overall_timeout(tmp_path) -> None:
    fake = Fake(tmp_path, chunks=10, gap=0.2)
    tts = fake.tts(timeout=0.5)
    with pytest.raises(TTSError, match="the whole sentence"):
        await _collect(tts, "Hello.")
    await tts.aclose()


async def test_crash_before_audio_is_retried_once(tmp_path) -> None:
    fake = Fake(tmp_path, die_before=1)
    tts = fake.tts()
    assert _values(await _collect(tts, "Hello.")) == [200, 201, 202]
    assert len(fake.pids()) == 4  # two workers, each with its child
    await tts.aclose()
    assert _gone(fake.pids())


async def test_crash_twice_before_audio_fails(tmp_path) -> None:
    fake = Fake(tmp_path, die_before=2)
    tts = fake.tts()
    with pytest.raises(TTSError, match="died"):
        await _collect(tts, "Hello.")
    await tts.aclose()


async def test_crash_after_audio_is_an_error(tmp_path) -> None:
    fake = Fake(tmp_path, die_after=1)
    tts = fake.tts()
    got = []
    with pytest.raises(TTSError, match="died during a sentence"):
        async for chunk in tts.synthesize("Hello."):
            got.append(chunk)
    assert len(got) == 1
    assert _values(await _collect(tts, "Again.")) == [200, 201, 202]  # restarted
    await tts.aclose()


async def test_unknown_voice_lists_the_valid_ones(tmp_path) -> None:
    fake = Fake(tmp_path)
    tts = fake.tts(voice="Zed")
    with pytest.raises(TTSError, match="unknown voice 'Zed'.*Aria, Jason, John, Leo, Sofia"):
        await tts.warm_up()
    assert _gone(fake.pids())
    await tts.aclose()


async def test_voice_is_case_insensitive(tmp_path) -> None:
    tts = Fake(tmp_path).tts(voice="aria")
    await tts.warm_up()
    await tts.aclose()


async def test_fatal_load_names_what_to_download(tmp_path) -> None:
    tts = Fake(tmp_path, fatal="RuntimeError: cannot open model").tts()
    with pytest.raises(TTSError, match="cannot open model.*docs/voice.md"):
        await tts.warm_up()
    await tts.aclose()


async def test_aclose_and_interpreter_exit_leave_no_process(tmp_path) -> None:
    fake = Fake(tmp_path)
    tts = fake.tts()
    await tts.warm_up()
    assert tts.loaded and fake.pids()
    stop_all()
    assert _gone(fake.pids())
    await tts.aclose()
    await tts.aclose()  # idempotent
    with pytest.raises(TTSError, match="closed"):
        await _collect(tts, "Hello.")


async def test_blank_text_starts_nothing(tmp_path) -> None:
    fake = Fake(tmp_path)
    tts = fake.tts()
    assert await _collect(tts, "  ") == b""
    assert fake.pids() == []
    await tts.aclose()


async def test_missing_files_are_named(tmp_path) -> None:
    tts = MagpieTTS(root=str(tmp_path / "nowhere"))
    with pytest.raises(TTSError, match="missing runtime library folder.*model.*codec.*tokenizer"):
        await tts.warm_up()
    await tts.aclose()
    root = tmp_path / "nemo"
    tokenizer = "models/nvidia/magpie_tts_multilingual_357m/abc/tokenizer"
    for rel in (module.LIB_DIR, module.TN_DIR, tokenizer):
        (root / rel).mkdir(parents=True)
    for rel in (module.MODEL_FILE, module.CODEC_FILE, module.LIB_DIR + "/libnemo_speech_tts.dylib"):
        (root / rel).write_bytes(b"")
    paths = module.resolve_paths(str(root), None, None, None, None)
    assert paths["tokenizer folder"].parent.name == "abc"
    (root / module.CODEC_FILE).unlink()
    with pytest.raises(TTSError, match=r"missing codec \(.*nano_codec.*\): install NeMo-Speech"):
        module.resolve_paths(str(root), None, None, None, None)


async def test_warm_up_covers_both_languages(tmp_path) -> None:
    fake = Fake(tmp_path, chunks=1)
    tts = fake.tts()
    await tts.warm_up()
    assert [r["lang"] for r in fake.requests()] == ["en-US", "hi-IN"]
    await tts.aclose()
    (tmp_path / "forced").mkdir()
    forced = Fake(tmp_path / "forced", chunks=1)
    tts = forced.tts(language="en-US")
    await tts.warm_up()
    assert [r["lang"] for r in forced.requests()] == ["en-US"]
    await tts.aclose()


@pytest.mark.parametrize(
    "options",
    [
        {"voice": ""},
        {"language": "hi"},
        {"threads": 0},
        {"threads": True},
        {"seed": 1.5},
        {"seed": -1},
        {"root": ""},
        {"timeout": 0},
        {"first_audio_timeout": -1},
    ],
)
def test_bad_options_are_rejected(options) -> None:
    with pytest.raises(ValueError):
        MagpieTTS(**options)


def test_registered_and_built_from_config() -> None:
    from lucy.app import BUILTIN_ADAPTER_MODULES

    assert "lucy.adapters.tts_magpie" in BUILTIN_ADAPTER_MODULES
    assert "lucy.adapters.tts_maya2" not in BUILTIN_ADAPTER_MODULES
    assert registry.get("tts", "magpie") is MagpieTTS
    text = """
    adapter = "magpie"
    voice = "Aria"
    threads = 4
    seed = 0
    """
    config = AdapterConfig.model_validate(tomllib.loads(textwrap.dedent(text)))
    tts = registry.create("tts", config.adapter, **config.options)
    assert isinstance(tts, MagpieTTS) and tts.voice == "Aria" and tts.threads == 4


def test_worker_script_is_shipped_and_uses_only_the_standard_library() -> None:
    script = module.WORKER_SCRIPT.read_text()
    assert "import ctypes" in script and "nemo_speech_tts_synthesize_text" in script
    assert "import lucy" not in script and "numpy" not in script


def test_audio_payload_is_base64_pcm() -> None:
    assert base64.b64decode(base64.b64encode(b"\x01\x00")) == b"\x01\x00"


@pytest.mark.parametrize(
    ("word", "spoken"),
    [
        ("meeting", "मीटिंग"),
        ("Meeting", "मीटिंग"),
        ("calendar", "कैलेंडर"),
        ("PM", "पीएम"),
        ("time", "टाइम"),
        ("late", "लेट"),
        ("teacher", "टीचर"),
        ("weekend", "वीकेंड"),
        ("happy", "हैपी"),
    ],
)
def test_respell_word(word, spoken) -> None:
    from lucy.core.devanagari import respell_word

    assert respell_word(word) == spoken


def test_respell_latin_words_counts_words() -> None:
    from lucy.core.devanagari import respell_latin_words

    assert respell_latin_words("आपकी meeting कल 7:30 बजे है।") == ("आपकी मीटिंग कल 7:30 बजे है।", 1, 5)
    assert respell_latin_words("12 34") == ("12 34", 0, 0)


async def test_romanised_hindi_routing(tmp_path) -> None:
    fake = Fake(tmp_path, chunks=1)
    tts = fake.tts()
    texts = (
        "Kal aapke calendar mein teen events hain.",  # romanised Hindi: converted
        "Ok.",  # too short: follows the previous sentence (Hindi)
        "What is the capital of Japan?",  # English stays en-US
        "Ok.",  # follows English
        "मैं ठीक हूँ।",  # Devanagari as is
    )
    for text in texts:
        await _collect(tts, text)
    requests = fake.requests()
    assert [r["lang"] for r in requests] == ["hi-IN", "hi-IN", "en-US", "en-US", "hi-IN"]
    assert requests[0]["text"] == "कल आपके कैलेंडर में तीन इवेंट्स हैं."
    assert requests[1]["text"] == "ओके."
    assert requests[2]["text"] == "What is the capital of Japan?"
    assert requests[4]["text"] == "मैं ठीक हूँ।"
    await tts.aclose()
    (tmp_path / "off").mkdir()
    off = Fake(tmp_path / "off", chunks=1)
    tts = off.tts(roman_hindi=False)
    await _collect(tts, "Kal aapke calendar mein teen events hain.")
    assert off.requests()[0] == {
        **off.requests()[0],
        "lang": "en-US",
        "text": "Kal aapke calendar mein teen events hain.",
    }
    await tts.aclose()


def test_prepare_romanised_hindi() -> None:
    assert prepare("Main theek hoon, shukriya.") == ("मैं ठीक हूँ, शुक्रिया.", "hi-IN")
    assert prepare("Ok.") == ("Ok.", "en-US")
    assert prepare("Ok.", previous="hi-IN") == ("ओके.", "hi-IN")
    assert prepare("The main reason is cost.", previous="hi-IN")[1] == "en-US"
    assert prepare("Main theek hoon.", roman_hindi=False) == ("Main theek hoon.", "en-US")
    assert prepare("Main theek hoon.", "en-US") == ("Main theek hoon.", "en-US")
