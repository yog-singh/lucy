"""The indic_flex STT adapter against a fake worker (a tiny script speaking the protocol;
no uv, no PyTorch, no model)."""

import json
import logging
import os
import sys
import textwrap
import time
import tomllib
from pathlib import Path

import numpy as np
import pytest

import lucy.adapters.stt_indic_flex as module
from lucy.adapters.stt_indic_flex import IndicFlexSTT, STTError
from lucy.adapters.stt_indic_flex_guard import Guard, collapse_repeats, junk_reason
from lucy.core import registry
from lucy.core.config import AdapterConfig
from lucy.core.types import AudioChunk
from tests.voice_fakes import silence, tone

FAKE_WORKER = textwrap.dedent(
    """
    import base64, json, os, subprocess, sys, time
    cfg = json.load(open(sys.argv[1]))
    state = cfg["state"]
    with open(state + ".argv", "w") as f:
        json.dump(sys.argv[2:], f)
    with open(state + ".env", "w") as f:
        json.dump(dict(os.environ), f)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(state + ".pids", "a") as f:
        f.write(f"{os.getpid()} {child.pid}\\n")
    print("loading the fake model", file=sys.stderr, flush=True)
    print("a library printed this")  # stdout noise that is not JSON
    if cfg.get("fatal"):
        print(json.dumps({"fatal": cfg["fatal"]}), flush=True)
        sys.exit(2)
    print(json.dumps({"ready": True, "load_s": 0.01, "device": "mps", "dtype": "float16"}),
          flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        n = 0
        if os.path.exists(state + ".count"):
            n = int(open(state + ".count").read())
        open(state + ".count", "w").write(str(n + 1))
        if n < cfg.get("die", 0):
            os._exit(3)
        if n < cfg.get("slow", 0):
            time.sleep(5)
        seconds = len(base64.b64decode(req["pcm"])) / 32000
        reply = {"id": req["id"], "text": cfg.get("text", "hello world"),
                 "lang": cfg.get("lang", "en"), "tokens": cfg.get("tokens", 3),
                 "mean_logprob": cfg.get("mean_logprob", -0.1),
                 "max_new_tokens": int(10 + 25 * seconds), "seconds": 0.01}
        if cfg.get("error"):
            reply = {"id": req["id"], "error": cfg["error"]}
        print(json.dumps(reply, ensure_ascii=False), flush=True)
    """
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie (exited, not yet reaped by its parent) counts as gone.
    status = os.popen(f"ps -o stat= -p {pid}").read().strip()
    return bool(status) and not status.startswith("Z")


class Fake:
    def __init__(self, tmp_path: Path, **cfg: object) -> None:
        self.dir = tmp_path
        self.state = str(tmp_path / "state")
        script = tmp_path / "fake_worker.py"
        script.write_text(FAKE_WORKER)
        config = tmp_path / "cfg.json"
        config.write_text(json.dumps({"state": self.state, **cfg}))
        self.command = [sys.executable, str(script), str(config)]

    def stt(self, **options: object) -> IndicFlexSTT:
        return IndicFlexSTT(command=self.command, **options)  # type: ignore[arg-type]

    def pids(self) -> list[int]:
        path = Path(self.state + ".pids")
        if not path.exists():
            return []
        return [int(p) for p in path.read_text().split()]

    def argv(self) -> list[str]:
        return json.loads(Path(self.state + ".argv").read_text())

    def env(self) -> dict[str, str]:
        return json.loads(Path(self.state + ".env").read_text())


def _subdir(path: Path, name: str) -> Path:
    (path / name).mkdir()
    return path / name


def _gone(pids: list[int], within: float = 3.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.05)
    return False


async def test_happy_path_is_lazy_and_sets_the_language(tmp_path) -> None:
    fake = Fake(tmp_path)
    stt = fake.stt()
    assert not stt.loaded and fake.pids() == []
    result = await stt.transcribe(tone(1.0))
    assert result.text == "hello world" and result.language == "en"
    assert stt.loaded
    await stt.aclose()
    assert _gone(fake.pids())


async def test_devanagari_from_the_worker_comes_out_in_latin(tmp_path) -> None:
    fake = Fake(tmp_path, text="मैं घर जा रहा हूँ", lang="hi")
    stt = fake.stt()
    result = await stt.transcribe(tone(1.0))
    assert result.text == "main ghar ja raha hun" and result.language == "hi"
    await stt.aclose()


async def test_native_output_when_latin_is_off(tmp_path) -> None:
    fake = Fake(tmp_path, text="नमस्ते", lang="hi")
    stt = fake.stt(latin=False)
    assert (await stt.transcribe(tone(1.0))).text == "नमस्ते"
    await stt.aclose()


async def test_warm_up_starts_the_worker_and_the_args_follow_the_options(tmp_path) -> None:
    fake = Fake(tmp_path)
    stt = fake.stt()
    await stt.warm_up()
    assert stt.loaded
    argv = fake.argv()
    assert "--lang=auto" in argv and "--allowed=en,hi" in argv and "--mode=mixed" in argv
    assert "--dtype=float16" in argv and "--tokens-base=10.0" in argv
    await stt.aclose()
    fake2 = Fake(_subdir(tmp_path, "b"))
    stt = fake2.stt(languages=["hi"], device="cpu")
    await stt.warm_up()
    argv = fake2.argv()
    assert "--lang=hi" in argv and not any(a.startswith("--allowed") for a in argv)
    assert "--dtype=float32" in argv
    await stt.aclose()


async def test_the_worker_gets_a_minimal_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUCY_TEST_SECRET", "s3cret")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/uvcache")
    fake = Fake(tmp_path)
    stt = fake.stt()
    await stt.warm_up()
    env = fake.env()
    assert "LUCY_TEST_SECRET" not in env and env["UV_CACHE_DIR"] == "/tmp/uvcache"
    assert env["PYTHONUNBUFFERED"] == "1"
    await stt.aclose()


async def test_input_checks_and_the_silence_shortcut(tmp_path) -> None:
    fake = Fake(tmp_path)
    stt = fake.stt()
    with pytest.raises(ValueError):
        await stt.transcribe(AudioChunk(data=bytes(3200), sample_rate=8000))
    with pytest.raises(ValueError):
        await stt.transcribe(AudioChunk(data=bytes(3201), sample_rate=16000))
    assert (await stt.transcribe(tone(0.05))).text == ""
    assert (await stt.transcribe(silence(1.0))).text == ""
    assert fake.pids() == []  # the worker was never needed
    await stt.aclose()


@pytest.mark.parametrize(
    ("cfg", "audio", "reason"),
    [
        ({"tokens": 35}, tone(1.0), "cap"),  # 35 = the cap for 1 s
        ({"tokens": 30}, tone(1.0), "rate"),  # under the fake's cap of 35, over 10 + 10 x 1
        ({"mean_logprob": -2.0}, tone(1.0), "unsure"),
        ({"text": "REDACTED"}, tone(1.0), "redacted"),
        ({"text": "Thank you"}, tone(1.0, amp=0.002), "quiet"),  # -57 dBFS: sent, then dropped
    ],
)
async def test_each_guard_rule_drops_the_transcript(tmp_path, caplog, cfg, audio, reason) -> None:
    options = {"max_tokens_per_s": 10.0} if reason == "rate" else {}
    fake = Fake(tmp_path, **cfg)
    stt = fake.stt(**options)
    with caplog.at_level(logging.INFO, logger=module.__name__):
        result = await stt.transcribe(audio)
    assert result.text == ""
    assert f"dropped a transcript ({reason}" in caplog.text
    await stt.aclose()


async def test_a_loop_is_collapsed_to_one_copy(tmp_path) -> None:
    fake = Fake(tmp_path, text="install it install it install it install it now", tokens=12)
    stt = fake.stt()
    assert (await stt.transcribe(tone(2.0))).text == "install it now"
    await stt.aclose()


async def test_a_slow_request_times_out_and_the_next_one_restarts(tmp_path) -> None:
    fake = Fake(tmp_path, slow=1)
    stt = fake.stt(timeout=0.5)
    with pytest.raises(STTError, match="longer than"):
        await stt.transcribe(tone(1.0))
    first = fake.pids()
    assert _gone(first)
    assert (await stt.transcribe(tone(1.0))).text == "hello world"
    assert len(fake.pids()) == 4  # a second worker (and its child)
    await stt.aclose()
    assert _gone(fake.pids())


async def test_a_crash_is_restarted_once_then_reported(tmp_path) -> None:
    fake = Fake(tmp_path, die=1)
    stt = fake.stt()
    assert (await stt.transcribe(tone(1.0))).text == "hello world"
    await stt.aclose()
    fake = Fake(_subdir(tmp_path, "b"), die=5)
    stt = fake.stt()
    with pytest.raises(STTError, match="died twice"):
        await stt.transcribe(tone(1.0))
    await stt.aclose()
    assert _gone(fake.pids())


async def test_a_worker_error_reply_is_an_stt_error(tmp_path) -> None:
    fake = Fake(tmp_path, error="RuntimeError: out of memory")
    stt = fake.stt()
    with pytest.raises(STTError, match="out of memory"):
        await stt.transcribe(tone(1.0))
    await stt.aclose()


async def test_a_fatal_load_error_is_clear(tmp_path) -> None:
    fake = Fake(tmp_path, fatal="cannot load the model: access is refused")
    stt = fake.stt()
    with pytest.raises(STTError, match="access is refused"):
        await stt.warm_up()
    assert _gone(fake.pids())
    await stt.aclose()


async def test_shutdown_leaves_no_process(tmp_path) -> None:
    fake = Fake(tmp_path)
    stt = fake.stt()
    await stt.warm_up()
    pids = fake.pids()
    assert len(pids) == 2 and all(_alive(p) for p in pids)
    await stt.aclose()
    await stt.aclose()  # idempotent
    assert _gone(pids)
    with pytest.raises(STTError):
        await stt.transcribe(tone(1.0))


async def test_interpreter_exit_stops_live_workers(tmp_path) -> None:
    fake = Fake(tmp_path)
    stt = fake.stt()
    await stt.warm_up()
    module._stop_all()
    assert _gone(fake.pids())
    await stt.aclose()


async def test_the_worker_stderr_goes_to_the_debug_log(tmp_path, caplog) -> None:
    fake = Fake(tmp_path)
    stt = fake.stt()
    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        await stt.warm_up()
        await stt.aclose()
    assert "[stt worker] loading the fake model" in caplog.text


async def test_prepare_only_when_running_and_idle(tmp_path, monkeypatch) -> None:
    fake = Fake(tmp_path)
    stt = fake.stt()
    await stt.prepare()
    assert fake.pids() == []  # not started by prepare
    await stt.warm_up()
    monkeypatch.setattr(module, "PREPARE_AFTER", 0.0)
    await stt.prepare()
    assert Path(fake.state + ".count").read_text() == "2"
    await stt.aclose()
    await stt.prepare()  # closed: nothing, no error


async def test_missing_uv_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(module, "find_uv", lambda: None)
    stt = IndicFlexSTT()
    with pytest.raises(STTError, match="needs `uv`"):
        await stt.warm_up()
    await stt.aclose()


def test_find_uv_prefers_the_uv_variable(tmp_path, monkeypatch) -> None:
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o755)
    monkeypatch.setenv("UV", str(uv))
    assert module.find_uv() == str(uv)


@pytest.mark.parametrize(
    "options",
    [
        {"languages": []},
        {"languages": "en"},
        {"languages": ["en", "ta"]},
        {"mode": "loud"},
        {"device": "cuda"},
        {"latin": "yes"},
        {"timeout": 0},
        {"revision": "main"},
        {"min_dbfs": 3.0},
        {"min_mean_logprob": 0.5},
        {"max_repeats": 1},
    ],
)
def test_bad_options_are_rejected(options) -> None:
    with pytest.raises(ValueError):
        IndicFlexSTT(**options)


def test_registered_and_configured_from_toml() -> None:
    import lucy.app

    assert "lucy.adapters.stt_indic_flex" in lucy.app.BUILTIN_ADAPTER_MODULES
    assert registry.get("stt", "indic_flex") is IndicFlexSTT
    raw = tomllib.loads(
        'adapter = "indic_flex"\nlanguages = ["en", "hi"]\ntimeout = 15\nmin_mean_logprob = -1.2\n'
    )
    stt_config = AdapterConfig.model_validate(raw)
    assert stt_config.adapter == "indic_flex"
    stt = registry.create("stt", "indic_flex", **stt_config.options)
    assert isinstance(stt, IndicFlexSTT)
    assert stt.timeout == 15.0 and stt.guard.min_mean_logprob == -1.2
    assert stt.capabilities.streaming is False


def test_guard_rules_directly() -> None:
    real = {"tokens": 3, "max_new_tokens": 35, "mean_logprob": -0.1}
    assert junk_reason("Yes", 0.8, real, -30.0) == ""
    assert junk_reason("", 0.8, real, -80.0) == ""
    assert junk_reason("Yes", 0.8, real, -56.0) == "quiet"
    assert junk_reason("x", 1.0, {"tokens": 35, "max_new_tokens": 35}, -30.0) == "cap"
    assert junk_reason("x", 1.0, {"tokens": 36}, -30.0) == "rate"
    assert junk_reason("x", 1.0, {"tokens": 3, "mean_logprob": -1.5}, -30.0) == "unsure"
    assert junk_reason(" REDACTED. ", 1.0, real, -30.0) == "redacted"
    assert junk_reason("it was REDACTED today", 1.5, real, -30.0) == ""
    assert Guard().token_cap(2.0) == 60 and Guard().token_cap(100.0) == 512
    assert collapse_repeats("ji ji ji Jharkhand") == "ji ji ji Jharkhand"


def test_pcm_level() -> None:
    assert module._pcm_dbfs(b"") == -np.inf
    assert module._pcm_dbfs(silence(0.5).data) == -np.inf
    assert -10 < module._pcm_dbfs(tone(0.5).data) < -8
