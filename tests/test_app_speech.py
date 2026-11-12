"""LucyApp builds speech adapters only for voice (speech=True)."""

from pathlib import Path

import pytest

from lucy.app import LucyApp
from lucy.core.config import AdapterConfig, LLMSettings, Settings
from lucy.core.registry import Registry
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.fakes import FakeLLM, adapter_for
from tests.voice_fakes import FakeSTT, FakeTTS


class Warmable:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.capabilities = inner.capabilities
        self.warmed = 0
        self.closed = 0

    async def warm_up(self) -> None:
        self.warmed += 1

    async def aclose(self) -> None:
        self.closed += 1


class WarmSTT(Warmable):
    async def transcribe(self, audio):
        return await self.inner.transcribe(audio)


class WarmTTS(Warmable):
    def synthesize(self, text):
        return self.inner.synthesize(text)


def setup(tmp_path: Path, *, stt: bool = True, tts: bool = True):
    stt_adapter, tts_adapter = WarmSTT(FakeSTT()), WarmTTS(FakeTTS())
    registry = Registry()
    registry.register("llm", "fake")(adapter_for(FakeLLM([])))
    registry.register("memory", "sqlite")(SQLiteMemoryStore)
    registry.register("stt", "fake")(adapter_for(stt_adapter))
    registry.register("tts", "fake")(adapter_for(tts_adapter))
    settings = Settings(
        llm=LLMSettings(roles={"chat": AdapterConfig(adapter="fake")}),
        memory=AdapterConfig(adapter="sqlite", options={"path": str(tmp_path / "lucy.db")}),
        stt=AdapterConfig(adapter="fake", options={"model": "m"}) if stt else None,
        tts=AdapterConfig(adapter="fake") if tts else None,
    )
    return settings, registry, stt_adapter, tts_adapter


async def test_chat_builds_no_speech_adapters(tmp_path: Path) -> None:
    settings, registry, stt, tts = setup(tmp_path)
    async with LucyApp(settings, registry=registry) as app:
        with pytest.raises(RuntimeError, match="speech=True"):
            _ = app.stt
        with pytest.raises(RuntimeError, match="speech=True"):
            _ = app.tts
    assert stt.warmed == tts.warmed == 0


async def test_voice_builds_warms_and_closes_them(tmp_path: Path) -> None:
    settings, registry, stt, tts = setup(tmp_path)
    async with LucyApp(settings, registry=registry, speech=True) as app:
        assert app.stt is stt and app.tts is tts
        assert stt.warmed == tts.warmed == 1
    assert stt.closed == tts.closed == 1
    with pytest.raises(RuntimeError):
        _ = app.stt


@pytest.mark.parametrize(
    ("stt", "tts", "missing"), [(False, True, "[stt]"), (True, False, "[tts]")]
)
async def test_voice_without_a_speech_section_fails_to_start(
    tmp_path: Path, stt: bool, tts: bool, missing: str
) -> None:
    settings, registry, *_ = setup(tmp_path, stt=stt, tts=tts)
    app = LucyApp(settings, registry=registry, speech=True)
    with pytest.raises(ValueError, match=f"voice needs \\{missing}"):
        await app.start()
