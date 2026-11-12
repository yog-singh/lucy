"""Live speech tests: real phonon on Ollama and real Parakeet (opt in with `pytest -m live`).

Needs Ollama serving `phonon` and the Parakeet weights in the Hugging Face cache.
"""

import re

import pytest

from lucy.adapters.stt_parakeet_mlx import ParakeetMLXSTT
from lucy.adapters.tts_openai_compat import OllamaTTS
from lucy.core.types import OUTPUT_SAMPLE_RATE, AudioChunk
from lucy.voice.audio import float_to_pcm, pcm_to_float, resample

pytestmark = pytest.mark.live

SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank."


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z ]", "", text.lower()).split()


async def synthesise(text: str, voice: str = "default") -> AudioChunk:
    tts = OllamaTTS(model="phonon", voice=voice)
    try:
        chunks = [chunk async for chunk in tts.synthesize(text)]
    finally:
        await tts.aclose()
    assert len(chunks) > 1  # streamed, not delivered whole
    assert all(len(c.data) % 2 == 0 for c in chunks)
    return AudioChunk(data=b"".join(c.data for c in chunks), sample_rate=OUTPUT_SAMPLE_RATE)


async def test_phonon_streams_speech_in_the_core_output_format() -> None:
    audio = await synthesise(SENTENCE)
    assert audio.sample_rate == OUTPUT_SAMPLE_RATE
    assert 2.0 < audio.duration < 10.0
    assert abs(pcm_to_float(audio.data)).max() > 0.05


async def test_parakeet_transcribes_synthesised_speech() -> None:
    audio = await synthesise(SENTENCE, voice="Hadley")
    at_16k = resample(pcm_to_float(audio.data), OUTPUT_SAMPLE_RATE, 16000)
    stt = ParakeetMLXSTT()
    try:
        await stt.warm_up()
        transcript = await stt.transcribe(AudioChunk(data=float_to_pcm(at_16k), sample_rate=16000))
        silent = await stt.transcribe(AudioChunk(data=bytes(32000), sample_rate=16000))
    finally:
        await stt.aclose()
    expected, heard = words(SENTENCE), words(transcript.text)
    missed = len(set(expected) - set(heard))
    assert missed <= 1, transcript.text
    assert silent.text == ""
