"""The VAD protocol, the scripted fake, and the vendored Silero model file (not loaded)."""

import hashlib
from importlib import resources

import pytest

from lucy.voice.vad import SILERO_MODEL, VAD, ScriptedVAD, SileroVAD
from tests.voice_fakes import silence, tone

SILERO_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


def test_the_vendored_model_and_its_notice_ship_with_the_package() -> None:
    models = resources.files("lucy.voice").joinpath("models")
    data = models.joinpath(SILERO_MODEL).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SILERO_SHA256
    notice = models.joinpath("NOTICE.md").read_text()
    assert "MIT License" in notice and SILERO_SHA256 in notice


def test_silero_is_lazy_and_satisfies_the_protocol() -> None:
    vad = SileroVAD()
    assert isinstance(vad, VAD)
    assert vad._session is None  # nothing loaded until used


def test_scripted_vad_plays_its_script_then_the_default() -> None:
    vad = ScriptedVAD([0.9, 0.1], default=0.3)
    frame = silence(0.032)
    assert [vad(frame) for _ in range(4)] == [0.9, 0.1, 0.3, 0.3]
    assert vad.calls == 4
    vad.reset()
    assert vad.resets == 1


def test_scripted_vad_by_level() -> None:
    vad = ScriptedVAD(level=0.1)
    assert vad(tone(0.032, amp=0.5)) == 1.0
    assert vad(tone(0.032, amp=0.05)) == 0.0
    assert vad(silence(0.032)) == 0.0


@pytest.mark.live
def test_silero_tells_speech_from_silence() -> None:
    """Loads the real (vendored) model; synthetic tones are not speech."""
    vad = SileroVAD()
    vad.load()
    assert vad(silence(0.032)) < 0.1
    probabilities = [vad(tone(0.032, amp=0.3)) for _ in range(10)]
    assert max(probabilities) < 0.5
