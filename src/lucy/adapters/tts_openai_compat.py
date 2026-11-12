"""TTS adapter for OpenAI-compatible `/audio/speech` servers (Ollama, LM Studio, ...).

Asks for raw PCM (`response_format = "pcm"`: signed 16-bit little-endian mono)
and yields it as core output-format `AudioChunk`s while the response streams,
so playback can start on the first bytes. Chunks always hold whole samples: an
odd byte left at the end of a network read is carried over to the next.

Probed on Ollama 0.40.16 with `phonon:latest` (2026-11-12):

- `POST /v1/audio/speech` with `{"model", "input", "voice", "response_format":
  "pcm"}` answers `200`, `Content-Type: audio/pcm`, chunked, the first bytes in
  about 0.01 to 0.04 s once the model is loaded (about 1.1 s when it is not).
- The rate is 24 kHz: raw PCM carries no header, so it was checked by asking the
  same endpoint for `"wav"`, whose header says 24000 Hz, mono, 16-bit. It is
  also OpenAI's documented PCM format. `sample_rate` covers a server that
  differs; its audio is resampled to the core format.
- Voices: `default` plus 37 named ones; an unknown voice is a 400 that lists
  them. `speed` other than 1, `instructions` and `stream_format = "sse"` are
  rejected with 400, so `speed` is sent only when configured.
- Errors are OpenAI-style JSON: `{"error": {"message": ...}}`; an unknown model
  is a 404, a model without speech a 400.

Stopping speech early is closing the generator (`aclose`), which closes the
HTTP response at once.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Final

import httpx

from lucy.core import registry
from lucy.core.types import OUTPUT_SAMPLE_RATE, SAMPLE_WIDTH_BYTES, AudioChunk, TTSCaps

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL: Final = "http://localhost:11434/v1"
"""Default endpoint of Ollama's OpenAI-compatible API."""


class TTSError(RuntimeError):
    """The speech server could not be reached, refused the request or sent broken audio."""


def _error_detail(response: httpx.Response) -> str:
    """The server's error message from a non-200 response body."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:500] or response.reason_phrase
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str):
        return error
    return json.dumps(body)[:500]


@registry.register("tts", "openai_compat")
class OpenAICompatTTS:
    """Text to speech over the OpenAI-compatible `/audio/speech` API, streamed as raw PCM.

    Options come from config: `model`, `voice` (sent when set), `speed` (sent
    when set; Ollama accepts only 1), `sample_rate` (the server's PCM rate,
    24000 for OpenAI and Ollama; anything else is resampled), `timeout` in
    seconds for connecting and between received bytes, `api_key`, and
    `transport` for tests.
    """

    server_name: str = "the speech server"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        voice: str | None = None,
        speed: float | None = None,
        sample_rate: int = OUTPUT_SAMPLE_RATE,
        api_key: str | None = None,
        timeout: float = 60,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.model = model
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.base_url = base_url.rstrip("/")
        self.capabilities = TTSCaps(streaming=True)
        self._timeout = timeout
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _where(self) -> str:
        return f"{self.server_name} at {self.base_url}"

    async def warm_up(self) -> None:
        """Synthesise one word and discard it, so the server loads the model now
        (about 1 s on Ollama) rather than before the first reply. Raises `TTSError`."""
        async for _ in self.synthesize("Ready."):
            pass

    def _request_body(self, text: str) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "input": text, "response_format": "pcm"}
        if self.voice is not None:
            body["voice"] = self.voice
        if self.speed is not None:
            body["speed"] = self.speed
        return body

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Stream `text` as speech in the core output format (24 kHz mono 16-bit).

        Blank text yields nothing without a request. Raises `TTSError`.
        """
        if self._client.is_closed:
            raise TTSError(f"the TTS adapter for {self.model!r} has been closed")
        if not text.strip():
            return
        resampler = _Resampler(self.sample_rate) if self.sample_rate != OUTPUT_SAMPLE_RATE else None
        try:
            async with self._client.stream(
                "POST", "/audio/speech", json=self._request_body(text)
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise TTSError(self._status_message(response))
                content_type = response.headers.get("content-type", "")
                if content_type.startswith(("application/json", "text/")):
                    await response.aread()
                    raise TTSError(
                        f"{self._where()} sent {content_type!r} instead of audio for model "
                        f"{self.model!r}: {_error_detail(response)}"
                    )
                carry = b""
                async for data in response.aiter_bytes():
                    data = carry + data
                    whole = len(data) - len(data) % SAMPLE_WIDTH_BYTES
                    carry = data[whole:]
                    if whole:
                        pcm = data[:whole]
                        if resampler is not None:
                            pcm = resampler.feed(pcm)
                        if pcm:
                            yield AudioChunk(data=pcm, sample_rate=OUTPUT_SAMPLE_RATE)
                if resampler is not None and (tail := resampler.flush()):
                    yield AudioChunk(data=tail, sample_rate=OUTPUT_SAMPLE_RATE)
                if carry:
                    logger.debug("speech from %s ended with half a sample", self._where())
        except httpx.ConnectError as exc:
            raise TTSError(f"cannot reach {self._where()}; is it running? ({exc})") from exc
        except httpx.TimeoutException as exc:
            raise TTSError(
                f"{self._where()} did not respond within {self._timeout:g}s "
                f"(model {self.model!r}); it may still be loading the model"
            ) from exc
        except httpx.HTTPError as exc:
            raise TTSError(f"connection to {self._where()} failed: {exc!r}") from exc

    def _status_message(self, response: httpx.Response) -> str:
        detail = _error_detail(response)
        message = (
            f"{self._where()} returned HTTP {response.status_code} for speech model "
            f"{self.model!r}: {detail}"
        )
        if response.status_code == 404 and "not found" in detail.lower():
            message += f" (is the model installed? e.g. `ollama pull {self.model}`)"
        elif "voice" in detail.lower():
            message += " (check `voice` in the [tts] settings)"
        return message


@registry.register("tts", "ollama")
class OllamaTTS(OpenAICompatTTS):
    """`OpenAICompatTTS` defaulting to a local Ollama server."""

    server_name = "Ollama"

    def __init__(self, *, base_url: str = OLLAMA_BASE_URL, **options: Any) -> None:
        super().__init__(base_url=base_url, **options)


class _Resampler:
    """Streaming resampler from a server's PCM rate to the core output rate (soxr)."""

    def __init__(self, rate: int) -> None:
        import soxr

        self._stream = soxr.ResampleStream(rate, OUTPUT_SAMPLE_RATE, 1, dtype="int16")

    def feed(self, pcm: bytes) -> bytes:
        import numpy as np

        samples = np.frombuffer(pcm, dtype="<i2")
        return self._stream.resample_chunk(samples).astype("<i2").tobytes()

    def flush(self) -> bytes:
        import numpy as np

        return self._stream.resample_chunk(np.zeros(0, "<i2"), last=True).astype("<i2").tobytes()
