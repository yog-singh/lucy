"""The OpenAI-compatible TTS adapter, over httpx.MockTransport (no server)."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import httpx
import numpy as np
import pytest

from lucy.adapters.tts_openai_compat import (
    OLLAMA_BASE_URL,
    OllamaTTS,
    OpenAICompatTTS,
    TTSError,
)
from lucy.core import registry
from lucy.core.types import OUTPUT_SAMPLE_RATE

PCM = bytes(range(256)) * 40  # 10240 bytes = 5120 samples


class Server:
    """A MockTransport handler recording requests and streaming `body` in `pieces`."""

    def __init__(
        self,
        body: bytes = PCM,
        *,
        pieces: list[int] | None = None,
        status: int = 200,
        content_type: str = "audio/pcm",
        error: Exception | None = None,
    ) -> None:
        self.body = body
        self.pieces = pieces or [len(body)]
        self.status = status
        self.content_type = content_type
        self.error = error
        self.requests: list[httpx.Request] = []
        self.sent = 0
        self.closed = asyncio.Event()

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]

    def stream(self) -> httpx.AsyncByteStream:
        server = self

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                position = 0
                for size in server.pieces:
                    yield server.body[position : position + size]
                    server.sent += 1
                    position += size
                    await asyncio.sleep(0)
                if position < len(server.body):
                    yield server.body[position:]

            async def aclose(self) -> None:
                server.closed.set()

        return Body()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.status != 200 or self.content_type.startswith("application/json"):
            return httpx.Response(
                self.status, headers={"content-type": self.content_type}, content=self.body
            )
        return httpx.Response(
            self.status, headers={"content-type": self.content_type}, stream=self.stream()
        )


def make(server: Callable, **options) -> OpenAICompatTTS:
    options.setdefault("model", "phonon")
    return OpenAICompatTTS(
        base_url="http://speech.test/v1", transport=httpx.MockTransport(server), **options
    )


async def synth(tts: OpenAICompatTTS, text: str = "Hello there.") -> list:
    return [chunk async for chunk in tts.synthesize(text)]


async def test_request_asks_for_raw_pcm() -> None:
    server = Server()
    tts = make(server, voice="default", api_key="k")
    await synth(tts)
    (request,) = server.requests
    assert request.method == "POST"
    assert str(request.url) == "http://speech.test/v1/audio/speech"
    assert request.headers["authorization"] == "Bearer k"
    assert server.bodies() == [
        {"model": "phonon", "input": "Hello there.", "response_format": "pcm", "voice": "default"}
    ]
    assert tts.capabilities.streaming
    await tts.aclose()


async def test_voice_and_speed_are_sent_only_when_set() -> None:
    server = Server()
    await synth(make(server))
    await synth(make(server, speed=1.25))
    assert "voice" not in server.bodies()[0] and "speed" not in server.bodies()[0]
    assert server.bodies()[1]["speed"] == 1.25


async def test_chunks_follow_arrival_and_keep_whole_samples() -> None:
    server = Server(pieces=[3, 5, 1, 7, 1000, 2001])
    chunks = await synth(make(server))
    assert len(chunks) >= 5  # yielded as bytes arrive, not at the end
    assert all(len(c.data) % 2 == 0 and c.data for c in chunks)
    assert all(c.sample_rate == OUTPUT_SAMPLE_RATE and c.channels == 1 for c in chunks)
    assert b"".join(c.data for c in chunks) == PCM


async def test_a_trailing_half_sample_is_dropped() -> None:
    chunks = await synth(make(Server(body=PCM + b"\x01")))
    assert b"".join(c.data for c in chunks) == PCM


async def test_blank_text_makes_no_request() -> None:
    server = Server()
    assert await synth(make(server), "   ") == []
    assert server.requests == []


async def test_closing_early_closes_the_response() -> None:
    server = Server(pieces=[100] * 100)
    stream = make(server).synthesize("A long sentence.")
    first = await anext(stream)
    assert first.data
    await stream.aclose()
    await asyncio.wait_for(server.closed.wait(), 1)
    assert server.sent < 10


@pytest.mark.parametrize(
    ("status", "detail", "expected"),
    [
        (404, 'model "phonon" not found, try pulling it first', "ollama pull phonon"),
        (400, 'voice "Nobody" not found; available voices: default', "[tts] settings"),
        (400, "only speed 1 is supported", "only speed 1"),
    ],
)
async def test_http_errors_are_actionable(status: int, detail: str, expected: str) -> None:
    body = json.dumps({"error": {"message": detail}}).encode()
    server = Server(body=body, status=status, content_type="application/json")
    with pytest.raises(TTSError) as info:
        await synth(make(server))
    message = str(info.value)
    assert f"HTTP {status}" in message and "'phonon'" in message and expected in message


async def test_a_json_body_with_status_200_is_an_error() -> None:
    body = json.dumps({"error": "busy"}).encode()
    server = Server(body=body, content_type="application/json")
    with pytest.raises(TTSError, match="instead of audio.*busy"):
        await synth(make(server))


async def test_unreachable_server() -> None:
    server = Server(error=httpx.ConnectError("refused"))
    with pytest.raises(TTSError, match="cannot reach the speech server at http://speech.test/v1"):
        await synth(make(server))


async def test_timeout() -> None:
    server = Server(error=httpx.ReadTimeout("slow"))
    with pytest.raises(TTSError, match="did not respond within 7s"):
        await synth(make(server, timeout=7))


async def test_closed_adapter_refuses() -> None:
    tts = make(Server())
    await tts.aclose()
    with pytest.raises(TTSError, match="closed"):
        await synth(tts)


async def test_another_server_rate_is_resampled_to_the_core_format() -> None:
    rate = 22050
    t = np.arange(rate) / rate
    pcm = (0.5 * np.sin(2 * np.pi * 300 * t) * 32767).astype("<i2").tobytes()
    server = Server(body=pcm, pieces=[4410] * 10)
    chunks = await synth(make(server, sample_rate=rate))
    total = sum(c.frames for c in chunks)
    assert total == pytest.approx(OUTPUT_SAMPLE_RATE, abs=50)
    assert all(c.sample_rate == OUTPUT_SAMPLE_RATE for c in chunks)


async def test_warm_up_synthesises_a_word() -> None:
    server = Server()
    await make(server).warm_up()
    assert server.bodies()[0]["input"] == "Ready."


async def test_ollama_defaults_and_registration() -> None:
    tts = OllamaTTS(model="phonon")
    assert tts.base_url == OLLAMA_BASE_URL
    await tts.aclose()
    assert registry.get("tts", "ollama") is OllamaTTS
    assert registry.get("tts", "openai_compat") is OpenAICompatTTS
    with pytest.raises(TypeError):
        OpenAICompatTTS(model="x")  # base_url is required
