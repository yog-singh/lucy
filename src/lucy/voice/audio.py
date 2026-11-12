"""Audio capture and playback for the voice client, behind one small interface (ADR 0014).

`AudioIO` is everything the voice session needs from the sound hardware:
fixed-size capture frames in the core input format, playback of core
output-format audio, an immediate stop, and a count of what has really been
played. Stage 2 (barge-in) will replace the two separate streams here with one
duplex stream for echo cancellation without changing the interface.

Two implementations:

- `SoundDeviceIO`: the real microphone and speaker through `sounddevice`
  (PortAudio), opened at their native rates; capture is resampled to 16 kHz
  and playback from 24 kHz with soxr.
- `FileAudioIO`: WAV files or in-memory audio as the "microphone" and memory or
  a WAV file as the "speaker", in real time or on a simulated clock. Tests,
  evals and `lucy voice --input-wav` use it.

Clocks: every `Frame.time`, `now()` and playback time of one `AudioIO` are on
the same clock (seconds; `time.monotonic` for the real devices and real-time
files, a simulated clock otherwise), so the session can compare them.

Capture never stops. A session that is busy (thinking or speaking) simply does
not read frames; they wait in the implementation's backlog and the session
discards those captured before it was ready to listen again.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import wave
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from lucy.core.types import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    AudioChunk,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

FRAME_SAMPLES: Final = 512
"""Samples per capture frame at 16 kHz: 32 ms, the frame size Silero VAD takes."""
FRAME_SECONDS: Final = FRAME_SAMPLES / INPUT_SAMPLE_RATE

MIC_PERMISSION_HINT: Final = (
    "On macOS, allow your terminal app under System Settings > Privacy & Security >"
    " Microphone, then restart the terminal."
)


class AudioDeviceError(RuntimeError):
    """A sound device is missing, refused or failed. The message says what to do."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One capture frame: `FRAME_SAMPLES` samples in the core input format.

    `time` is when its last sample was captured, on the `AudioIO`'s clock.
    """

    audio: AudioChunk
    time: float


@runtime_checkable
class AudioIO(Protocol):
    """The sound hardware as the voice session sees it."""

    def frames(self) -> AsyncIterator[Frame]:
        """Capture frames in order, for as long as input lasts. Call once.

        Frames captured while the caller is not reading are kept (up to an
        implementation-defined backlog) and delivered late, with their capture
        times.
        """
        ...

    async def play(
        self, chunks: AsyncIterable[AudioChunk], *, on_start: Callable[[], None] | None = None
    ) -> None:
        """Play core output-format chunks as they arrive; return once all have been played.

        `on_start` is called when the first sample is actually played. If
        `stop_playback` is called meanwhile, `play` returns early (at the latest
        when `chunks` next yields; cancel the caller to stop waiting on it).
        """
        ...

    def stop_playback(self) -> None:
        """Silence output now and discard queued audio."""
        ...

    @property
    def played_samples(self) -> int:
        """Total output-format samples (24 kHz) really played so far, never counting
        discarded audio. Compare two readings to see how much of a `play` was heard."""
        ...

    def now(self) -> float:
        """The current time on this `AudioIO`'s clock."""
        ...

    def describe(self) -> str:
        """The devices in use, for a banner."""
        ...

    async def close(self) -> None:
        """Stop capture and playback and release the devices. Idempotent."""
        ...


# --- Sample helpers -------------------------------------------------------------


def pcm_to_float(data: bytes) -> np.ndarray:
    """16-bit little-endian PCM to float32 samples in [-1, 1)."""
    import numpy as np

    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm(samples: np.ndarray) -> bytes:
    """Float samples to 16-bit little-endian PCM, clipping out-of-range values."""
    import numpy as np

    scaled = np.clip(np.asarray(samples, dtype=np.float32) * 32768.0, -32768, 32767)
    return np.round(scaled).astype("<i2").tobytes()


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample float32 mono audio in one go (soxr, high quality)."""
    import numpy as np

    if from_rate == to_rate:
        return np.asarray(samples, dtype=np.float32)
    import soxr

    return soxr.resample(np.asarray(samples, dtype=np.float32), from_rate, to_rate, "HQ")


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """A PCM WAV file as float32 mono samples (channels averaged) and its sample rate."""
    import numpy as np

    try:
        with wave.open(str(path), "rb") as wav:
            rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
            raw = wav.readframes(wav.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"cannot read WAV file {path}: {exc}") from exc
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: {8 * width}-bit WAV is not supported (use 16-bit PCM)")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def write_wav(path: str | Path, pcm: bytes, sample_rate: int) -> None:
    """Write mono 16-bit PCM to a WAV file, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def to_input_format(source: str | Path | AudioChunk) -> np.ndarray:
    """A WAV file or chunk as float32 mono samples at the core input rate (16 kHz)."""
    import numpy as np

    if isinstance(source, AudioChunk):
        samples = pcm_to_float(source.data)
        if source.channels > 1:
            samples = samples[: len(samples) - len(samples) % source.channels]
            samples = samples.reshape(-1, source.channels).mean(axis=1)
        rate = source.sample_rate
    else:
        samples, rate = read_wav(source)
    return np.asarray(resample(samples, rate, INPUT_SAMPLE_RATE), dtype=np.float32)


def _check_output_chunk(chunk: AudioChunk) -> None:
    if chunk.sample_rate != OUTPUT_SAMPLE_RATE or chunk.channels != 1:
        raise ValueError(
            f"playback takes {OUTPUT_SAMPLE_RATE} Hz mono audio (the core output format), "
            f"got {chunk.sample_rate} Hz with {chunk.channels} channel(s)"
        )


# --- Files ----------------------------------------------------------------------------


@dataclass
class PlayRecord:
    """One `play` call of a `FileAudioIO`: when it started (on its clock), how many
    samples were played, and whether it was stopped."""

    start: float
    samples: int = 0
    stopped: bool = False


class FileAudioIO:
    """Audio files as the microphone, memory (and optionally a WAV file) as the speaker.

    `inputs` (WAV paths or `AudioChunk`s, any rate, mono or stereo) are
    converted to 16 kHz and played as one input stream in order, cut into
    frames; the last frame is padded with silence.

    `realtime=True` releases each frame when the wall clock reaches its capture
    time and plays audio at real speed, like the real devices. Otherwise time is
    simulated and nothing waits: a frame is released at once when nothing is
    playing (time jumps to it), and while audio plays, time advances by the
    audio played (in `play_step_ms` steps) and releases the frames captured
    meanwhile. Thinking therefore takes no simulated time and speaking takes
    exactly its audio's length, which makes sessions deterministic in tests.

    `wait_for_reply=True` treats each input as one user turn: after an input
    ends, silence is captured until a reply has been played and `reply_idle_s`
    has passed since it ended (or `reply_timeout_s` passed with no reply), and
    only then does the next input start. Without it the inputs follow each
    other directly, as a person who does not wait would.

    Output: `output` holds everything played, placed on the input's timeline
    (silence where nothing played), at 24 kHz mono 16-bit; `output_path`, if
    given, receives it as a WAV file on `close`. `plays` records each `play`.
    """

    def __init__(
        self,
        inputs: Sequence[str | Path | AudioChunk],
        *,
        realtime: bool = False,
        output_path: str | Path | None = None,
        wait_for_reply: bool = False,
        reply_idle_s: float = 1.0,
        reply_timeout_s: float = 30.0,
        play_step_ms: int = 10,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._segments = [to_input_format(source) for source in inputs]
        self._names = [Path(s).name if not isinstance(s, AudioChunk) else "audio" for s in inputs]
        self._realtime = realtime
        self._output_path = Path(output_path) if output_path is not None else None
        self._wait_for_reply = wait_for_reply
        self._reply_idle = reply_idle_s
        self._reply_timeout = reply_timeout_s
        self._step = max(1, OUTPUT_SAMPLE_RATE * play_step_ms // 1000)
        self._clock = clock
        self._origin: float | None = None
        self._sim = 0.0  # simulated time (timeline seconds)
        self._advanced = asyncio.Event()
        self._playing = False
        self._generation = 0
        self._played = 0
        self._plays_started = 0
        self._last_play_end = 0.0  # timeline seconds
        self._playhead = 0.0  # real-time playback position (timeline seconds)
        self.output = bytearray()
        self.plays: list[PlayRecord] = []
        self.input_done = False
        self._closed = False

    # --- Clock ------------------------------------------------------------------

    def _timeline(self) -> float:
        """Seconds since the input started."""
        if not self._realtime:
            return self._sim
        if self._origin is None:
            self._origin = self._clock()
        return self._clock() - self._origin

    def _to_clock(self, timeline: float) -> float:
        if not self._realtime:
            return timeline
        if self._origin is None:
            self._origin = self._clock()
        return self._origin + timeline

    def now(self) -> float:
        return self._to_clock(self._timeline())

    @property
    def input_seconds(self) -> float:
        """Total length of the inputs."""
        return sum(len(s) for s in self._segments) / INPUT_SAMPLE_RATE

    def describe(self) -> str:
        names = ", ".join(self._names) or "nothing"
        pace = "real time" if self._realtime else "simulated time"
        out = str(self._output_path) if self._output_path else "memory"
        return f"input: {names} ({pace}); output: {out}"

    # --- Capture ----------------------------------------------------------------

    async def _release(self, end: float) -> None:
        """Wait until the frame ending at timeline time `end` has been captured."""
        if self._realtime:
            delay = end - self._timeline()
            if delay > 0:
                await asyncio.sleep(delay)
            return
        while end > self._sim + 1e-9:
            if not self._playing:
                self._sim = end
                break
            self._advanced.clear()
            await self._advanced.wait()
        await asyncio.sleep(0)  # let others run, as real capture would

    def _reply_done(self, pos: float, segment_end: float, plays_before: int) -> bool:
        if self._playing:
            return False
        if self._plays_started > plays_before:
            return pos >= self._last_play_end + self._reply_idle
        return pos >= segment_end + self._reply_timeout

    async def frames(self) -> AsyncIterator[Frame]:
        import numpy as np

        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        silent_pcm = float_to_pcm(silence)
        pos = 0.0  # timeline time at the end of the last frame
        plays_before = 0
        for index, segment in enumerate(self._segments):
            if index and self._wait_for_reply:
                segment_end = pos
                while not self._closed and not self._reply_done(pos, segment_end, plays_before):
                    pos += FRAME_SECONDS
                    await self._release(pos)
                    chunk = AudioChunk(data=silent_pcm, sample_rate=INPUT_SAMPLE_RATE)
                    yield Frame(chunk, self._to_clock(pos))
            for start in range(0, len(segment), FRAME_SAMPLES):
                if self._closed:
                    return
                samples = segment[start : start + FRAME_SAMPLES]
                if len(samples) < FRAME_SAMPLES:
                    samples = np.concatenate([samples, silence[: FRAME_SAMPLES - len(samples)]])
                pos += FRAME_SECONDS
                await self._release(pos)
                if start + FRAME_SAMPLES >= len(segment):
                    plays_before = self._plays_started  # replies after this answer the segment
                chunk = AudioChunk(data=float_to_pcm(samples), sample_rate=INPUT_SAMPLE_RATE)
                yield Frame(chunk, self._to_clock(pos))
        self.input_done = True

    # --- Playback ---------------------------------------------------------------

    def _write(self, at: float, pcm: bytes) -> None:
        """Place `pcm` on the output timeline at timeline time `at`."""
        position = int(round(at * OUTPUT_SAMPLE_RATE)) * SAMPLE_WIDTH_BYTES
        if position > len(self.output):
            self.output.extend(bytes(position - len(self.output)))
        self.output[position : position + len(pcm)] = pcm

    async def play(
        self, chunks: AsyncIterable[AudioChunk], *, on_start: Callable[[], None] | None = None
    ) -> None:
        generation = self._generation
        record = PlayRecord(start=self.now())
        self.plays.append(record)
        self._plays_started += 1
        self._playing = True
        started = False
        try:
            async for chunk in chunks:
                if generation != self._generation:
                    break
                _check_output_chunk(chunk)
                total = chunk.frames
                offset = 0
                while offset < total:
                    if generation != self._generation:
                        break
                    step = min(self._step, total - offset)
                    pcm = chunk.data[offset * 2 : (offset + step) * 2]
                    if self._realtime:
                        self._playhead = max(self._playhead, self._timeline())
                        at = self._playhead
                    else:
                        at = self._sim
                    if not started:
                        started = True
                        record.start = self._to_clock(at)
                        if on_start is not None:
                            on_start()
                    self._write(at, pcm)
                    if self._realtime:
                        self._playhead = at + step / OUTPUT_SAMPLE_RATE
                        delay = self._playhead - self._timeline()
                        if delay > 0:
                            await asyncio.sleep(delay)
                    else:
                        self._sim = at + step / OUTPUT_SAMPLE_RATE
                        self._advanced.set()
                        await asyncio.sleep(0)
                    if generation != self._generation:
                        break  # stopped while this step was "playing": it was not heard in full
                    self._played += step
                    record.samples += step
                    offset += step
                if generation != self._generation:
                    break
        finally:
            record.stopped = generation != self._generation
            self._playing = False
            self._last_play_end = self._timeline()
            self._advanced.set()

    def stop_playback(self) -> None:
        self._generation += 1
        self._advanced.set()

    @property
    def played_samples(self) -> int:
        return self._played

    @property
    def output_chunk(self) -> AudioChunk:
        """Everything played so far, as one chunk on the input's timeline."""
        return AudioChunk(data=bytes(self.output), sample_rate=OUTPUT_SAMPLE_RATE)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_playback()
        if self._output_path is not None:
            write_wav(self._output_path, bytes(self.output), OUTPUT_SAMPLE_RATE)


# --- Real devices -------------------------------------------------------------------


def _import_sounddevice() -> Any:
    try:
        import sounddevice
    except OSError as exc:  # PortAudio library missing
        raise AudioDeviceError(f"cannot load the PortAudio sound library: {exc}") from exc
    return sounddevice


def list_devices(sd: Any = None) -> str:
    """The input and output devices, one per line, with the defaults marked."""
    sd = sd if sd is not None else _import_sounddevice()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise AudioDeviceError(f"cannot list sound devices: {exc}") from exc
    try:
        default_in, default_out = sd.default.device
    except Exception:
        default_in = default_out = None
    lines = ["Input devices (microphones):"]
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            mark = " (default)" if index == default_in else ""
            lines.append(
                f"  {index}: {device['name']}, {device['max_input_channels']} ch,"
                f" {device['default_samplerate']:g} Hz{mark}"
            )
    lines.append("Output devices (speakers):")
    for index, device in enumerate(devices):
        if device["max_output_channels"] > 0:
            mark = " (default)" if index == default_out else ""
            lines.append(
                f"  {index}: {device['name']}, {device['max_output_channels']} ch,"
                f" {device['default_samplerate']:g} Hz{mark}"
            )
    return "\n".join(lines)


class SoundDeviceIO:
    """The microphone and speaker through `sounddevice` (PortAudio).

    Both devices are opened at their native sample rate with callback streams
    when `start` is called, and stay open until `close` (output plays silence
    while idle, so playback starts without device start-up delay).

    Threading: PortAudio calls the callbacks on its own threads. The input
    callback copies each block and hands it to the event loop with
    `call_soon_threadsafe`; resampling to 16 kHz (soxr, streaming) and cutting
    into frames happen on the loop. Playback audio is resampled on the loop and
    appended under a lock to a queue that the output callback drains; the
    callback counts what it really played and wakes waiting `play` calls
    through the loop. Captured frames not yet read are kept up to `backlog_s`
    seconds (the oldest are dropped beyond that).

    `input_device` / `output_device` are a device number, part of a device
    name, or None for the system default. `sd` is the `sounddevice` module
    (tests pass a fake).
    """

    def __init__(
        self,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        backlog_s: float = 60.0,
        silent_input_warning_s: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
        sd: Any = None,
    ) -> None:
        self._sd = sd
        self._want_input = input_device
        self._want_output = output_device
        self._clock = clock
        self._backlog = max(1, int(backlog_s / FRAME_SECONDS))
        self._silent_warning_s = silent_input_warning_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._input_stream: Any = None
        self._output_stream: Any = None
        self._in_name = self._out_name = ""
        self._in_rate = self._out_rate = 0
        self._out_channels = 1
        self._in_resampler: Any = None
        self._out_resampler: Any = None
        self._pending: Any = None
        self._frames: deque[Frame] = deque()
        self._frame_ready = asyncio.Event()
        self._dropped_frames = 0
        self._started_at = 0.0
        self._heard_signal = False
        self._warned_silent = False
        # Output state shared with the PortAudio thread, under `_lock`.
        self._lock = threading.Lock()
        self._out_queue: deque[Any] = deque()
        self._out_offset = 0
        self._queued_frames = 0
        self._played_frames = 0
        self._generation = 0
        self._waiters: list[asyncio.Future[None]] = []
        self._status_errors = 0
        self._closed = False

    # --- Setup ------------------------------------------------------------------

    def _device(self, wanted: int | str | None, kind: str) -> dict[str, Any]:
        sd = self._sd
        channels_key = f"max_{kind}_channels"
        what = "microphone" if kind == "input" else "speaker"
        try:
            if wanted is None:
                info = dict(sd.query_devices(kind=kind))
            elif isinstance(wanted, int):
                info = dict(sd.query_devices(wanted))
            else:
                matches = [
                    dict(d, index=i)
                    for i, d in enumerate(sd.query_devices())
                    if wanted.lower() in str(d["name"]).lower() and d[channels_key] > 0
                ]
                if not matches:
                    raise AudioDeviceError(
                        f"no {kind} device matches {wanted!r}; see `lucy voice --list-devices`"
                    )
                info = matches[0]
        except AudioDeviceError:
            raise
        except Exception as exc:
            if wanted is None:
                hint = f" {MIC_PERMISSION_HINT}" if kind == "input" else ""
                raise AudioDeviceError(f"no {what} found ({exc}).{hint}") from exc
            raise AudioDeviceError(
                f"{kind} device {wanted!r} not found ({exc}); see `lucy voice --list-devices`"
            ) from exc
        if info.get(channels_key, 0) < 1:
            raise AudioDeviceError(
                f"device {info.get('name', wanted)!r} has no {kind} channels; "
                "see `lucy voice --list-devices`"
            )
        return info

    async def start(self) -> None:
        """Open and start both devices. Raises `AudioDeviceError` with what to do."""
        import numpy as np
        import soxr

        if self._sd is None:
            self._sd = _import_sounddevice()
        self._loop = asyncio.get_running_loop()
        in_info = self._device(self._want_input, "input")
        out_info = self._device(self._want_output, "output")
        self._in_name, self._out_name = str(in_info["name"]), str(out_info["name"])
        self._in_rate = int(in_info["default_samplerate"])
        self._out_rate = int(out_info["default_samplerate"])
        self._out_channels = min(2, int(out_info["max_output_channels"]))
        if self._in_rate != INPUT_SAMPLE_RATE:
            self._in_resampler = soxr.ResampleStream(
                self._in_rate, INPUT_SAMPLE_RATE, 1, dtype="float32"
            )
        if self._out_rate != OUTPUT_SAMPLE_RATE:
            self._out_resampler = soxr.ResampleStream(
                OUTPUT_SAMPLE_RATE, self._out_rate, 1, dtype="float32"
            )
        self._pending = np.zeros(0, dtype=np.float32)
        sd = self._sd
        try:
            self._output_stream = sd.OutputStream(
                device=out_info.get("index"),
                samplerate=self._out_rate,
                channels=self._out_channels,
                dtype="float32",
                latency="low",
                callback=self._output_callback,
            )
            self._output_stream.start()
        except Exception as exc:
            await self.close()
            raise AudioDeviceError(f"cannot open the speaker {self._out_name!r}: {exc}") from exc
        try:
            self._input_stream = sd.InputStream(
                device=in_info.get("index"),
                samplerate=self._in_rate,
                channels=1,
                dtype="float32",
                callback=self._input_callback,
            )
            self._input_stream.start()
        except Exception as exc:
            await self.close()
            raise AudioDeviceError(
                f"cannot open the microphone {self._in_name!r}: {exc}. {MIC_PERMISSION_HINT}"
            ) from exc
        self._started_at = self._clock()

    def describe(self) -> str:
        return (
            f"microphone: {self._in_name} ({self._in_rate} Hz); "
            f"speaker: {self._out_name} ({self._out_rate} Hz)"
        )

    def now(self) -> float:
        return self._clock()

    # --- Capture ----------------------------------------------------------------

    def _input_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio thread: hand a copy of the block to the event loop."""
        if status:
            self._status_errors += 1
        block = indata[:, 0].copy()
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._on_input, block, self._clock())
        except RuntimeError:  # the loop has closed
            pass

    def _on_input(self, block: Any, captured_at: float) -> None:
        """Event loop: resample, cut into frames, queue them."""
        import numpy as np

        if self._closed:
            return
        if not self._heard_signal and block.size and float(np.max(np.abs(block))) > 1e-4:
            self._heard_signal = True
        if (
            not self._heard_signal
            and not self._warned_silent
            and captured_at - self._started_at >= self._silent_warning_s
        ):
            self._warned_silent = True
            logger.warning(
                "the microphone %r delivers only silence. %s", self._in_name, MIC_PERMISSION_HINT
            )
        if self._in_resampler is not None:
            block = self._in_resampler.resample_chunk(block)
        pending = np.concatenate([self._pending, block]) if len(self._pending) else block
        count = len(pending) // FRAME_SAMPLES
        for k in range(count):
            samples = pending[k * FRAME_SAMPLES : (k + 1) * FRAME_SAMPLES]
            # The last frame ends at the newest sample; earlier ones are a frame apart.
            at = captured_at - (count - 1 - k) * FRAME_SECONDS
            frame = Frame(AudioChunk(data=float_to_pcm(samples), sample_rate=16000), at)
            if len(self._frames) >= self._backlog:
                self._frames.popleft()
                self._dropped_frames += 1
            self._frames.append(frame)
        self._pending = pending[count * FRAME_SAMPLES :].copy()
        if count:
            self._frame_ready.set()

    async def frames(self) -> AsyncIterator[Frame]:
        while True:
            while not self._frames:
                if self._closed:
                    return
                self._frame_ready.clear()
                await self._frame_ready.wait()
            yield self._frames.popleft()

    # --- Playback ---------------------------------------------------------------

    def _output_callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio thread: play queued audio, silence when there is none."""
        if status:
            self._status_errors += 1
        filled = 0
        with self._lock:
            while filled < frames and self._out_queue:
                buffer = self._out_queue[0]
                take = min(len(buffer) - self._out_offset, frames - filled)
                outdata[filled : filled + take, :] = buffer[
                    self._out_offset : self._out_offset + take, None
                ]
                filled += take
                self._out_offset += take
                if self._out_offset >= len(buffer):
                    self._out_queue.popleft()
                    self._out_offset = 0
            self._played_frames += filled
        if filled < frames:
            outdata[filled:, :] = 0
        if filled and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._notify)
            except RuntimeError:
                pass

    def _notify(self) -> None:
        waiters, self._waiters = self._waiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    async def _progress(self) -> None:
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        await waiter

    async def _played_past(self, mark: int, generation: int) -> bool:
        """Wait until the device has played beyond frame `mark`; False if stopped first."""
        while generation == self._generation and not self._closed:
            with self._lock:
                if self._played_frames > mark:
                    return True
            await self._progress()
        return False

    def _enqueue(self, samples: Any, generation: int) -> int | None:
        """Queue device-rate samples; returns the queue position they start at."""
        if not len(samples):
            return None
        with self._lock:
            if generation != self._generation:
                return None
            start = self._queued_frames
            self._out_queue.append(samples)
            self._queued_frames += len(samples)
        return start

    async def play(
        self, chunks: AsyncIterable[AudioChunk], *, on_start: Callable[[], None] | None = None
    ) -> None:
        import numpy as np

        generation = self._generation
        first: int | None = None
        watcher: asyncio.Task[None] | None = None

        async def notify_start(mark: int) -> None:
            if await self._played_past(mark, generation) and on_start is not None:
                on_start()

        try:
            async for chunk in chunks:
                if generation != self._generation or self._closed:
                    return
                _check_output_chunk(chunk)
                samples = pcm_to_float(chunk.data)
                if self._out_resampler is not None:
                    samples = self._out_resampler.resample_chunk(samples)
                start = self._enqueue(samples, generation)
                if first is None and start is not None:
                    first = start
                    watcher = asyncio.create_task(notify_start(first))
            if generation != self._generation:
                return
            if self._out_resampler is not None:
                tail = self._out_resampler.resample_chunk(np.zeros(0, np.float32), last=True)
                start = self._enqueue(tail, generation)
                if first is None and start is not None:
                    first = start
                    watcher = asyncio.create_task(notify_start(first))
                self._out_resampler.clear()
            with self._lock:
                end = self._queued_frames
            if first is not None:
                await self._played_past(end - 1, generation)
            if watcher is not None:
                await watcher
        finally:
            if watcher is not None and not watcher.done():
                watcher.cancel()
            if generation != self._generation and self._out_resampler is not None:
                self._out_resampler.clear()

    def stop_playback(self) -> None:
        with self._lock:
            self._out_queue.clear()
            self._out_offset = 0
            self._queued_frames = self._played_frames
            self._generation += 1
        if self._out_resampler is not None:
            self._out_resampler.clear()
        self._notify()

    @property
    def played_samples(self) -> int:
        if not self._out_rate:
            return 0
        with self._lock:
            played = self._played_frames
        return int(round(played * OUTPUT_SAMPLE_RATE / self._out_rate))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._out_queue.clear()
            self._generation += 1
        for stream in (self._input_stream, self._output_stream):
            if stream is None:
                continue
            try:
                stream.stop()
            except Exception:
                logger.debug("stopping a sound stream failed", exc_info=True)
            try:
                stream.close()
            except Exception:
                logger.debug("closing a sound stream failed", exc_info=True)
        self._input_stream = self._output_stream = None
        self._notify()
        self._frame_ready.set()
        if self._status_errors:
            logger.info("sound devices reported %d over/underflows", self._status_errors)
        if self._dropped_frames:
            logger.info("dropped %d unread capture frames", self._dropped_frames)
