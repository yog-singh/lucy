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
import contextlib
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


# --- The system's default devices ---------------------------------------------------


class DefaultDevices(Protocol):
    """Where the system's current default devices are read from.

    PortAudio reads the defaults once, when it starts, and sends no
    notification when they change, so `SoundDeviceIO` polls this instead.
    """

    def current(self) -> tuple[int, int] | None:
        """(default input id, default output id) as the system numbers them, or
        None when they cannot be read (then the devices are not followed)."""
        ...


class CoreAudioDefaults:
    """The macOS default input and output device ids, read from CoreAudio with ctypes.

    The ids are CoreAudio's `AudioDeviceID`s (not PortAudio device numbers):
    they only tell that the default changed. `name(id)` gives a device's name.
    """

    _SYSTEM_OBJECT: Final = 1  # kAudioObjectSystemObject
    _DEFAULT_INPUT: Final = int.from_bytes(b"dIn ")  # kAudioHardwarePropertyDefaultInputDevice
    _DEFAULT_OUTPUT: Final = int.from_bytes(b"dOut")  # ...DefaultOutputDevice
    _RUN_LOOP: Final = int.from_bytes(b"rnlp")  # kAudioHardwarePropertyRunLoop
    _NAME: Final = int.from_bytes(b"lnam")  # kAudioObjectPropertyName
    _SCOPE_GLOBAL: Final = int.from_bytes(b"glob")
    _ELEMENT_MAIN: Final = 0

    def __init__(self) -> None:
        import ctypes

        class Address(ctypes.Structure):
            _fields_ = [
                ("selector", ctypes.c_uint32),
                ("scope", ctypes.c_uint32),
                ("element", ctypes.c_uint32),
            ]

        self._ctypes = ctypes
        self._address = Address
        audio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        self._get = audio.AudioObjectGetPropertyData
        self._get.restype = ctypes.c_int32
        self._get.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(Address),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self._cf_string = cf.CFStringGetCString
        self._cf_string.restype = ctypes.c_bool
        self._cf_string.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_uint32,
        ]
        self._cf_release = cf.CFRelease
        self._cf_release.argtypes = [ctypes.c_void_p]
        # Without a run loop of its own the HAL may not see device changes in a
        # process that runs none (a terminal program): NULL makes it use its own thread.
        set_data = audio.AudioObjectSetPropertyData
        set_data.restype = ctypes.c_int32
        set_data.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(Address),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        loop = ctypes.c_void_p(None)
        address = Address(self._RUN_LOOP, self._SCOPE_GLOBAL, self._ELEMENT_MAIN)
        set_data(
            self._SYSTEM_OBJECT,
            ctypes.byref(address),
            0,
            None,
            ctypes.sizeof(loop),
            ctypes.byref(loop),
        )

    def _device(self, selector: int) -> int | None:
        ctypes = self._ctypes
        address = self._address(selector, self._SCOPE_GLOBAL, self._ELEMENT_MAIN)
        value = ctypes.c_uint32(0)
        size = ctypes.c_uint32(ctypes.sizeof(value))
        status = self._get(
            self._SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size),
            ctypes.byref(value),
        )  # fmt: skip
        return int(value.value) if status == 0 else None

    def current(self) -> tuple[int, int] | None:
        devices = self._device(self._DEFAULT_INPUT), self._device(self._DEFAULT_OUTPUT)
        if devices[0] is None or devices[1] is None:
            return None
        return devices[0], devices[1]

    def name(self, device: int) -> str | None:
        """A device's name, or None if it cannot be read."""
        ctypes = self._ctypes
        address = self._address(self._NAME, self._SCOPE_GLOBAL, self._ELEMENT_MAIN)
        ref = ctypes.c_void_p(None)
        size = ctypes.c_uint32(ctypes.sizeof(ref))
        status = self._get(device, ctypes.byref(address), 0, None, ctypes.byref(size),
                           ctypes.byref(ref))  # fmt: skip
        if status != 0 or not ref.value:
            return None
        try:
            buffer = ctypes.create_string_buffer(512)
            if not self._cf_string(ref, buffer, len(buffer), 0x08000100):  # UTF-8
                return None
            return buffer.value.decode("utf-8", "replace")
        finally:
            self._cf_release(ref)


def system_default_devices() -> DefaultDevices | None:
    """The default-device source for this platform: CoreAudio on macOS, else None."""
    import sys

    if sys.platform != "darwin":
        return None
    try:
        source = CoreAudioDefaults()
    except (OSError, AttributeError) as exc:
        logger.info("cannot read the default sound devices from CoreAudio: %s", exc)
        return None
    return source if source.current() is not None else None


@dataclass(frozen=True, slots=True)
class _Remap:
    """Output queue positions after a device switch: positions from `at` on were
    re-resampled from `old_rate` to `new_rate`."""

    at: int
    old_rate: int
    new_rate: int

    def apply(self, mark: int) -> int:
        if mark < self.at or self.old_rate == self.new_rate:
            return mark
        return self.at + int(round((mark - self.at) * self.new_rate / self.old_rate))


class SoundDeviceIO:
    """The microphone and speaker through `sounddevice` (PortAudio).

    Both devices are opened at their native sample rate with callback streams
    when `start` is called, and stay open until `close` (output plays silence
    while idle, so playback starts without device start-up delay).

    Following the system devices: a side whose device is not set (None) follows
    the system default while running. `check_devices` (run every `poll_s`
    seconds) reads the defaults from `default_devices`; once a change has held
    for `settle_s` (Bluetooth headphones change output and input a moment
    apart), both streams are closed, PortAudio is restarted so it sees the new
    device list, and both are reopened at their native rates. The same reopen
    runs when a stream fails on its own (stopped, or no callback for `stall_s`
    seconds: a device unplugged before the poll noticed), retried with growing
    pauses if no device can be opened. A side set explicitly keeps its device.

    Audio queued but not yet played when the devices switch carries on on the
    new speaker, re-resampled if its rate differs; the few milliseconds already
    handed to the old device are lost. Waiting `play` calls and `frames()` carry
    on across the switch, and `played_samples` stays in output-format samples
    throughout. If no speaker can be opened, queued audio is dropped as if
    stopped, so the turn ends instead of waiting. The switch runs on the event
    loop, so it never interleaves with `play` or `_on_input`; it blocks the loop
    for as long as opening the devices takes (usually well under a second).
    `on_notice`, if set, gets one line per change ("sound: now using ...");
    otherwise these lines are logged.

    `default_devices` defaults to CoreAudio on macOS when `sd` is the real
    module, and to nothing (no following) when a fake `sd` is passed.

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
        default_devices: DefaultDevices | None | bool = True,
        poll_s: float | None = 1.0,
        settle_s: float = 1.0,
        stall_s: float = 3.0,
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        if default_devices is True:
            default_devices = system_default_devices() if sd is None else None
        self._defaults: DefaultDevices | None = default_devices or None
        self._poll_s = poll_s
        self._settle_s = settle_s
        self._stall_s = stall_s
        self.on_notice = on_notice
        self._seen: tuple[int, int] | None = None  # defaults the open streams were chosen by
        self._candidate: tuple[int, int] | None = None
        self._candidate_since = 0.0
        self._stream_epoch = 0  # which pair of streams callbacks belong to
        self._broken = False  # a stream stopped or failed on its own
        self._last_in_callback = self._last_out_callback = 0.0
        self._retry_at: float | None = None
        self._retry_pause = 0.0
        self._failing = False
        self._watcher: asyncio.Task[None] | None = None
        self._remaps: list[_Remap] = []
        self._played_base = 0  # output-format samples played on earlier speakers
        self._played_mark = 0  # `_played_frames` when the current speaker opened
        self._opened_at = 0.0
        self._announced = ("", "")  # (microphone, speaker) the user was last told of
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

        if self._sd is None:
            self._sd = _import_sounddevice()
        self._loop = asyncio.get_running_loop()
        self._pending = np.zeros(0, dtype=np.float32)
        self._seen = self._read_defaults()
        try:
            self._open_streams()
        except AudioDeviceError:
            await self.close()
            raise
        self._announced = (self._in_name, self._out_name)
        if self._poll_s is not None:
            self._watcher = asyncio.create_task(self._watch(), name="lucy-sound-devices")

    def _open_streams(self) -> None:
        """Open and start both streams on the wanted devices (the current defaults
        for those not set). Raises `AudioDeviceError`; nothing is left open then."""
        import soxr

        sd = self._sd
        in_info = self._device(self._want_input, "input")
        out_info = self._device(self._want_output, "output")
        in_name, out_name = str(in_info["name"]), str(out_info["name"])
        in_rate = int(in_info["default_samplerate"])
        out_rate = int(out_info["default_samplerate"])
        out_channels = min(2, int(out_info["max_output_channels"]))
        if self._out_rate and out_rate != self._out_rate:
            self._retarget_output(out_rate)
        self._out_rate = out_rate
        self._out_resampler = (
            soxr.ResampleStream(OUTPUT_SAMPLE_RATE, out_rate, 1, dtype="float32")
            if out_rate != OUTPUT_SAMPLE_RATE
            else None
        )
        self._in_rate = in_rate
        self._in_resampler = (
            soxr.ResampleStream(in_rate, INPUT_SAMPLE_RATE, 1, dtype="float32")
            if in_rate != INPUT_SAMPLE_RATE
            else None
        )
        self._in_name, self._out_name = in_name, out_name
        self._out_channels = out_channels
        self._stream_epoch += 1
        epoch = self._stream_epoch
        now = self._clock()
        self._last_in_callback = self._last_out_callback = self._opened_at = now
        self._broken = False

        def on_input(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            self._input_callback(indata, frames, time_info, status, epoch=epoch)

        def finished() -> None:
            self._stream_finished(epoch)

        try:
            self._output_stream = sd.OutputStream(
                device=out_info.get("index"),
                samplerate=out_rate,
                channels=out_channels,
                dtype="float32",
                latency="low",
                callback=self._output_callback,
                finished_callback=finished,
            )
            self._output_stream.start()
        except Exception as exc:
            self._close_streams()
            raise AudioDeviceError(f"cannot open the speaker {out_name!r}: {exc}") from exc
        try:
            self._input_stream = sd.InputStream(
                device=in_info.get("index"),
                samplerate=in_rate,
                channels=1,
                dtype="float32",
                callback=on_input,
                finished_callback=finished,
            )
            self._input_stream.start()
        except Exception as exc:
            self._close_streams()
            raise AudioDeviceError(
                f"cannot open the microphone {in_name!r}: {exc}. {MIC_PERMISSION_HINT}"
            ) from exc
        self._started_at = now
        self._heard_signal = self._warned_silent = False

    def _close_streams(self) -> None:
        """Stop and close both streams; callbacks still in flight are ignored."""
        streams = (self._input_stream, self._output_stream)
        self._input_stream = self._output_stream = None
        self._stream_epoch += 1
        for stream in streams:
            if stream is None:
                continue
            try:
                getattr(stream, "abort", stream.stop)()
            except Exception:
                logger.debug("stopping a sound stream failed", exc_info=True)
            try:
                stream.close()
            except Exception:
                logger.debug("closing a sound stream failed", exc_info=True)

    def _retarget_output(self, rate: int) -> None:
        """Re-resample the audio not yet played to the new speaker's `rate`, keeping
        `played_samples` and the queue positions waiting `play` calls hold right."""
        import numpy as np

        old = self._out_rate
        tail = None
        if self._out_resampler is not None:  # what the old resampler still holds
            tail = self._out_resampler.resample_chunk(np.zeros(0, np.float32), last=True)
        with self._lock:
            rest = list(self._out_queue)
            if rest:
                rest[0] = rest[0][self._out_offset :]
            if tail is not None and len(tail):
                rest.append(tail)
            pending = np.concatenate(rest) if rest else np.zeros(0, np.float32)
            if len(pending):
                pending = resample(pending, old, rate)
            played = self._played_frames
            self._played_base += int(round((played - self._played_mark) * OUTPUT_SAMPLE_RATE / old))
            self._played_mark = played
            self._remaps.append(_Remap(played, old, rate))
            self._out_queue = deque([pending] if len(pending) else [])
            self._out_offset = 0
            self._queued_frames = played + len(pending)
            self._out_rate = rate

    def _restart_portaudio(self) -> None:
        """Restart PortAudio so it reads the device list and the defaults again (it
        reads them only when it starts). Every stream of the process must be closed:
        `SoundDeviceIO` is the only user of `sounddevice` in Lucy."""
        sd = self._sd
        terminate = getattr(sd, "_terminate", None)
        initialize = getattr(sd, "_initialize", None)
        if terminate is None or initialize is None:
            return
        try:
            terminate()
        except Exception:
            logger.debug("stopping PortAudio failed", exc_info=True)
        try:
            initialize()
        except Exception as exc:
            raise AudioDeviceError(f"cannot restart the PortAudio sound library: {exc}") from exc

    def describe(self) -> str:
        return (
            f"microphone: {self._in_name} ({self._in_rate} Hz); "
            f"speaker: {self._out_name} ({self._out_rate} Hz)"
        )

    def now(self) -> float:
        return self._clock()

    # --- Following the devices ------------------------------------------------------

    def _tell(self, text: str) -> None:
        if self.on_notice is not None:
            try:
                self.on_notice(text)
                return
            except Exception:
                logger.debug("showing a sound notice failed", exc_info=True)
        logger.warning("%s", text)

    def _read_defaults(self) -> tuple[int, int] | None:
        if self._defaults is None:
            return None
        try:
            return self._defaults.current()
        except Exception:
            logger.debug("reading the default sound devices failed", exc_info=True)
            return None

    def _followed(self, defaults: tuple[int, int] | None) -> tuple[int | None, int | None]:
        """The parts of `defaults` that matter: those of the sides not set explicitly."""
        if defaults is None:
            return None, None
        return (
            defaults[0] if self._want_input is None else None,
            defaults[1] if self._want_output is None else None,
        )

    def _stream_finished(self, epoch: int) -> None:
        """PortAudio thread: a stream stopped. Only matters if we did not stop it."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._mark_broken, epoch)
        except RuntimeError:
            pass

    def _mark_broken(self, epoch: int) -> None:
        if epoch == self._stream_epoch and not self._closed:
            self._broken = True

    def _failed(self, now: float) -> str | None:
        """Why the open streams no longer work, or None."""
        if self._output_stream is None or self._input_stream is None:
            return "a sound device is not open"
        if self._broken:
            return "a sound stream stopped"
        for stream in (self._output_stream, self._input_stream):
            if not getattr(stream, "active", True):
                return "a sound stream stopped"
        if now - self._last_out_callback > self._stall_s:
            return "the speaker stopped responding"
        if now - self._last_in_callback > self._stall_s:
            return "the microphone stopped responding"
        return None

    def check_devices(self) -> None:
        """Reopen the devices if a followed default changed (and has settled) or a
        stream failed. Runs every `poll_s` seconds; tests call it directly."""
        if self._closed or self._loop is None:
            return
        now = self._clock()
        if self._retry_at is not None:
            if now >= self._retry_at:
                self._switch("retry")
            return
        problem = self._failed(now)
        if problem is not None:
            if now - self._opened_at < 5 * self._stall_s and self._retry_pause:
                self._schedule_retry(now)  # it failed again soon after reopening
                return
            logger.info("%s: reopening the sound devices", problem)
            self._switch("failed")
            return
        if now - self._opened_at > 5 * self._stall_s:
            self._retry_pause = 0.0  # healthy for a while
        current = self._read_defaults()
        if current is None or self._followed(current) == self._followed(self._seen):
            self._candidate = None
            return
        if self._candidate != current:
            self._candidate, self._candidate_since = current, now
            return
        if now - self._candidate_since >= self._settle_s:
            self._switch("changed")

    def _schedule_retry(self, now: float) -> None:
        self._retry_pause = min(30.0, max(1.0, 2 * self._retry_pause))
        self._retry_at = now + self._retry_pause

    def _switch(self, reason: str) -> None:
        """Close both streams, restart PortAudio and reopen on the current devices."""
        before = self._announced
        defaults = self._read_defaults()
        self._candidate = None
        self._close_streams()
        try:
            self._restart_portaudio()
            self._open_streams()
        except Exception as exc:
            self._close_streams()
            self._schedule_retry(self._clock())
            if not self._failing:
                self._failing = True
                self.stop_playback()  # nothing can play it: let the turn end
                self._tell(f"sound: cannot open the sound devices ({exc}); retrying")
            else:
                logger.debug("reopening the sound devices failed again: %s", exc)
            return
        self._seen = defaults
        self._announced = (self._in_name, self._out_name)
        self._retry_at = None
        recovered, self._failing = self._failing, False
        if reason == "changed":
            self._retry_pause = 0.0
        in_name, out_name = self._in_name, self._out_name
        changed = [
            (name, role)
            for name, old, role in (
                (out_name, before[1], "speaker"),
                (in_name, before[0], "microphone"),
            )
            if name != old
        ]
        if len(changed) == 2 and in_name == out_name:
            self._tell(f"sound: now using {out_name}")
        elif changed:
            self._tell("sound: now using " + ", ".join(f"{n} ({r})" for n, r in changed))
        elif recovered or reason == "failed":
            self._tell(f"sound: reopened {out_name} and {in_name}")
        logger.info("sound devices: %s", self.describe())

    async def _watch(self) -> None:
        assert self._poll_s is not None
        while not self._closed:
            await asyncio.sleep(self._poll_s)
            try:
                self.check_devices()
            except Exception:
                logger.exception("checking the sound devices failed")

    # --- Capture ----------------------------------------------------------------

    def _input_callback(
        self, indata: Any, frames: int, time_info: Any, status: Any, *, epoch: int | None = None
    ) -> None:
        """PortAudio thread: hand a copy of the block to the event loop."""
        if status:
            self._status_errors += 1
        captured_at = self._clock()
        self._last_in_callback = captured_at
        block = indata[:, 0].copy()
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._on_input, block, captured_at, epoch)
        except RuntimeError:  # the loop has closed
            pass

    def _on_input(self, block: Any, captured_at: float, epoch: int | None = None) -> None:
        """Event loop: resample, cut into frames, queue them."""
        import numpy as np

        if self._closed or (epoch is not None and epoch != self._stream_epoch):
            return  # from a stream closed since (its rate may differ)
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
        self._last_out_callback = self._clock()
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

    async def _played_past(self, mark: int, generation: int, remaps: int) -> bool:
        """Wait until the device has played beyond queue position `mark`, taken when
        `remaps` device switches had happened; False if stopped first."""
        while generation == self._generation and not self._closed:
            with self._lock:
                at = mark
                for remap in self._remaps[remaps:]:  # the speaker changed since
                    at = remap.apply(at)
                if self._played_frames > min(at, self._queued_frames - 1):
                    return True
            await self._progress()
        return False

    def _enqueue(self, samples: Any, generation: int) -> int | None:
        """Queue device-rate samples; returns the queue position they start at."""
        if not len(samples):
            return None
        with self._lock:
            if generation != self._generation or self._failing:
                return None  # stopped, or no speaker to play it on
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

        async def notify_start(mark: int, remaps: int) -> None:
            if await self._played_past(mark, generation, remaps) and on_start is not None:
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
                    watcher = asyncio.create_task(notify_start(first, len(self._remaps)))
            if generation != self._generation:
                return
            if self._out_resampler is not None:
                tail = self._out_resampler.resample_chunk(np.zeros(0, np.float32), last=True)
                start = self._enqueue(tail, generation)
                if first is None and start is not None:
                    first = start
                    watcher = asyncio.create_task(notify_start(first, len(self._remaps)))
                self._out_resampler.clear()
            with self._lock:
                end, remaps = self._queued_frames, len(self._remaps)
            if first is not None:
                await self._played_past(end - 1, generation, remaps)
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
        with self._lock:
            if not self._out_rate:
                return self._played_base
            played = self._played_frames - self._played_mark  # on the current speaker
            return self._played_base + int(round(played * OUTPUT_SAMPLE_RATE / self._out_rate))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
            self._watcher = None
        with self._lock:
            self._out_queue.clear()
            self._generation += 1
        self._close_streams()
        self._notify()
        self._frame_ready.set()
        if self._status_errors:
            logger.info("sound devices reported %d over/underflows", self._status_errors)
        if self._dropped_frames:
            logger.info("dropped %d unread capture frames", self._dropped_frames)
