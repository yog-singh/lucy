"""Audio I/O: `FileAudioIO` on simulated and real time, `SoundDeviceIO` against a fake
`sounddevice` module (no real device is ever opened)."""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lucy.core.types import OUTPUT_SAMPLE_RATE, AudioChunk
from lucy.voice.audio import (
    FRAME_SAMPLES,
    FRAME_SECONDS,
    AudioDeviceError,
    AudioIO,
    FileAudioIO,
    SoundDeviceIO,
    list_devices,
    pcm_to_float,
    read_wav,
    write_wav,
)
from tests.voice_fakes import concat, silence, tone

TIMEOUT = 5


async def collect(iterator, n: int | None = None) -> list:
    frames = []
    async with asyncio.timeout(TIMEOUT):
        async for frame in iterator:
            frames.append(frame)
            if n is not None and len(frames) >= n:
                break
    return frames


async def chunks_of(*chunks: AudioChunk):
    for chunk in chunks:
        yield chunk


def dominant_frequency(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.argmax(spectrum) * rate / len(samples))


# --- FileAudioIO --------------------------------------------------------------


def test_both_implementations_satisfy_the_protocol() -> None:
    assert isinstance(FileAudioIO([]), AudioIO)
    assert isinstance(SoundDeviceIO(sd=FakeSD()), AudioIO)


async def test_file_frames_are_fixed_size_and_timed() -> None:
    io = FileAudioIO([tone(1.0)])
    frames = await collect(io.frames())
    assert len(frames) == 32  # 16000 / 512 = 31.25, the last frame padded
    assert all(len(f.audio.data) == 2 * FRAME_SAMPLES for f in frames)
    assert all(f.audio.sample_rate == 16000 for f in frames)
    assert [f.time for f in frames[:3]] == pytest.approx([FRAME_SECONDS * k for k in (1, 2, 3)])
    assert io.input_done
    assert io.now() == pytest.approx(32 * FRAME_SECONDS)


async def test_file_input_is_resampled_and_downmixed(tmp_path) -> None:
    stereo_48k = tone(1.0, rate=48000)
    mono = np.frombuffer(stereo_48k.data, "<i2")
    path = tmp_path / "stereo.wav"
    import wave

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(np.repeat(mono, 2).astype("<i2").tobytes())
    io = FileAudioIO([path])
    frames = await collect(io.frames())
    samples = np.concatenate([pcm_to_float(f.audio.data) for f in frames])[:16000]
    assert dominant_frequency(samples, 16000) == pytest.approx(440, abs=5)
    assert len(frames) == 32
    assert "stereo.wav" in io.describe()


async def test_simulated_capture_waits_for_playback_time() -> None:
    io = FileAudioIO([silence(3.0)])
    frames = io.frames()
    first = await collect(frames, 10)
    assert first[-1].time == pytest.approx(0.32)
    seen: list[tuple[float, float]] = []

    async def reader() -> None:
        async for frame in frames:
            seen.append((frame.time, io.now()))
            if len(seen) >= 20:
                return

    play = asyncio.create_task(io.play(chunks_of(tone(1.0, rate=OUTPUT_SAMPLE_RATE))))
    await asyncio.sleep(0)
    task = asyncio.create_task(reader())
    async with asyncio.timeout(TIMEOUT):
        await asyncio.gather(play, task)
    # Every frame read during playback had been captured by then.
    assert all(at <= now + 1e-9 for at, now in seen)
    assert io.plays[0].start == pytest.approx(0.32)
    assert io.played_samples == OUTPUT_SAMPLE_RATE


async def test_simulated_output_lies_on_the_input_timeline(tmp_path) -> None:
    out = tmp_path / "out" / "lucy.wav"
    io = FileAudioIO([silence(2.0)], output_path=out)
    await collect(io.frames(), 31)  # about 1 s
    started: list[float] = []
    reply = tone(0.5, rate=OUTPUT_SAMPLE_RATE)
    await io.play(chunks_of(reply), on_start=lambda: started.append(io.now()))
    assert started == [pytest.approx(31 * FRAME_SECONDS)]
    await io.close()
    samples, rate = read_wav(out)
    assert rate == OUTPUT_SAMPLE_RATE
    onset = int(np.argmax(np.abs(samples) > 0.01))
    assert onset / rate == pytest.approx(31 * FRAME_SECONDS, abs=0.002)
    assert len(samples) == pytest.approx((31 * FRAME_SECONDS + 0.5) * rate, abs=2)


async def test_stop_playback_cuts_a_simulated_play_short() -> None:
    io = FileAudioIO([])
    task = asyncio.create_task(io.play(chunks_of(tone(2.0, rate=OUTPUT_SAMPLE_RATE))))
    async with asyncio.timeout(TIMEOUT):
        while io.played_samples < OUTPUT_SAMPLE_RATE // 2:
            await asyncio.sleep(0)
    io.stop_playback()
    await asyncio.wait_for(task, TIMEOUT)
    assert io.plays[0].stopped
    assert OUTPUT_SAMPLE_RATE // 2 <= io.played_samples < OUTPUT_SAMPLE_RATE
    assert io.plays[0].samples == io.played_samples


async def test_play_rejects_the_wrong_format() -> None:
    io = FileAudioIO([])
    with pytest.raises(ValueError, match="24000 Hz mono"):
        await io.play(chunks_of(tone(0.1, rate=16000)))


async def test_realtime_capture_and_playback_take_real_time() -> None:
    io = FileAudioIO([silence(0.2)], realtime=True)
    started = time.monotonic()
    frames = await collect(io.frames())
    assert time.monotonic() - started == pytest.approx(7 * FRAME_SECONDS, abs=0.08)
    assert frames[-1].time == pytest.approx(io.now(), abs=0.05)
    started = time.monotonic()
    await io.play(chunks_of(tone(0.2, rate=OUTPUT_SAMPLE_RATE)))
    assert time.monotonic() - started == pytest.approx(0.2, abs=0.08)
    assert io.played_samples == int(0.2 * OUTPUT_SAMPLE_RATE)


async def test_wait_for_reply_holds_the_next_input_until_lucy_has_answered() -> None:
    io = FileAudioIO(
        [tone(0.5), tone(0.5)], wait_for_reply=True, reply_idle_s=1.0, reply_timeout_s=30
    )
    frames = io.frames()
    first = await collect(frames, 16)  # the first input (0.512 s)
    assert first[-1].time == pytest.approx(16 * FRAME_SECONDS)
    # Lucy answers for 2 s, then the next input follows after 1 s of quiet.
    await io.play(chunks_of(tone(2.0, rate=OUTPUT_SAMPLE_RATE)))
    reply_end = io.now()
    rest = await collect(frames)
    loud = [f for f in rest if np.max(np.abs(pcm_to_float(f.audio.data))) > 0.1]
    assert loud[0].time - FRAME_SECONDS >= reply_end + 1.0 - 1e-6
    assert loud[0].time - FRAME_SECONDS < reply_end + 1.0 + FRAME_SECONDS


async def test_wait_for_reply_gives_up_after_the_timeout() -> None:
    io = FileAudioIO([tone(0.5), tone(0.5)], wait_for_reply=True, reply_timeout_s=3.0)
    frames = await collect(io.frames())
    loud = [f for f in frames if np.max(np.abs(pcm_to_float(f.audio.data))) > 0.1]
    gap = loud[16].time - loud[15].time
    assert gap == pytest.approx(3.0, abs=FRAME_SECONDS * 1.5)


def test_wav_round_trip(tmp_path) -> None:
    chunk = tone(0.25, rate=OUTPUT_SAMPLE_RATE)
    write_wav(tmp_path / "a.wav", chunk.data, OUTPUT_SAMPLE_RATE)
    samples, rate = read_wav(tmp_path / "a.wav")
    assert rate == OUTPUT_SAMPLE_RATE
    assert np.allclose(samples, pcm_to_float(chunk.data))


def test_unreadable_wav_is_a_value_error(tmp_path) -> None:
    (tmp_path / "bad.wav").write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="cannot read WAV"):
        FileAudioIO([tmp_path / "bad.wav"])


# --- A fake sounddevice ----------------------------------------------------------


class FakePortAudioError(Exception):
    pass


class FakeStream:
    def __init__(self, sd: "FakeSD", kind: str, **options: Any) -> None:
        self.sd = sd
        self.kind = kind
        self.options = options
        self.callback = options["callback"]
        self.started = self.stopped = self.closed = False
        self.dead = False  # the device went away under it

    @property
    def active(self) -> bool:
        return self.started and not self.stopped and not self.dead

    def start(self) -> None:
        if self.kind in self.sd.fail_start:
            raise FakePortAudioError(f"Error opening {self.kind} stream [PaErrorCode -9986]")
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _device(name: str, inputs: int, outputs: int, rate: float) -> dict[str, Any]:
    return {
        "name": name,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "default_samplerate": rate,
    }


DEVICES = [
    _device("Monitor", 0, 2, 44100.0),
    _device("Built-in Microphone", 1, 0, 48000.0),
    _device("Built-in Speakers", 0, 2, 48000.0),
    _device("USB Headset", 1, 1, 16000.0),
]


class FakeSD:
    PortAudioError = FakePortAudioError

    def __init__(self, devices=DEVICES, default=(1, 2), fail_start=(), no_input=False) -> None:
        self.devices = [dict(d) for d in devices]
        self.default = SimpleNamespace(device=list(default))
        self.fail_start = set(fail_start)
        self.no_input = no_input
        self.streams: dict[str, FakeStream] = {}
        self.restarts = 0  # PortAudio terminated and initialised again
        self.opened = 0

    def _terminate(self) -> None:
        self.restarts += 1

    def _initialize(self) -> None:
        pass

    def query_devices(self, device: Any = None, kind: str | None = None) -> Any:
        if kind is not None:
            if kind == "input" and self.no_input:
                raise FakePortAudioError("No input device available")
            index = self.default.device[0 if kind == "input" else 1]
            return dict(self.devices[index], index=index)
        if device is not None:
            if not 0 <= device < len(self.devices):
                raise FakePortAudioError(f"Error querying device {device}")
            return dict(self.devices[device], index=device)
        return list(self.devices)

    def InputStream(self, **options: Any) -> FakeStream:  # noqa: N802
        stream = self.streams["input"] = FakeStream(self, "input", **options)
        self.opened += 1
        return stream

    def OutputStream(self, **options: Any) -> FakeStream:  # noqa: N802
        stream = self.streams["output"] = FakeStream(self, "output", **options)
        self.opened += 1
        return stream


def capture_block(stream: FakeStream, samples: np.ndarray) -> None:
    """Call the input callback from another thread, as PortAudio does."""
    thread = threading.Thread(
        target=stream.callback,
        args=(samples.reshape(-1, 1).astype(np.float32), len(samples), None, None),
    )
    thread.start()
    thread.join()


async def render(stream: FakeStream, frames: int) -> np.ndarray:
    """Ask the output callback for `frames` frames from another thread."""
    channels = stream.options["channels"]
    out = np.full((frames, channels), np.nan, dtype=np.float32)
    await asyncio.to_thread(stream.callback, out, frames, None, None)
    return out


async def started_io(**options: Any) -> tuple[SoundDeviceIO, FakeSD]:
    sd = options.pop("sd", None) or FakeSD()
    options.setdefault("poll_s", None)  # tests call `check_devices` themselves
    io = SoundDeviceIO(sd=sd, **options)
    await io.start()
    return io, sd


async def test_devices_open_at_their_native_rates() -> None:
    io, sd = await started_io()
    inp, out = sd.streams["input"], sd.streams["output"]
    assert inp.started and out.started
    assert inp.options["samplerate"] == 48000 and inp.options["channels"] == 1
    assert out.options["samplerate"] == 48000 and out.options["channels"] == 2
    assert inp.options["dtype"] == out.options["dtype"] == "float32"
    assert "Built-in Microphone (48000 Hz)" in io.describe()
    await io.close()
    assert inp.closed and out.closed


async def test_devices_can_be_chosen_by_name_or_number() -> None:
    io, sd = await started_io(input_device="usb", output_device=0)
    assert sd.streams["input"].options["device"] == 3
    assert sd.streams["input"].options["samplerate"] == 16000
    assert sd.streams["output"].options["device"] == 0
    assert sd.streams["output"].options["samplerate"] == 44100
    await io.close()


async def test_capture_is_resampled_to_16k_frames_across_threads() -> None:
    io, sd = await started_io()
    t = np.arange(48000) / 48000
    signal = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    for start in range(0, len(signal), 480):  # 10 ms blocks, as PortAudio delivers
        capture_block(sd.streams["input"], signal[start : start + 480])
    await asyncio.sleep(0.05)  # let the loop run the handed-over callbacks
    frames = await collect(io.frames(), 28)
    assert all(f.audio.sample_rate == 16000 and len(f.audio.data) == 1024 for f in frames)
    samples = np.concatenate([pcm_to_float(f.audio.data) for f in frames])
    assert dominant_frequency(samples[2000:], 16000) == pytest.approx(1000, abs=10)
    assert np.max(np.abs(samples[2000:])) == pytest.approx(0.5, abs=0.05)
    times = [f.time for f in frames]
    assert times == sorted(times)
    await io.close()


async def test_capture_backlog_drops_the_oldest_frames() -> None:
    io, sd = await started_io(backlog_s=0.32)  # 10 frames
    block = np.zeros(48000 // 2, np.float32)  # 0.5 s, about 15 frames
    capture_block(sd.streams["input"], block)
    await asyncio.sleep(0.05)
    io._frame_ready.set()
    frames = []
    async with asyncio.timeout(TIMEOUT):
        iterator = io.frames()
        while len(io._frames) or not frames:
            frames.append(await anext(iterator))
    assert len(frames) == 10
    await io.close()


async def test_silent_microphone_is_reported(caplog) -> None:
    now = [100.0]
    io, sd = await started_io(clock=lambda: now[0], silent_input_warning_s=3.0)
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            now[0] += 1.0
            capture_block(sd.streams["input"], np.zeros(480, np.float32))
        await asyncio.sleep(0.05)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "only silence" in warnings[0] and "Privacy & Security" in warnings[0]
    await io.close()


async def test_playback_is_resampled_played_and_counted() -> None:
    io, sd = await started_io()
    out = sd.streams["output"]
    started: list[bool] = []
    whole = tone(0.5, rate=OUTPUT_SAMPLE_RATE).data
    reply = [
        AudioChunk(data=whole[i : i + 4800], sample_rate=OUTPUT_SAMPLE_RATE)
        for i in range(0, len(whole), 4800)
    ]
    task = asyncio.create_task(io.play(chunks_of(*reply), on_start=lambda: started.append(True)))
    rendered = []
    async with asyncio.timeout(TIMEOUT):
        while not task.done():
            rendered.append(await render(out, 480))
            await asyncio.sleep(0)
    await task
    audio = np.concatenate(rendered)
    assert not np.isnan(audio).any()
    assert np.array_equal(audio[:, 0], audio[:, 1])  # mono on both channels
    loud = np.abs(audio[:, 0]) > 0.01
    assert loud.sum() == pytest.approx(0.5 * 48000, rel=0.02)
    first, last = np.flatnonzero(loud)[[0, -1]]
    assert dominant_frequency(audio[first:last, 0], 48000) == pytest.approx(440, abs=3)
    assert started == [True]
    assert io.played_samples == pytest.approx(0.5 * OUTPUT_SAMPLE_RATE, rel=0.01)
    await io.close()


async def test_output_is_silent_when_nothing_is_queued() -> None:
    io, sd = await started_io()
    audio = await render(sd.streams["output"], 256)
    assert not audio.any()
    assert io.played_samples == 0
    await io.close()


async def test_stop_playback_silences_at_once() -> None:
    io, sd = await started_io()
    out = sd.streams["output"]
    task = asyncio.create_task(io.play(chunks_of(tone(2.0, rate=OUTPUT_SAMPLE_RATE))))
    async with asyncio.timeout(TIMEOUT):
        while io.played_samples < 0.2 * OUTPUT_SAMPLE_RATE:
            await render(out, 480)
            await asyncio.sleep(0)
    io.stop_playback()
    heard = io.played_samples
    after = await render(out, 4800)
    assert not after.any()
    await asyncio.wait_for(task, TIMEOUT)
    assert io.played_samples == heard
    # A new play after a stop works normally.
    task = asyncio.create_task(io.play(chunks_of(tone(0.1, rate=OUTPUT_SAMPLE_RATE))))
    async with asyncio.timeout(TIMEOUT):
        while not task.done():
            await render(out, 480)
            await asyncio.sleep(0)
    assert io.played_samples - heard == pytest.approx(0.1 * OUTPUT_SAMPLE_RATE, rel=0.03)
    await io.close()


async def test_close_ends_frames_and_pending_play() -> None:
    io, sd = await started_io()
    frames = io.frames()
    reader = asyncio.create_task(collect(frames))
    player = asyncio.create_task(io.play(chunks_of(tone(1.0, rate=OUTPUT_SAMPLE_RATE))))
    await asyncio.sleep(0.01)
    await io.close()
    assert await asyncio.wait_for(reader, TIMEOUT) == []
    await asyncio.wait_for(player, TIMEOUT)


async def test_no_microphone_gives_a_readable_error() -> None:
    io = SoundDeviceIO(sd=FakeSD(no_input=True))
    with pytest.raises(AudioDeviceError, match="no microphone found") as info:
        await io.start()
    assert "Privacy & Security > Microphone" in str(info.value)


async def test_refused_microphone_gives_a_readable_error_and_closes_the_speaker() -> None:
    sd = FakeSD(fail_start={"input"})
    io = SoundDeviceIO(sd=sd)
    with pytest.raises(AudioDeviceError, match="cannot open the microphone 'Built-in Microphone'"):
        await io.start()
    assert sd.streams["output"].closed


async def test_unknown_device_names_point_to_the_list() -> None:
    with pytest.raises(AudioDeviceError, match="--list-devices"):
        await SoundDeviceIO(sd=FakeSD(), input_device="nonexistent").start()
    with pytest.raises(AudioDeviceError, match="no output channels"):
        await SoundDeviceIO(sd=FakeSD(), output_device=1).start()
    with pytest.raises(AudioDeviceError, match="not found"):
        await SoundDeviceIO(sd=FakeSD(), output_device=17).start()


def test_list_devices_marks_the_defaults() -> None:
    text = list_devices(FakeSD())
    assert "1: Built-in Microphone, 1 ch, 48000 Hz (default)" in text
    assert "2: Built-in Speakers, 2 ch, 48000 Hz (default)" in text
    assert "3: USB Headset" in text.split("Output devices")[0]
    assert "0: Monitor" in text.split("Output devices")[1]


def test_sound_device_wav_helpers_reject_float_wav(tmp_path) -> None:
    import wave

    path = tmp_path / "f.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(3)
        wav.setframerate(16000)
        wav.writeframes(bytes(30))
    with pytest.raises(ValueError, match="24-bit"):
        read_wav(path)


def test_concat_helper() -> None:
    assert concat(silence(0.1), tone(0.1)).duration == pytest.approx(0.2)


# --- Following the system's default devices ------------------------------------------


class FakeDefaults:
    """The system's default devices; here the ids are the fake PortAudio numbers."""

    def __init__(self, sd: FakeSD) -> None:
        self.sd = sd
        self.supported = True

    def current(self) -> tuple[int, int] | None:
        return tuple(self.sd.default.device) if self.supported else None  # type: ignore[return-value]

    def set(self, inp: int, out: int) -> None:
        self.sd.default.device = [inp, out]


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


async def following_io(**options: Any) -> tuple[SoundDeviceIO, FakeSD, FakeDefaults, Clock, list]:
    sd = FakeSD()
    defaults = FakeDefaults(sd)
    clock = Clock()
    notices: list[str] = []
    io, _ = await started_io(
        sd=sd, default_devices=defaults, clock=clock, on_notice=notices.append, **options
    )
    return io, sd, defaults, clock, notices


def settle(io: SoundDeviceIO, clock: Clock) -> None:
    """Two polls a second apart: the change is seen, then acted on."""
    io.check_devices()
    clock.t += 1.0
    io.check_devices()


async def test_an_output_change_reopens_on_the_new_speaker() -> None:
    io, sd, defaults, clock, notices = await following_io()
    old_out, old_in = sd.streams["output"], sd.streams["input"]
    defaults.set(1, 3)  # headphones: USB Headset at 16 kHz
    settle(io, clock)
    assert sd.restarts == 1
    assert old_out.closed and old_in.closed
    out = sd.streams["output"]
    assert out is not old_out and out.started
    assert out.options["samplerate"] == 16000 and out.options["channels"] == 1
    assert sd.streams["input"].options["device"] == 1  # the microphone is unchanged
    assert notices == ["sound: now using USB Headset (speaker)"]
    assert "speaker: USB Headset (16000 Hz)" in io.describe()
    await io.close()


async def test_an_input_change_reopens_and_frames_carry_on() -> None:
    io, sd, defaults, clock, notices = await following_io()
    old_in = sd.streams["input"]
    frames = io.frames()
    capture_block(old_in, np.full(48000 // 10, 0.1, np.float32))
    defaults.set(3, 2)
    settle(io, clock)
    assert notices == ["sound: now using USB Headset (microphone)"]
    new_in = sd.streams["input"]
    assert new_in.options["samplerate"] == 16000
    capture_block(old_in, np.full(4800, 0.9, np.float32))  # late block of the old stream
    capture_block(new_in, np.full(16000 // 10, 0.5, np.float32))
    await asyncio.sleep(0.05)
    got = await collect(frames, 3)
    levels = [float(np.max(np.abs(pcm_to_float(f.audio.data)))) for f in got]
    assert all(level < 0.6 for level in levels)  # the old stream's late block was ignored
    assert levels[-1] == pytest.approx(0.5, abs=0.01)
    await io.close()


async def test_both_changing_together_gives_one_switch_and_one_line() -> None:
    io, sd, defaults, clock, notices = await following_io()
    defaults.set(3, 3)
    settle(io, clock)
    assert sd.restarts == 1
    assert notices == ["sound: now using USB Headset"]
    await io.close()


async def test_changes_settle_before_switching() -> None:
    io, sd, defaults, clock, _ = await following_io()
    defaults.set(1, 3)  # output first ...
    io.check_devices()
    clock.t += 0.5
    defaults.set(3, 3)  # ... the microphone a moment later
    io.check_devices()
    clock.t += 0.5
    io.check_devices()
    assert sd.restarts == 0
    clock.t += 0.6
    io.check_devices()
    assert sd.restarts == 1
    io.check_devices()
    assert sd.restarts == 1
    await io.close()


async def test_an_explicit_device_stays_fixed() -> None:
    io, sd, defaults, clock, notices = await following_io(output_device="Built-in Speakers")
    defaults.set(1, 3)  # only the (explicit) output default changes: nothing to do
    settle(io, clock)
    assert sd.restarts == 0
    defaults.set(3, 3)  # the followed microphone changes
    settle(io, clock)
    assert sd.restarts == 1
    assert sd.streams["output"].options["device"] == 2
    assert sd.streams["input"].options["device"] == 3
    assert notices == ["sound: now using USB Headset (microphone)"]
    await io.close()


async def test_no_following_when_the_defaults_cannot_be_read() -> None:
    io, sd, defaults, clock, _ = await following_io()
    defaults.supported = False
    defaults.set(3, 3)
    settle(io, clock)
    assert sd.restarts == 0
    await io.close()
    io2, sd2 = await started_io(default_devices=None)
    sd2.default.device = [3, 3]
    io2.check_devices()
    assert sd2.restarts == 0
    await io2.close()


async def test_a_switch_while_idle_keeps_the_count_and_silence() -> None:
    io, sd, defaults, clock, _ = await following_io()
    task = asyncio.create_task(io.play(chunks_of(tone(0.2, rate=OUTPUT_SAMPLE_RATE))))
    async with asyncio.timeout(TIMEOUT):
        while not task.done():
            await render(sd.streams["output"], 480)
            await asyncio.sleep(0)
    heard = io.played_samples
    assert heard == pytest.approx(0.2 * OUTPUT_SAMPLE_RATE, rel=0.03)
    defaults.set(1, 3)
    settle(io, clock)
    assert io.played_samples == heard
    assert not (await render(sd.streams["output"], 256)).any()
    assert io.played_samples == heard
    await io.close()


async def test_a_switch_during_playback_continues_on_the_new_speaker() -> None:
    io, sd, defaults, clock, notices = await following_io()
    started: list[bool] = []
    whole = tone(1.0, rate=OUTPUT_SAMPLE_RATE).data
    reply = [
        AudioChunk(data=whole[i : i + 4800], sample_rate=OUTPUT_SAMPLE_RATE)
        for i in range(0, len(whole), 4800)
    ]
    task = asyncio.create_task(io.play(chunks_of(*reply), on_start=lambda: started.append(True)))
    before = []
    async with asyncio.timeout(TIMEOUT):
        while io.played_samples < 0.3 * OUTPUT_SAMPLE_RATE:
            before.append(await render(sd.streams["output"], 480))
            await asyncio.sleep(0)
    defaults.set(1, 3)  # 48 kHz stereo -> 16 kHz mono
    settle(io, clock)
    out = sd.streams["output"]
    after = []
    async with asyncio.timeout(TIMEOUT):
        while not task.done():
            after.append(await render(out, 160))
            await asyncio.sleep(0)
    await task
    assert started == [True]
    loud_before = int((np.abs(np.concatenate(before)[:, 0]) > 0.01).sum())
    loud_after = int((np.abs(np.concatenate(after)[:, 0]) > 0.01).sum())
    heard = loud_before / 48000 + loud_after / 16000
    assert heard == pytest.approx(1.0, abs=0.02)  # nothing lost or repeated beyond a few ms
    assert dominant_frequency(np.concatenate(after)[:4000, 0], 16000) == pytest.approx(440, abs=5)
    assert io.played_samples == pytest.approx(1.0 * OUTPUT_SAMPLE_RATE, rel=0.02)
    # Later playback is counted at the new rate.
    mark = io.played_samples
    task = asyncio.create_task(io.play(chunks_of(tone(0.25, rate=OUTPUT_SAMPLE_RATE))))
    async with asyncio.timeout(TIMEOUT):
        while not task.done():
            await render(out, 160)
            await asyncio.sleep(0)
    assert io.played_samples - mark == pytest.approx(0.25 * OUTPUT_SAMPLE_RATE, rel=0.03)
    await io.close()


async def test_a_failed_stream_is_reopened() -> None:
    io, sd, _, clock, notices = await following_io()
    sd.streams["output"].dead = True  # e.g. the AUHAL -50 error after unplugging
    io.check_devices()
    assert sd.restarts == 1 and sd.streams["output"].active
    assert notices == ["sound: reopened Built-in Speakers and Built-in Microphone"]
    # A stream that reports it finished (PortAudio's finished callback, on its thread).
    clock.t += 100
    io.check_devices()  # healthy: callbacks are not checked by the fake clock ... until now
    await asyncio.to_thread(sd.streams["input"].options["finished_callback"])
    await asyncio.sleep(0.01)
    io.check_devices()
    assert sd.restarts >= 2
    await io.close()


async def test_a_stalled_stream_is_reopened_and_our_own_stops_are_not_failures() -> None:
    io, sd, _, clock, _ = await following_io()
    old = sd.streams["output"]
    clock.t += 2.0
    await render(old, 64)
    capture_block(sd.streams["input"], np.zeros(480, np.float32))
    io.check_devices()
    assert sd.restarts == 0  # callbacks still come
    clock.t += 3.5  # and now they stopped
    io.check_devices()
    assert sd.restarts == 1
    # Our own stop of the old stream reports finished: ignored.
    await asyncio.to_thread(old.options["finished_callback"])
    await asyncio.sleep(0.01)
    assert not io._broken
    await io.close()


async def test_reopening_retries_with_pauses_and_ends_waiting_playback() -> None:
    io, sd, defaults, clock, notices = await following_io()
    task = asyncio.create_task(io.play(chunks_of(tone(2.0, rate=OUTPUT_SAMPLE_RATE))))
    await asyncio.sleep(0.01)
    sd.fail_start = {"output"}
    defaults.set(1, 3)
    settle(io, clock)
    await asyncio.wait_for(task, TIMEOUT)  # the turn ends instead of hanging
    assert len(notices) == 1 and notices[0].startswith("sound: cannot open the sound devices")
    opened = sd.opened
    for _ in range(5):  # within the pause nothing is tried: no spinning
        io.check_devices()
    assert sd.opened == opened
    clock.t += 1.0
    io.check_devices()  # retry 1 fails, quietly
    assert sd.opened == opened + 1 and len(notices) == 1
    played = io.played_samples
    await io.play(chunks_of(tone(0.1, rate=OUTPUT_SAMPLE_RATE)))  # no speaker: returns at once
    assert io.played_samples == played
    sd.fail_start = set()
    clock.t += 2.0
    io.check_devices()
    assert sd.streams["output"].active and sd.streams["input"].active
    assert notices[-1] == "sound: now using USB Headset (speaker)"
    task = asyncio.create_task(io.play(chunks_of(tone(0.1, rate=OUTPUT_SAMPLE_RATE))))
    async with asyncio.timeout(TIMEOUT):
        while not task.done():
            await render(sd.streams["output"], 160)
            await asyncio.sleep(0)
    assert io.played_samples - played == pytest.approx(0.1 * OUTPUT_SAMPLE_RATE, rel=0.05)
    await io.close()


async def test_the_watcher_polls_while_running_and_stops_on_close() -> None:
    sd = FakeSD()
    defaults = FakeDefaults(sd)
    io = SoundDeviceIO(sd=sd, default_devices=defaults, poll_s=0.01, settle_s=0.0, stall_s=60)
    await io.start()
    defaults.set(1, 3)
    async with asyncio.timeout(TIMEOUT):
        while sd.restarts == 0:
            await asyncio.sleep(0.01)
    await io.close()
    assert io._watcher is None


def test_defaults_are_not_followed_off_macos(monkeypatch) -> None:
    import sys

    from lucy.voice.audio import system_default_devices

    monkeypatch.setattr(sys, "platform", "linux")
    assert system_default_devices() is None
    assert SoundDeviceIO(sd=FakeSD())._defaults is None  # a fake `sd`: nothing to follow
