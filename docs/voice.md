# Lucy voice design

Voice lets the user talk to Lucy and hear her answer. It sits in front of the existing text brain: speech becomes a `UserMessage`, and the sentence stream the orchestrator already produces is spoken aloud.

The decisions are recorded in ADRs [0014](adr/0014-voice-pipeline-and-stages.md), [0015](adr/0015-speech-stack.md) and [0016](adr/0016-recording-what-was-spoken.md). The system around it is described in [`architecture.md`](architecture.md).

Status: stage 1 is built (`lucy voice`) and works end to end with the real models through audio files. It has not yet been tried with a real microphone and speakers. Measured latency is far above the estimate below because the machine is short of memory; see "Measured, stage 1".

## Pipeline

```
mic → resample → VAD → endpointing → STT → UserMessage(source="voice")
                                                  ↓
                                            orchestrator
                                                  ↓
speaker ← resample ← playback queue ← TTS ← SentenceReady events
```

Everything runs in one process, started with `lucy voice`, the same way `lucy chat` runs today. The WebSocket server comes later, when a second client (a menu-bar or phone app, or a room satellite) needs it.

| Step | What it does |
|---|---|
| Audio in | Captures the microphone at the device's native rate (48 kHz) and resamples to the core input format, 16 kHz mono 16-bit. |
| VAD | Marks each 32 ms frame as speech or not. |
| Endpointing | Decides the user has finished: speech followed by a silence of 0.5 to 0.7 s. |
| STT | Transcribes the whole utterance in one pass after the endpoint. |
| Orchestrator | Unchanged. It receives a `UserMessage` and emits `SentenceReady` events as the reply streams. |
| TTS | Synthesises each sentence as it becomes ready, streaming audio at 24 kHz mono 16-bit. |
| Playback | Queues audio, plays it, and tracks which text has actually been heard. |

Voice-activity detection, endpointing, audio I/O and playback are client-side signal processing and live in `src/lucy/voice/`. STT and TTS are core ports with adapters, like the LLM.

## Stages

1. **Half-duplex.** The microphone is ignored while Lucy is speaking and for a short time after. A key press stops her. This avoids the echo problem and gives a usable voice loop first.
2. **Barge-in.** Speaking over Lucy stops her. With speakers this needs echo cancellation, or she hears herself; with headphones it does not. Capture and playback move into one duplex stream so the echo canceller gets the exact audio being played, time-aligned.
3. **Wake word.** "Hey Lucy" starts a turn without the keyboard.

### Stage 3 so far: `lucy listen` with a stand-in wake word

Built on 2026-11-13 ([ADR 0018](adr/0018-wake-word-listener.md)) ahead of stage 2, because it does not depend on barge-in. It uses the pretrained "hey jarvis" model; a custom "Hey Lucy" model and running at login are still to do. Verified through audio files only.

The wake phrase is removed from the start of each transcript in `lucy listen`, including misheard forms: Parakeet often hears "Hey Jarvis" as "Hi Javas.", "Hey, Davis." or, further off, "Hey, Douglas." A greeting followed by a sound-alike of the name is always removed; a greeting followed by any uncommon word or two that ends the sentence only in the first transcript after a detection, since that audio holds the phrase. If nothing is left, the utterance was only the wake phrase: there is no chat turn and Lucy keeps listening for the request. The phrase is `[wake] phrase`; the log line "removed the wake phrase" keeps the original transcript. `lucy voice` has no wake phrase and does not remove anything.

Due reminders (ADR 0020) are spoken in `lucy voice` and `lucy listen` as soon as Lucy is idle, never over the user or another reply. In `lucy listen` the wake chime plays first, the follow-up window opens afterwards so the user can answer without the wake phrase, and a reminder due before the first wake starts Lucy on its own (paying the model-loading delay).

| | Result |
|---|---|
| Waiting footprint | 1.75% of one core, 67 MB (idle `lucy voice`: 1,604 MB) |
| Positives detected (synthesised, 8 voices) | 8 of 8, scores 0.997 to 0.999 |
| Negatives that woke Lucy | 3 of 12: "Hey Charles" in one voice (0.985), a sentence containing "Jarvis" (0.992), "Hey Travis" (0.568) |
| "Hey Lucy", ordinary sentences, Lucy's own replies, silence | 0.000 |
| Cold wake: wake phrase to first audio | about 3 s (1.6 s of loading overlaps the request) |
| Follow-up turn: end of speech to first audio | 1.57 s |

In the end-to-end run, "Hey Jarvis, what's the capital of Portugal?" was detected mid-sentence, transcribed with the phrase removed, and answered; a follow-up without the phrase was answered inside the eight-second window; a sentence after the window closed was ignored and never transcribed.

## Speech stack

Chosen from measurements on the target machine (Apple M5 Pro, 24 GB), with the chat model running. The full tables are in the journal entry of 2026-11-12.

| Component | Choice | Measured | Runner-up |
|---|---|---|---|
| STT | Parakeet TDT 0.6B v2 via `parakeet-mlx` | 4.3% word error rate; 0.05 s for a short utterance; nothing output on silence or noise | Whisper large-v3-turbo via `mlx-whisper`: 3.1% word error rate, 3.7 times slower, outputs "Thank you." on silence, pulls in PyTorch |
| TTS | `phonon` (Apache-2.0 build) served by Ollama at `/v1/audio/speech` | First voiced audio in about 0.05 s; streams; 24 kHz output; 177 MB | Kokoro via `kokoro-onnx`: 0.3 to 0.5 s to first audio |
| VAD | Silero VAD v6, ONNX model run with `onnxruntime` | 64 µs per frame; loads in 0.07 s | WebRTC VAD |
| Audio I/O | `sounddevice` (PortAudio) at the device's native rate | 61 ms output latency | Apple voice-processing I/O through `pyobjc` |
| Echo cancellation (stage 2) | WebRTC AEC3 via `pywebrtc-audio` | In simulation, echo no longer triggered the VAD or STT, and a barge-in was transcribed correctly | Apple voice-processing I/O |
| Wake word (stage 3) | openWakeWord with a custom-trained "Hey Lucy" model | 1.7% of one core while listening | livekit-wakeword |

The stack needs no PyTorch.

Not measurable without a person at the machine: how each voice sounds, real microphone and room behaviour, real echo through the laptop speakers, Bluetooth headsets, and wake-word error rates on real voices.

## English and Hindi: the `indic_flex` recogniser

`[stt] adapter = "indic_flex"` replaces Parakeet with `bodhan-ai/indic-transcribe-flex`, which understands English, Hindi and the two mixed, and writes everything in Latin script (ADR 0023). It runs in a worker process (`src/lucy/adapters/indic_flex_worker.py`, its own PyTorch environment built by `uv run --script`; `uv` must be installed), language identification between English and Hindi, "mixed" mode, float16 on the GPU; any Devanagari left is transliterated (`lucy.voice.translit`) and a guard drops text invented from noise (quiet audio, runaway or unsure generations, a bare "REDACTED").

- Setup: the model is gated. Accept its licence (Indic Open Model License) at <https://huggingface.co/bodhan-ai/indic-transcribe-flex>, log in with `hf auth login` (or set `HF_TOKEN`), and start `lucy voice` once with network access: uv builds the worker's environment and the 4.9 GB weights are downloaded. After that it works offline.
- Measured (checks/README.md, stt_accuracy): 16.5 % word error on the user's English (Parakeet 9.7 %), 6.2 % on Hindi (romanised), about 0.35 s per utterance after an 8 s warm-up, about 2.3 GB of GPU memory. No name biasing; a short noise burst can still come out as "Thank you".
- Options: `languages` (default `["en", "hi"]`), `mode` (`mixed`), `latin` (true), `device` (`mps` or `cpu`), `timeout` (20 s per utterance), `start_timeout` (900 s, for the first environment build), and the guard's `min_dbfs` (-55), `max_tokens_per_s` (25), `token_slack` (10), `min_mean_logprob` (-1.0), `max_repeats` (4). Dropped transcripts are logged at info level with the reason and the raw text; the worker's own output is in the debug log.
- Switching back to Parakeet is one line: `adapter = "parakeet_mlx"`.

## English and Hindi speech: the `magpie` voice

`[tts] adapter = "magpie"` replaces phonon with NVIDIA Magpie TTS Multilingual 357M (Q8_0) on the CPU through NeMo-Speech.cpp 0.3.0, one voice for English and Hindi (ADR 0024). A worker process (`src/lucy/adapters/magpie_worker.py`, standard library only, started with Lucy's own Python) loads the runtime's C library and streams each sentence in 186 ms chunks; the runtime's HTTP server is not used (it returns whole clips).

- Install by hand (no `pip`, no network at run time; NVIDIA's install script is not used because it edits the shell profile). Everything goes in `data/tools/nemo-speech/` (gitignored; delete it to remove):
  - Runtime: `nemo-speech-0.3.0-macos-aarch64-cpu.tar.gz` and `tn_configs.tar.bz2` from the NeMo-Speech.cpp 0.3.0 GitHub release (github.com/NVIDIA-NeMo/NeMo-Speech.cpp/releases; check them against the release's `SHA256SUMS`). Unpack the first into `cpu/` (so `cpu/lib/libnemo_speech_tts.dylib` exists) and the second into `tn_configs/` (`tn_configs/hi/`, `tn_configs/en/`, ...).
  - Model: `magpie_tts_multilingual_357m.v2607.q8_0.gguf` from `nvidia/magpie_tts_multilingual_357m` and `nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf` from `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` (Hugging Face, the GGUF files the runtime's docs name), into `models/`; the tokenizer folder of the Magpie repository into `models/nvidia/magpie_tts_multilingual_357m/<revision>/tokenizer` (the layout `nemo-speech pull` creates). Other paths can be given with the `model`, `codec`, `tokenizer` and `tn` options; a missing file is named in the error.
  - Exact files (as installed and measured, 2026-11-17; SHA-256 computed on the installed files):
    - `nemo-speech-0.3.0-macos-aarch64-cpu.tar.gz`: `2c3a7a719b778daeb14f147e011891f16e4b2ac62388d8b8d8b365cc2eebdf15` (matches the release's `SHA256SUMS`).
    - `tn_configs.tar.bz2` (0.3.0 release; not listed in `SHA256SUMS`): `2ca242c6d29f551eba3663d7e508c0d9dad10440e287628234154c6d1a72c7bc`.
    - `magpie_tts_multilingual_357m.v2607.q8_0.gguf`, `nvidia/magpie_tts_multilingual_357m` revision `446bff55`: `318c8f8376a28d8fdf3ce99b67b268af843c5128367000245ae30a1398c50cf9`.
    - `nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf`, `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` revision `fc00890b`: `cc86d36d821a27cdc1d4ef600a3e2b0dabe76e88fcc2a8652d9543134c07ef2d`.
    - Tokenizer folder: `nvidia/magpie_tts_multilingual_357m` revision `19806879b16d3f2ccf28fb112b1bcd16a3c7923e`.
- Language per sentence: `hi-IN` when the sentence contains Devanagari, else `en-US` (`language` forces one). With `hi-IN` Magpie reads Devanagari and digits (times, dates, amounts, percentages are normalised to Hindi words) but spells Latin letters out one by one, so Lucy writes Hindi replies entirely in Devanagari (persona) and the adapter respells any Latin word left inside a Hindi sentence in Devanagari (`lucy.core.devanagari`); a sentence that is mostly Latin goes as English.
- On screen (`lucy chat`, `lucy voice`, `lucy listen`) the replies' Devanagari is shown in Latin letters (`[display] latin = true`, the default); the conversation is stored as written.
- Measured (ADR 0024; checks/README.md, tts_listen `light1`): first audio 0.13 to 0.18 s per sentence through the worker on a quiet machine, 0.27 to 0.37 s through the adapter with the chat model loaded; real-time factor about 0.25 to 0.4 with 4 threads; about 1 GB; the chat model keeps its speed. Warm-up loads the model (under a second) and speaks one phrase per language (each language's tokenizer and normaliser load on first use, about 0.5 s).
- Options: `voice` ("Sofia"; Aria, Jason, John, Leo, Sofia; checked when the worker starts), `threads` (4), `seed` (0: the same sentence sounds the same; null for random), `language`, `timeout` (30 s per sentence), `first_audio_timeout` (5 s), `start_timeout` (60 s), `root` and the path options.
- Interrupting: the adapter cancels the sentence in the worker, which stops at the next chunk; none of its audio leaks into the next sentence.
- Licences: NeMo-Speech.cpp Apache-2.0; the Magpie model and the NanoCodec are under the NVIDIA Open Model License.
- Switching back to phonon is one line: `adapter = "ollama"` (Hindi replies are then Devanagari that phonon cannot read; see ADR 0024).

## Latency

Estimated from the measurements, from the end of the user's speech to the first audio of the reply:

| Stage | Time |
|---|---|
| Endpointing (0.5 s silence threshold) | 0.57 s |
| STT | 0.05 to 0.12 s |
| LLM to first complete sentence | about 1.0 s |
| TTS first audio | about 0.05 s |
| Output buffer | 0.06 to 0.08 s |
| **Total** | **about 1.75 s** |

The original target was about one second. Two planned improvements close most of the gap:

- Speak the first clause (the first six to eight words) instead of waiting for a whole sentence: about 1.2 to 1.3 s.
- Add a semantic end-of-turn check so the silence threshold can drop to 0.2 to 0.3 s: about 1.0 s. This needs testing on real speech.

Stage 1 ships with the simple version and measures real latency from the traces before either is added.

## Measured, stage 1

Ten synthesised user turns run through `lucy voice` in file mode at real-time pace, with the real models, 2026-11-12.

| | Result |
|---|---|
| Transcripts exact | 9 of 10 |
| Reply audio transcribes back to the reply text | 9 of 10 |
| End of speech to first audio | median 7.35 s (4.0 to 12.0 s) |
| Endpointing | 0.61 s |
| STT | 0.5 to 10.1 s |
| LLM to first sentence | median 2.2 s |
| TTS first audio | 0.02 to 0.04 s |
| Lucy process with Parakeet loaded | 1.6 GB idle, 2.5 GB peak |
| Start to ready-to-listen | about 2.1 s |
| Idle listening | 2.2% of one core |

The latency is a memory problem, not a pipeline problem. Parakeet transcribes in 0.1 s when its memory is resident; after the chat model has run between turns it took 6 to 10 s, because the chat model's 16 GB plus the speech models exceed what the GPU can keep resident and the machine was swapping. Paging Parakeet back in as soon as the user starts speaking cut that stage to 3.4 to 5.5 s. The chat model's first sentence is also slower than estimated, partly from the same pressure and partly from the loop detection added to the chat role.

Other findings:

- Half-duplex holds: speech injected while Lucy was speaking never reached the VAD.
- Interrupting 2.5 s into a reply stopped audio at 2.5 s; the recorded spoken text was one word short of what the audio contained, and the next turn's history held only that text.
- Facts stated by voice were extracted like typed ones.
- A pause longer than the 0.6 s threshold ends the turn, and with half-duplex the rest of the sentence is lost. In one run "and I work as a marine biologist" was dropped this way. This argues for smarter end-of-turn detection or a short grace period.
- Ollama's speech endpoint accepts only speed 1, and offers `default` plus 37 named voices.

## Recording what was spoken

With voice, what Lucy generated and what the user heard can differ: the user may interrupt while audio is still queued, even after the model has finished generating. The log should hold what was heard.

- The playback queue knows which sentences, and how much of the current one, have been played.
- When playback is cut short, the voice client publishes a `PlaybackInterrupted` event for that turn carrying the text actually spoken.
- The event is appended to the episode log. When the conversation is rebuilt, the reply for that turn is replaced by the spoken text and marked interrupted.
- Nothing already in the log is edited, so the log stays append-only.

## Who is speaking

Optional and off by default (`[speakers]`, ADR 0021). Each utterance is saved as a 16 kHz WAV in the workspace (`data/workspace/utterances/`, kept 1 day by default) and identified with `voice-fingerprint` concurrently with transcription; the turn waits at most `wait_ms` (150) after the transcript for the result. The name and score are recorded on the `UserMessage` and in the timings; nothing else uses them yet. Setup: `lucy speakers download`, then `lucy speakers enroll NAME a.wav b.wav c.wav`; `lucy speakers identify FILE` shows the scores. A voice match is not a security check. With the `enroll_voice` and `replace_voice` tools enabled, saying "enrol my voice" in `lucy voice` or `lucy listen` starts an enrolment Lucy leads herself: each sentence is shown on screen, the reading is recorded (not sent to the model), and the voice is enrolled at the end; "cancel" or Enter stops it.

## Memory budget

The survey found the chat model occupying 16.2 GB in Ollama, shown with a 262,144-token context. With the speech models added the machine would be at about 20 GB of 24 and was already swapping.

The context length turned out not to be the cause. Loading the model at 32,768 through Ollama's native API starts it at 7.7 GB, but memory then grows by about 0.73 GB per distinct long prompt up to the same 16 GB ceiling at either setting: it is the MLX runner's prompt cache, and Ollama exposes no setting to bound it. The context length also cannot be set through the OpenAI-compatible endpoint Lucy uses.

Options considered: a GGUF (llama.cpp) build of the chat model, whose runner honours the context length and has a prompt-cache limit; a shorter `keep_alive`, since unloading frees the cache and reloading takes about 1.5 s; or raising it with Ollama upstream.

### Keeping the MLX build with bounded memory (implemented; ADR 0017)

Built into Lucy on 2026-11-13 and on by default. Three ten-turn voice runs with the policy inside Lucy measured a 1.26 to 1.34 s median from end of speech to first audio (worst turn 1.8 s), with the model at 10.0 GB peak and 8.4 GB median, matching the scripted measurements below. A longer voice conversation with tools and 3,100 to 3,700-token prompts measured a 1.53 s median; its re-warms took 9 to 11 s and finished before every next utterance. The measurements that led to it follow.

The MLX runner's cache limit is a compile-time constant in Ollama (8 GiB of cached state on top of the 7.2 GiB of weights), so no setting can change it. Each request, of any kind, adds about 0.7 to 1.0 GB until the ceiling is reached after 9 to 12 requests.

Unloading the model (`keep_alive: 0`, honoured on the OpenAI-compatible endpoint too) frees it in 0.3 s; a reload takes about 1 s and comes back at 7.75 GB. The policy that works: when Lucy has been idle for about a second after background work, unload, reload at once, and pre-fill the next turn's prompt prefix with two one-token requests. It requires the `stable_prefix` prompt layout, because in the default layout the date and memories change the system message on every turn and nothing can be pre-filled.

Note 2026-11-17: the prompt had grown to 10,443 tokens (history 4,857 from the 200-event window, 18 tool specifications 3,522, system prompt 2,008), so the re-warm took about 16 s and turns waited 10 to 19 s for the first token. `[context] history_tokens` (default 1,200 estimated tokens) now caps the earlier conversation; with it the prompt is 6,847 tokens and the re-warm about 9 s (see the note in ADR 0017).


| Ten-turn voice run | End of speech to first audio (median) | STT median / max | Model memory, peak / median | Swap |
|---|---|---|---|---|
| No policy, default layout | 1.62 s | 0.09 / 0.48 s | 16.8 / 14.3 GB | 8.4 to 11.5 GB |
| No policy, `stable_prefix` | 1.30 s | 0.10 / 0.44 s | 16.8 / 14.2 GB | 11.2 to 14.0 GB |
| Unload and re-warm (4 runs) | 1.30 to 1.32 s | 0.09 to 0.10 / 0.15 to 0.26 s | 10.0 / 8.4 GB | falls or flat |
| No policy, with 4 GB of extra memory load | 1.57 s | 0.22 / 0.74 s | 16.8 / 14.2 GB | 7.3 to 19.3 GB |
| Unload and re-warm, same extra load | 1.30 s | 0.10 / 0.15 s | 10.0 / 8.4 GB | 9.4 to 10.0 GB |

- The 7.35 s median of the stage 1 measurement did not reproduce: with no policy at all, voice measured 1.3 to 1.6 s. Ollama's log shows only 2 to 4 GiB free at the time of the stage 1 run, so other processes were using more memory then. Bounding the model is therefore insurance against memory pressure and a large reduction in swap, not a fix for a fault that is always present.
- In voice the unload and re-warm happen while Lucy's reply is playing, so the next utterance found the model warm every time. Voice prompts in this test were short (300 to 780 tokens); a turn that arrives before a long prefix is re-warmed would cost about 4 s.
- In text chat with about 20 s between messages: median first word 0.37 s against 0.66 s without the policy. With 8 s between messages the re-warm cannot finish in time and turns after an unload take 3.5 to 7 s, so for text the unload should wait for a longer quiet period or a size threshold.
- Guard: Lucy counts as busy from the user message to the end of the turn, while speech is in progress, while any foreground or background stream runs or waits, and while a job is running or due. Checking only for an open stream would have unloaded in the gap between a tool call and the next step.
- Setting a zero keep-alive on background roles only is unsafe: a chat turn that pre-empts a background call starts cold.

### GGUF build, measured 2026-11-12 (not adopted yet)

`gemma4:12b-it-q4_K_M` (8.0 GB) was tested on a temporary second Ollama server with a 32,768 context and a bounded prompt cache.

| | GGUF | MLX (current) |
|---|---|---|
| End of speech to first audio, 10-turn voice run | median 1.77 s and 1.93 s | 7.35 s |
| STT between turns | 0.10 to 0.11 s | 0.5 to 10.1 s |
| Chat model memory over a session | 9 to 10 GiB, bounded | 7.7 growing to 16.1 GB |
| Generation speed with loop detection on | 44.6 tokens/s | 34.1 tokens/s |
| Grounding check | 40 to 41 of 56 | 48 of 56 |
| Tool-choice regression | 24 of 25 (the Apple CEO question answered from memory) | 25 of 25 |
| Memory eval | 17 of 18 | 17 of 18 |
| Extraction, new cases | 20 of 21 found, 7 spurious | 21 of 21, 4 spurious |

It fixes voice latency and matches the 1.75 s estimate, but only with two settings made on the Ollama side, and it is measurably worse at grounding answers in lookups.

- **Context length.** Ollama's app setting is 262,144 and applies to every model. It cannot be set per request on the OpenAI-compatible endpoint, and on this runner a native-API preload is undone by the next request. At 262,144 with the default prompt cache the GGUF build reached 22 to 23 GiB, worse than MLX.
- **Prompt cache.** llama.cpp's cache defaults to 8 GiB and Ollama has no setting for it; the environment variable `LLAMA_ARG_CACHE_RAM` reaches the runner. 2048 keeps the chat prefix across background calls for about 2 GiB.
- **Prefix reuse.** Lucy's prompt puts the date (to the minute) and memories before the history, so each turn diverges right after the persona and is reprocessed. Moving the date and memories after the history would let each turn extend the previous prompt, on either runner.
- The silent-loop fault still occurs on this build (5 of 174 replays), so the detection stays; it costs about 5% here instead of 35%.

## Configuration

```toml
[stt]
adapter = "parakeet_mlx"
model   = "mlx-community/parakeet-tdt-0.6b-v2"

[tts]
adapter = "ollama"
model   = "phonon"
voice   = "default"

[voice]
endpoint_silence_ms = 600
# input_device / output_device default to the system devices
```

### Sound devices

`input_device` and `output_device` take a device number or part of a name from `lucy voice --list-devices`; left unset, that side follows the system's default device while Lucy runs. PortAudio reads the defaults only when it starts and sends no notice when they change, so `SoundDeviceIO` reads them from CoreAudio (ctypes) about once a second. When a followed default has changed and held for a second (Bluetooth headphones switch output and input a moment apart), both streams are closed, PortAudio is restarted, and both reopen at the new devices' native rates; the terminal shows one line such as "sound: now using AirPods Pro". A side set explicitly keeps its device (a number can point elsewhere after devices come and go; a name is safer). A stream that stops on its own, or delivers no callback for 3 s (a device unplugged before the poll noticed: the `PaMacCore (AUHAL) err='-50'` case), takes the same reopen path, retried with growing pauses (1 s up to 30 s) and one "sound: cannot open the sound devices ...; retrying" line if nothing can be opened. Speech queued but not yet played when the speaker changes continues on the new one, re-resampled if its rate differs (the few milliseconds already handed to the old device are lost); the played-samples count used to record what was spoken stays in output samples across the switch. If no speaker can be opened, queued speech is dropped as if stopped, so the turn ends rather than waits. Following needs macOS; elsewhere only the recovery from failed streams applies. Tested against a fake device layer and a forced reopen on real devices; a real headphone switch has not been tested yet.

With Bluetooth headphones macOS makes the headset's microphone the default input too, in its low-quality call mode (a low sample rate such as 16 or 24 kHz, and lower playback quality while the microphone is in use): speech recognition and the sound of Lucy's voice may suffer. To keep the Mac's microphone while the sound follows the headphones, set the input in `[voice]`, for example `input_device = "MacBook Pro Microphone"`. Lucy does not change this by default.

## Later options

- **Wyoming adapters.** Wyoming is the Home Assistant ecosystem's protocol for voice services. STT, TTS and wake-word adapters that speak it would let Lucy use existing Wyoming services and receive audio from room satellites.
- **Partial transcripts.** The `STT` port transcribes a finished utterance. If live partial text is wanted, the port gains a streaming method; Moonshine is the measured candidate (7.4% word error rate, native streaming).
