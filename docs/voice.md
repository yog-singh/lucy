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

## Memory budget

The survey found the chat model occupying 16.2 GB in Ollama, shown with a 262,144-token context. With the speech models added the machine would be at about 20 GB of 24 and was already swapping.

The context length turned out not to be the cause. Loading the model at 32,768 through Ollama's native API starts it at 7.7 GB, but memory then grows by about 0.73 GB per distinct long prompt up to the same 16 GB ceiling at either setting: it is the MLX runner's prompt cache, and Ollama exposes no setting to bound it. The context length also cannot be set through the OpenAI-compatible endpoint Lucy uses.

Options considered: a GGUF (llama.cpp) build of the chat model, whose runner honours the context length and has a prompt-cache limit; a shorter `keep_alive`, since unloading frees the cache and reloading takes about 1.5 s; or raising it with Ollama upstream.

### Keeping the MLX build with bounded memory (implemented; ADR 0017)

Built into Lucy on 2026-11-13 and on by default. Three ten-turn voice runs with the policy inside Lucy measured a 1.26 to 1.34 s median from end of speech to first audio (worst turn 1.8 s), with the model at 10.0 GB peak and 8.4 GB median, matching the scripted measurements below. A longer voice conversation with tools and 3,100 to 3,700-token prompts measured a 1.53 s median; its re-warms took 9 to 11 s and finished before every next utterance. The measurements that led to it follow.

The MLX runner's cache limit is a compile-time constant in Ollama (8 GiB of cached state on top of the 7.2 GiB of weights), so no setting can change it. Each request, of any kind, adds about 0.7 to 1.0 GB until the ceiling is reached after 9 to 12 requests.

Unloading the model (`keep_alive: 0`, honoured on the OpenAI-compatible endpoint too) frees it in 0.3 s; a reload takes about 1 s and comes back at 7.75 GB. The policy that works: when Lucy has been idle for about a second after background work, unload, reload at once, and pre-fill the next turn's prompt prefix with two one-token requests. It requires the `stable_prefix` prompt layout, because in the default layout the date and memories change the system message on every turn and nothing can be pre-filled.

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

## Later options

- **Wyoming adapters.** Wyoming is the Home Assistant ecosystem's protocol for voice services. STT, TTS and wake-word adapters that speak it would let Lucy use existing Wyoming services and receive audio from room satellites.
- **Partial transcripts.** The `STT` port transcribes a finished utterance. If live partial text is wanted, the port gains a streaming method; Moonshine is the measured candidate (7.4% word error rate, native streaming).
