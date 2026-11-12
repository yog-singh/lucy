# 0014: Voice pipeline, in-process and in three stages

Date: 2026-11-12
Status: Accepted

## Context

Lucy has a working text brain that takes a `UserMessage` and emits `SentenceReady` events. Voice has to be added without disturbing it. A full-duplex voice loop with barge-in and a wake word has several hard parts (echo, endpointing, continuous listening) that are independent of the basic loop.

## Decision

- Voice is a client of the existing orchestrator: speech in becomes `UserMessage(source="voice")`, and `SentenceReady` events drive speech out. The orchestrator's turn logic does not change.
- Voice-activity detection, endpointing, audio I/O and playback are client-side and live in `src/lucy/voice/`. STT and TTS are core ports with adapters.
- `lucy voice` runs everything in one process. The WebSocket server is deferred until a second client needs it.
- Build in three stages: half-duplex with a key press to interrupt; barge-in with echo cancellation on a single duplex audio stream; wake word.

## Consequences

- A usable voice loop arrives early, and each later stage adds one hard problem at a time.
- Stage 1 cannot be interrupted by voice.
- Moving to a duplex stream in stage 2 changes the audio layer, so stage 1's audio code should keep capture and playback behind one small interface.
- Without a server, voice is tied to the machine Lucy runs on until the server is built.
