# 0015: Speech stack chosen by measurement

Date: 2026-11-12
Status: Accepted

## Context

The original proposal named Whisper large-v3-turbo, Kokoro and Silero VAD from memory. Before building, a survey measured the available options on the target machine with the chat model running. It also found that the installed Ollama serves speech endpoints and that the installed `phonon` model is a text-to-speech model.

## Decision

- **STT:** Parakeet TDT 0.6B v2 through `parakeet-mlx`, run in-process as one pass over the finished utterance. It was within one point of Whisper large-v3-turbo's word error rate (4.3% against 3.1% on about 190 words), 3.7 times faster, silent on silence and noise where Whisper produced "Thank you.", and needs no PyTorch.
- **TTS:** `phonon` served by Ollama through `/v1/audio/speech`, using the Apache-2.0 build (`phonon:latest`), not the non-commercial `phonon:40m-en`. First audio arrives in about 0.05 s, it streams, and its 24 kHz output is the core output format. Kokoro through `kokoro-onnx` is the fallback.
- **VAD:** the Silero VAD ONNX model run directly with `onnxruntime`, avoiding the `silero-vad` package's PyTorch dependency.
- **Audio I/O:** `sounddevice`, opening devices at their native rate and resampling in the client.
- **Stage 2 and 3 candidates:** WebRTC AEC3 for echo cancellation; openWakeWord with a custom-trained model for the wake word.

The chat model is not used for transcription although it accepts audio: it scored 11.2% word error rate and answered silence and noise as if they were chat messages.

## Consequences

- No PyTorch in the stack.
- TTS depends on Ollama serving speech, which is new; Kokoro behind the same port is the fallback if that changes.
- Parakeet's weights are a 2.5 GB download and use about 1.7 GB of memory; the MLX cache limit must be set or the footprint nearly doubles.
- Voice quality has not been judged by a person. The voice is a config value.
- All numbers come from synthetic speech and should be re-checked with a real microphone.
- openWakeWord's current release cannot be locked by uv on Python 3.12 without restricting the lock to macOS, and its pretrained models are licensed for non-commercial use, so a custom model is needed.
