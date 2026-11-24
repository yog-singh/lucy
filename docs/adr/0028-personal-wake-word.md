# 0028: A personal wake word trained on the user's voice

Date: 2026-11-24
Status: Accepted

## Context

`lucy listen` wakes on openWakeWord's pretrained "hey jarvis" (ADR 0018), a stand-in: the user wants "hey lucy", and the pretrained model woke on near-miss names in synthesised speech. openWakeWord's own training recipe needs thousands of synthetic clips from a TTS model, large background and room-impulse datasets, and torch, none of which Lucy carries. Lucy has one main user, who can record a few minutes of their own voice.

## Decision

- **A small head on frozen embeddings.** openWakeWord's mel-spectrogram and speech-embedding models stay as they are; only the per-phrase classifier is replaced, by a numpy network: the last 16 embeddings (16 x 96), normalised per dimension, flattened, Dense(32, ReLU), Dense(1, sigmoid). `OpenWakeWordDetector` loads it when `[wake] model` ends in `.npz`; streaming, warm-up and the ONNX path are unchanged.
- **npz, not ONNX.** The head is saved with `numpy.savez` (weights, normalisation, phrase, recommended threshold, format version) and loaded with `allow_pickle=False`. Exporting ONNX would need the `onnx` package or torch; training and inference need only numpy (onnxruntime still runs the two frozen models).
- **Trained on the user's own recordings** (`lucy wakeword record`, `train`, `test`). About 80 takes of the phrase (2 s each, from the `lucy listen` microphone) and about 10 minutes of everyday audio without it. Each take is augmented 30 times: placed on low noise or on a stretch of the negative audio at 5 to 20 dB SNR, with random gain (+-6 dB), speed (+-8%) and lead-in. Windows ending from the end of speech (by frame energy) to 320 ms after it are positive; those ending before the middle of the phrase or over 1.5 s after it are negative, so "hey" alone and the words after the phrase are learned as not-the-phrase. Every window of the negative audio, at three gains and also right after a detector reset, is negative.
- **Features exactly as when streaming.** `clip_embeddings` runs the mel model on the same 1760-sample pieces as the streaming path (its output depends on the loudest sample in its input, so one call over a whole clip differs) and batches the embedding model; on a real clip it gives the same embeddings and scores as the streaming detector to the last bit.
- **Training.** Adam, weighted binary cross-entropy (classes balanced, then negatives weighted 10 times), mini-batches of 256, fixed seed, early stopping on a held-out loss. Takes and one-minute pieces of the negative audio are split 80/20 by recording, never by augmented copy.
- **Evaluation as it runs.** The held-out takes (with silence around them) and the held-out negative audio go through the real streaming detector with the new head; a sweep from 0.3 to 0.9 counts takes detected and false wakes per hour (2 s refractory, as `WakeStage`). The recommended threshold, stored in the npz and printed with the `[wake]` lines, is the lowest with no held-out false wake that detects at least 90% of held-out takes; else the one with the fewest false wakes, then the most detections.
- The default `[wake]` stays hey jarvis: the personal model exists only after training.

## Rejected

- **openWakeWord's full synthetic training** (Piper-generated clips, ACAV100M features, room impulses, torch): hours of compute and gigabytes of data and dependencies, for a model of everyone's voice when one person's is wanted.
- **Template matching** (DTW over the user's embeddings or mel frames): simple, but brittle across loudness, distance and speed, and no principled threshold or false-wake estimate.
- **Porcupine** (Picovoice): good custom phrases, but a cloud console to make them, an access key and a licence for a closed engine.

## Consequences

- Tuned to the user's voice, but it is not speaker verification: other people saying the phrase will often wake Lucy, and it is no proof of who spoke.
- False wakes depend on the negative recordings: what was not in them (another TV channel, guests' voices, near-miss names) is not measured, and a few held-out minutes give a coarse false-wake rate. More negative audio can be dropped into the `negative/` directory and the model retrained.
- On 40 synthesised takes from macOS voices and 4 minutes of synthesised negatives with near misses, near misses such as "hey Lucas", "hey loosely" and "hello Lucy" still scored high in some voices; real recordings of one person have not been tried yet.
