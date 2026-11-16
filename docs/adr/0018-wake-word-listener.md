# 0018: Wake-word listener in front of the voice session

Date: 2026-11-13
Status: Accepted

## Context

Voice stage 1 needs `lucy voice` started by hand and holds the speech recogniser in memory the whole time (about 1.6 GB). The user wants a very lightweight listener that waits in the background for a wake phrase. This is stage 3 of the voice plan (ADR 0014); it does not depend on barge-in.

## Decision

- **`lucy listen`** runs the existing voice session behind a waiting stage. States: waiting, awake (a normal voice turn), a follow-up window in which the user can speak again without the phrase, then waiting again.
- **Nothing heavy before the first wake.** While waiting, only the microphone stream and the wake-word models are active. The whole app, including the speech recogniser and synthesiser, is built on the first wake and kept afterwards.
- **One breath.** A rolling two-second buffer is handed to the session on detection, wake phrase included, so the recogniser hears one continuous utterance; the phrase is then removed from the start of the transcript.
- **Detector.** openWakeWord's three ONNX models (mel-spectrogram, speech embedding, per-phrase classifier) run directly with `onnxruntime` behind a `WakeWordDetector` protocol. The `openwakeword` package is not used: its current release cannot be locked by uv on Python 3.12.
- **Models are downloaded, not shipped.** The pretrained phrase models are CC BY-NC-SA 4.0. `lucy listen --download-models` fetches them into `data/models/wakeword/` and checks each against a pinned SHA-256.
- **The pretrained "hey jarvis" model is a stand-in** to prove the pipeline. A custom "Hey Lucy" classifier, running at login, and splitting the waiting stage into its own small process are later steps.
- Half-duplex still applies: the wake phrase is ignored while Lucy is thinking or speaking.

## Consequences

- Waiting costs about 1.75% of one core and 67 MB, against 1.6 GB for an idle `lucy voice`.
- The first wake pays about 1.6 s of loading, which overlaps the user's request: the cold turn measured about 3 s from the wake phrase to the first audio, and 1.2 s from the end of speech.
- Background memory work does not run until Lucy has been woken once, because the app is not built before then.
- On synthesised speech the stand-in model detected 8 of 8 positives but also woke on 3 of 12 negatives, including "Hey Charles" in one voice at 0.985 and "Hey Travis" at 0.57. Near-miss names are a real false-wake risk; the threshold is configurable and a custom model needs a proper false-wake test.
- The wake phrase cannot interrupt Lucy until barge-in exists.

## Note, 2026-11-14: reminders

Due reminders are spoken whatever the listener's state (ADR 0020). While a reminder is thought about or spoken the wake stage is reset and not fed; the wake chime plays before it; afterwards the follow-up window opens without the wake phrase. Before the first wake a store-only scheduler calls `WakeListener.rouse()`, which builds the session without a wake phrase, so the app can now be built without a wake.

## Note, 2026-11-16: misheard wake phrase

On the user's voice Parakeet gets "Hey Jarvis" right in only a few clips (3 of 12 exact in `checks/stt_accuracy` run `flex1`); the rest come out as "Hey, Javis.", "Hi Javas.", "Hey, Douglas.", "Hey Joe is" and similar. The original matcher removed the close ones, but 6 user messages in `lucy listen` (2026-11-14 to 16) were the misheard phrase alone, and the model answered them as if they were messages. "Do I have a JavaS? Do I have anything on my calendar for today?" is not handled. `strip_wake_phrase` now has two more rules, both only at the very start: a greeting (hey, hi, hay, hai, hei, or the phrase's own first word) followed by a word that sounds like the name (a rough sound key, similarity at least 0.8: Javis, Javas, Javus, Davis, Darvis, Tarvis, Jobs; not Dave or Joe), always; and, for the first transcript after a detection only (`WakeListener.take_wake`, whose pre-roll holds the phrase), a greeting (also a, pay, he, okay, ok) followed by one or two words that end the sentence or clause, the first not a common word ("Hey, Douglas.", "Hey Joe is", "Pay Javis."; not "Hey, what's up?"). An empty result is the wake phrase alone: no chat turn, and the session keeps listening. The original transcript is logged when the phrase is removed. `lucy voice` is unchanged: it has no wake phrase, and none of these transcripts came from it.
