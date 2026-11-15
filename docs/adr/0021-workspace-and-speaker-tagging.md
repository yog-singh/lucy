# 0021: A workspace, and tagging who spoke (speaker awareness, stage 1)

Date: 2026-11-15
Status: Accepted

## Context

Lucy will need to tell apart the people talking to her (guests, per-person memory, a "listen to this conversation" mode). The first step is to know, for each voice utterance, who it probably was, without changing anything she does. Speaker recognition comes from the `voice-fingerprint` package (WeSpeaker ResNet34 on ONNX Runtime), whose documented API takes file paths. Voice recordings are sensitive, and Lucy makes no network requests of her own by default.

## Decision

- **A workspace.** `[workspace] path` (default `data/workspace`, gitignored with the rest of `data/`) is Lucy's own directory for files her code writes and reads. `lucy.core.workspace.Workspace` creates it and named subdirectories on demand with mode 0700, accepts only plain names, and refuses any path that resolves outside it. Today: `speakers/` (the speaker book `book.npz` and enrolment clips under `speakers/NAME/`) and `utterances/`. No tool gives the chat model access to it.
- **Stage 1 is a tag only.** With `[speakers] enabled = true`, each voice utterance is written as a 16 kHz mono WAV to `utterances/<UTC time>_<message id>.wav` and identified on a worker thread while it is transcribed. When the transcript is ready the turn waits at most `wait_ms` (default 150) for the result, then goes on with or without it. `UserMessage` gains optional `speaker` (accepted best match, else None), `speaker_score`, `audio_path` (relative to the workspace, only while the clip is kept) and `audio_s`; old events load unchanged. The prompt, history, memory extraction, confirmation and tools read only `text`, so the model request is identical with or without a tag (tested). The voice clients print `you (NAME)>` when a name is known, and the timings file records the speaker.
- **Retention.** Clips are kept `keep_utterances_days` (default 1) and pruned oldest-first beyond `max_utterances_mb` (default 200), at startup and every 50 utterances; 0 days deletes each clip right after identification. Clips are kept at all so that the user can later ask Lucy to identify or tag the voices in them; the period is short because family members also use Lucy and their voices end up in these clips.
- **Explicit download.** `lucy speakers download` fetches the weights into `data/models/speaker/`. The adapter always passes that file path to the library, whose weight resolution then never contacts the Hub (it returns the path or raises). If the file is missing, tagging is off with one notice naming the command.
- **A voice match is never an authorisation.** The embeddings are not anti-spoofing: a recording of an enrolled voice scores as high as the person. Later stages may use the tag to choose whose memory to use, but never to allow an action.

## Consequences

- Identification is ~15 ms for under 3 s of speech, ~140 ms at 3 to 4.5 s and ~280 ms at 12.7 s on this Mac, warm, so it usually finishes within transcription plus the wait bound. Under 3 s of speech (after the library's energy VAD) is zero-padded to its 3 s window, and such clips scored near 0 against an enrolment of the same voice: short replies ("yes") will mostly be untagged until that is improved upstream. `audio_s` is recorded so accuracy by length can be measured.
- The threshold (library default 0.40) is uncalibrated for Lucy's microphone.
