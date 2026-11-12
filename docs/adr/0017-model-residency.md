# 0017: Model residency: bound the chat model's memory by unloading when idle

Date: 2026-11-13
Status: Accepted

## Context

Ollama's MLX runner keeps a prompt cache that grows by about 0.7 to 1.0 GB per request, of any kind, up to a ceiling of about 16 GB for `gemma4:12b-mlx`. The limit is a constant in Ollama's source; no setting changes it. On a 24 GB machine that leaves too little for the speech models and whatever else is running, and under memory pressure speech recognition slowed from 0.1 s to several seconds.

Two alternatives were measured. A GGUF build of the same model stays bounded and gave fast voice, but needs two settings on the Ollama side and grounds answers in lookups less well (40 to 41 of 56 on the grounding check against 48). Running the same weights under `mlx-lm`'s own server is not practical: Ollama's stored weights do not load there and it lacks schema-constrained output.

Unloading the model frees the cache in about 0.3 s and reloading takes about 1 s.

## Decision

Keep `gemma4:12b-mlx` and bound its memory from Lucy's side.

- **Policy.** `core/residency.py` unloads the chat model once Lucy has been idle, loads it again at once, and pre-fills the start of the next turn's prompt so the next reply still starts warm.
- **Busy means** any of: a turn in flight, from the user's message to the end of the turn, which covers the gap between a tool call and the next step; speech or transcription in progress; a foreground stream, or a background stream running or waiting; a job running or due; a warm-up of its own in flight.
- **Timing.** In voice, 1 s of idleness is enough, because the cycle fits inside the playback of Lucy's reply. In text, the policy also waits for a quiet period after the last turn, 10 s by default; if a message arrives during a cycle the period doubles, up to four times, and resets after an undisturbed cycle. An optional size threshold exists and is unset.
- **Pre-fill.** The prefix is built with Lucy's own `ContextBuilder` exactly as the next turn will render it, and sent twice with a one-token limit and slightly different endings, because the runner's cache only keeps a resume point where a request branches from a cached path. This requires the prompt layout in which per-turn content comes last (see the amendment to ADR 0004).
- **Capability, not assumption.** An adapter declares `residency` and provides `unload`, `load`, `loaded_size` and `prefill` through an optional port. Only the Ollama registration does. Roles that share a model share one state; the embedder and speech model are never touched.
- **Outside both lanes.** Warm-up requests bypass the gateway: they do not pre-empt, are not counted, take no fallback, and leave no trace or episode. A turn that arrives during a warm-up cancels a first pre-fill in flight and waits for anything else, for at most 15 s.
- **Voice.** The voice session publishes a transient `SpeechActivity` event that is never persisted. When speech starts and the model is not loaded, it is loaded at once.
- **Failure.** Any error or timeout ends the cycle with a warning; the next real request loads the model as before.

## Consequences

- In voice the model stays at about 8.4 GB median and 10 GB peak instead of 14 to 16 GB, with end of speech to first audio unchanged at a 1.3 s median, and swap falls or stays flat.
- In text with about 20 s between messages the first word arrives sooner than without the policy (0.36 s median against 0.66 s). With replies 8 to 12 s apart the policy stands back and the model still grows to 16 GB: fast text chat is not bounded.
- A turn that arrives before a long prefix is re-warmed costs about 2 to 5 s once. With 3,100 to 3,700-token prompts the re-warm took 9 to 11 s and still finished before every next utterance in the test.
- Unloading frees the model for every client of the Ollama server, so another program using the same model at that moment has to wait for it to reload.
- Background memory work loses its cache at each unload and reprocesses its prompts; this is not visible to the user.
- The mechanism depends on details of one runner's cache. It can be switched off with `[residency] enabled = false`, and should be revisited if Ollama makes the cache limit configurable.
