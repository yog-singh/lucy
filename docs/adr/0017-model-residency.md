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

## Note 2026-11-17: the prompt outgrew the re-warm

Voice turns waited 10 to 19 s for the first token. The prompt had grown to 10,443 tokens (counted by Ollama for gemma4:12b-mlx): system prompt 2,008, 18 tool specifications 3,522 (5 when this was decided), 130 history messages 4,857, date line 35. History was bounded only by the 200-event window: the history section's budget is a share of the chat role's `context_window` (32768), about 14,700 estimated tokens, so it never applied. Re-warming took 1.4 s to load plus 14.5 s to pre-fill, longer than the gap before the next utterance, and a turn without the pre-fill waited 12.4 s. `[context] history_tokens` (default 1,200 estimated tokens, before the latest user message so the pre-fill and the turn keep the same messages) now caps it, and startup warns when the estimated prefix exceeds 4,500 tokens. With the cap and `fixed` defaults for the two Gmail read tools: 6,847 tokens, re-warm 0.9 + 7.9 s, first token 0.62 s after a re-warm and 8.5 s without one. Still above the 3,100 to 3,700 assumed here, because of the tool specifications and the system prompt.

## Note 2026-11-17: offering fewer tools

Dynamic tool loading (ADR 0025) offers the core tools (lookups, time, weather) and a `more_tools` meta tool on every turn and the other groups only while the conversation uses them. Tool specifications sit after the system text and before the history (measured from `cached_tokens`), so the offered set is derived from the episode log and `next_turn_prefix` uses the set the next turn will be offered; groups drop only at that pre-fill. With the user's configuration the prompt is 3,935 tokens with the core set (7,188 before), 4,741 with email on and 5,485 with the calendar on. The startup warning estimates the prefix from the core set.

