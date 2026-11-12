# 0016: Record what was spoken with an appended event

Date: 2026-11-12
Status: Accepted

## Context

ADR 0004 says an interrupted turn records only what was delivered. In text chat, delivered means streamed. In voice, the model can finish generating while audio is still queued, and the user can interrupt playback after the turn's `AssistantMessage` has already been written with the full text. The log would then claim Lucy said things the user never heard.

## Decision

The voice client tracks playback and, when it is cut short, publishes a `PlaybackInterrupted` event carrying the turn id and the text actually spoken. The event is persisted. When a conversation is rebuilt from the log, a turn with a `PlaybackInterrupted` event has its reply replaced by the spoken text and marked interrupted.

`Interrupt` keeps its meaning: stop generating. The voice client sends both when the user interrupts mid-generation.

## Consequences

- The episode log stays append-only; nothing is edited after the fact.
- History rebuild gains one rule, and anything else that reads replies from the log must apply it too.
- Spoken-text tracking is approximate within a sentence: it is as precise as the alignment between audio played and text, which stage 1 does at sentence granularity.
- Memory extraction is unaffected, because it reads only what the user said.
