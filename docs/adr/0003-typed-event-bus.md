# 0003: Typed events on an async bus

Date: 2026-11-11
Status: Accepted

## Context

A voice assistant streams output, gets interrupted mid-sentence, serves more than one client, and sometimes speaks first. Plain request/response function calls handle none of these well.

## Decision

All inputs and outputs of the core are typed events on an in-process async bus. Inputs include `UserMessage`, `Interrupt`, `TimerFired`; outputs include `TextDelta`, `SentenceReady`, `ToolStarted`, `ToolFinished`, `TurnEnded`. Events are persisted, and that persisted stream is the episode log used by memory.

Amended 2026-11-11 after building the skeleton:

- Every event carries `schema_version`, the version of that event type's fields, so persisted events can be migrated when a type changes.
- An `AssistantMessage` event is emitted once per LLM step, after streaming finishes. It holds the text actually delivered, the tool calls made, and whether the step was interrupted. The conversation is rebuilt from `UserMessage`, `AssistantMessage` and `ToolFinished`; `TextDelta` and `SentenceReady` are for live clients only and are not persisted.

## Consequences

- Barge-in is an `Interrupt` event cancelling the running turn.
- Proactive behaviour is a turn started by a non-user event.
- Memory, debugging and replay share one record.
- Event types are a long-lived schema: they need versioning, and changes need migration.
