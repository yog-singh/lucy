# 0002: Ports and adapters

Date: 2026-11-11
Status: Accepted

## Context

Local models improve every few months. Lucy has to outlive any particular LLM, speech model or database.

## Decision

The core depends only on `Protocol` interfaces (`LLM`, `STT`, `TTS`, `Embedder`, `MemoryStore`, `Tool`). Implementations live in `adapters/`, register under a name, and are selected by config at startup. `core/` may not import from `adapters/`, `server/` or `clients/`; a lint check enforces this.

## Consequences

- Replacing a model means adding one adapter file and changing config.
- The core is testable with fakes, with no models loaded.
- Interfaces must stay small; a feature unique to one model has to be expressed as a capability (see 0008) or stay inside its adapter.
