# 0009: Normalisation at the adapter boundary

Date: 2026-11-11
Status: Accepted

## Context

Every model has its own chat template, tool-call syntax, thinking markers and audio format. If these reach the core, each model swap becomes a core change.

## Decision

Adapters translate to and from the core's own types: one message format, one tool-call format, one input audio format (16 kHz mono PCM) and one output audio format. Model-specific quirks are handled inside the adapter and never appear in core types.

## Consequences

- The core and the episode log are independent of any model's format, so old logs stay readable after a swap.
- Adapters carry the translation work, including resampling audio.
- The core types must be general enough for future models; changing them later affects every adapter.
