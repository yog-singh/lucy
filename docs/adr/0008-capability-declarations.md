# 0008: Adapters declare capabilities

Date: 2026-11-11
Status: Accepted

## Context

Models behind the same port differ: some LLMs lack native tool calling or schema-constrained output, context windows vary, some TTS engines cannot stream.

## Decision

Each adapter exposes a `capabilities` value (for an LLM: `native_tools`, `json_schema`, `streaming`, `context_window`). The core reads these and adapts, for example by using prompt-based tool calling or sending whole sentences to a non-streaming TTS.

Amended 2026-11-12: structured output is the first capability with a real fallback. When an adapter declares `json_schema`, the gateway passes the schema for constrained decoding; when it does not, the gateway puts the schema in the prompt and validates the reply. Small models proved unreliable on the fallback path, so roles that need structured output should use an adapter with the capability.

## Consequences

- A swapped-in model that lacks a feature degrades gracefully instead of failing.
- The core carries fallback paths that need their own tests.
- `context_window` drives the `ContextBuilder` budgets, so they follow the model automatically.
