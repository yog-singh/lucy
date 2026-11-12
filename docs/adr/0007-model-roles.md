# 0007: LLMs are requested by role

Date: 2026-11-11
Status: Accepted

## Context

Conversation, fact extraction and summarisation have different needs. A single configured LLM forces one model to do all three.

## Decision

Config maps roles (`chat`, `extraction`, `summary`, more as needed) to an adapter and model. Code asks for `gateway.for_role(name)` and never names a model. The gateway decides which models stay loaded within available memory.

## Consequences

- A small fast model can handle extraction while a larger one handles conversation.
- Prompts are versioned per role, and a model change for a role is verified with that role's evals.
- Two roles on different models may cause load/unload delays on 24 GB; the gateway has to manage this.
