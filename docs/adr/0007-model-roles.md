# 0007: LLMs are requested by role

Date: 2026-11-11
Status: Accepted

## Context

Conversation, fact extraction and summarisation have different needs. A single configured LLM forces one model to do all three.

## Decision

Config maps roles (`chat`, `extraction`, `summary`, more as needed) to an adapter and model. Code asks for `gateway.for_role(name)` and never names a model. The gateway decides which models stay loaded within available memory.

Amended 2026-11-12: a role can carry routing options that the gateway reads and the adapter never sees.

- `fallback` names another role. If a foreground stream on the primary role produces no text or tool call (it ends empty, or `first_output_timeout` passes), the gateway closes it and continues once on the fallback role. Nothing is retried after any output has been yielded. If the fallback is also silent the stream raises `SilentStreamError`.
- `after_tool` names a role to serve the step that follows a tool result. It exists but is off: on the grounding check it gained about 3 of 56, within noise, for a median 1.6 s more before the first word.
- The chat role's fallback is `chat_deep`, the same model with thinking on low. This recovers from a model fault where, with thinking off and tools offered, the model repeats an empty thought block and emits nothing visible.
- Fallbacks are recorded in the turn trace.

## Consequences

- A small fast model can handle extraction while a larger one handles conversation.
- Prompts are versioned per role, and a model change for a role is verified with that role's evals.
- Two roles on different models may cause load/unload delays on 24 GB; the gateway has to manage this.
