# 0004: Turn lifecycle and a stateless orchestrator

Date: 2026-11-11
Status: Accepted

## Context

Responses need to be explainable ("why did she say that?"), survive restarts, and stay consistent when several clients share a conversation.

## Decision

Each input creates a `Turn` (id, session, cancellation token) that runs four steps: assemble context, run the LLM loop with tools, commit to the episode log, enqueue memory work.

- Context is assembled by a dedicated `ContextBuilder` with named sections and a token budget per section.
- An interrupted turn records only what was actually delivered.
- Every turn stores a trace with the exact prompt, model and timings.
- The orchestrator keeps no conversation state between turns; sessions are rebuilt from the log as a recent window plus a rolling summary.

Amended 2026-11-13: the context is laid out so each turn extends the previous one.

- The system message holds the persona and tool guidance, then standing instructions and the profile. The conversation history follows. The date line and retrieved memories, which change every turn, come last, as a system message right after the latest user message.
- The model server can then reuse its cached work for everything before the per-turn block, and the start of the next turn's prompt can be pre-filled (ADR 0017).
- Compared on the real model, two runs each: grounding 62 and 60 of 62 against 61 and 60 for the old layout; tool choice 25 and 25 of 25 against 25 and 24; memory eval 18 and 18 of 18 against 17 and 16; and the first word after a lookup arrived in about 3 s instead of 4 to 6 s.
- The old layout remains selectable with `[context] layout = "system"`.

## Consequences

- Restarts lose nothing and any turn can be inspected or replayed.
- Each turn pays a small cost to rebuild session state; cache it if it shows up in latency measurements.
- Memory extraction is never on the response path.
