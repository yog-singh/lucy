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

## Consequences

- Restarts lose nothing and any turn can be inspected or replayed.
- Each turn pays a small cost to rebuild session state; cache it if it shows up in latency measurements.
- Memory extraction is never on the response path.
