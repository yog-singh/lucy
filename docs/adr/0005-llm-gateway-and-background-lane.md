# 0005: LLM gateway with foreground and background lanes

Date: 2026-11-11
Status: Accepted

## Context

Live conversation and memory work (extraction, embedding, consolidation) compete for the same local hardware. Background work must never make Lucy slow to answer.

## Decision

All model calls go through an `LLMGateway` with two priorities. Foreground requests pause background work. Background jobs are stored in a SQLite-backed queue with retries.

Amended 2026-11-12, as implemented:

- A background stream does not start while a foreground stream is active. When a foreground stream starts, running background streams are stopped and their callers receive `Preempted`. Starting a foreground stream never waits on background cleanup.
- The job runner treats `Preempted` as a deferral: the job returns to the queue without using up an attempt.
- The queue assumes a single Lucy process per database file. Handlers must be safe to run twice, because a job interrupted by a crash is run again.

Amended 2026-11-13: a foreground stream first passes a gate held by the residency policy (ADR 0017). If a warm-up is in flight, the gate cancels a first pre-fill and waits for any other step, for at most 15 s. Warm-up requests themselves are outside both lanes.

## Consequences

- Conversation latency is protected.
- Jobs survive restarts and failures.
- During long conversations memory lags behind; retrieval must tolerate recent turns not being in the graph yet (the recent-conversation window covers this).
- Background jobs must be resumable or cheap to restart, since they can be paused at any time.
