# 0011: Typed, temporal statements with provenance

Date: 2026-11-11
Status: Accepted

## Context

Lucy has to learn more than facts: the user's preferences, habits, decisions, goals, and how they want Lucy to behave. What she learns changes over time, comes with varying certainty, and must be explainable.

## Decision

Everything Lucy believes is a statement: an edge from a subject entity to an entity or value, with:

- a **kind**: fact, preference, habit, decision, goal or instruction;
- an **origin**: stated by the user or inferred by Lucy;
- a **confidence** and an evidence count;
- **validity times** (`valid_from`, `invalid_at`) separate from the time it was recorded;
- **sources**: links to the episodes it came from;
- a natural-language **text** that is embedded and indexed.

Predicates are free text, not a fixed ontology. Contradicted statements are closed with `invalid_at` and `superseded_by`, never overwritten. Decisions are entities carrying their options, reasoning and outcome as statements.

## Consequences

- One schema covers all kinds of knowledge, and each kind can be learned, ranked and used differently.
- Lucy can answer what is true now, what was true before, and why she believes it.
- Free-text predicates drift into near-duplicates; reflection has to merge them, and retrieval relies on `text`, not predicates.
- The graph only grows. Ranking and the validity filter keep stale statements out of context.
- Confidence values are heuristics that need tuning against the eval set.
