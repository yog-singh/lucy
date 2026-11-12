# 0012: The learning loop

Date: 2026-11-11
Status: Accepted

## Context

Storing what the user says is not enough for Lucy to improve. Habits and many preferences are never stated; they only show up as patterns across many conversations. Inferences can also be wrong, and a wrong belief about the user is worse than a missing one.

## Decision

Memory is maintained by five background steps:

1. **Extract** candidate statements from each turn and resolve their entities.
2. **Reconcile** each candidate against existing statements: add, reinforce, supersede or ignore.
3. **Reflect** nightly across episodes to infer habits and preferences, update decisions and goals, and merge duplicates.
4. **Decay** the confidence of inferred statements that stop being observed.
5. **Rewrite** the core profile from the strongest current statements.

Two rules govern inference:

- A stated statement always outranks an inferred one.
- When a low-confidence inference would change Lucy's behaviour, she asks the user first, and the answer becomes a stated statement.

The user can inspect any belief and its sources, correct it, or have it forgotten.

Amended 2026-11-12, after building and evaluating against the local models:

- Extraction records only what the user stated and never infers. Inference happens only in reflection, which must cite supporting messages that code then checks (3 messages on 2 days for a habit, 2 for a preference).
- Code guard rails override the model's reconciliation decision where a rule can be checked mechanically.
- The profile is rewritten after extraction settles, not only daily, because a stale profile contradicted fresh statements.
- Decay uses a 30-day grace period and a 30-day half-life. The original 14 and 90 separated stopped habits from continuing ones too slowly.
- A correction made through the CLI is written to the episode log as a user turn, so the log remains the source of truth and a rebuild keeps it.
- One model (`gemma4:12b-mlx`) serves all roles. The smaller model was faster but missed contradictions, invented profile content and made wrong merges.
- Asking the user to confirm an inference is not built yet.

## Consequences

- Lucy's model of the user improves without the user having to teach her explicitly.
- Learning lags conversation, since all steps run in the background lane; the recent-conversation window covers the gap.
- Quality depends on the extraction and reflection prompts and the model behind them, so both are versioned and checked with the memory eval set.
- Confirmation questions must be rare enough not to annoy; the threshold needs tuning.
- Reflection is the most expensive job and must be resumable, since foreground work can pause it.
