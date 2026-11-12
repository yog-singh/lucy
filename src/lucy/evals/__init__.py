"""The standing memory eval (docs/memory.md "Measuring improvement"): `lucy eval`.

Scripted scenarios (`scenarios/*.toml`, `lucy.evals.scenario`) run against the
real configured models on a temporary database, each in its own fresh app, and
are scored by transparent checks (`lucy.evals.matching`): what the knowledge
graph holds afterwards, and keywords in Lucy's answers to questions asked in a
fresh session. Run it whenever a prompt or a model changes, and compare runs
with `lucy eval --compare OLD.json` (`lucy.evals.report`).

What is real and what is simulated (`lucy.evals.runner`):

- Real: the chat model (every `say` and every answer check is a full turn
  through the orchestrator, with retrieval), the extraction, reconciliation,
  reflection, decay, profile and summary jobs, the embedder, the SQLite stores
  and the `lucy memory correct` / `forget` / `rebuild` commands.
- Simulated: time. The app runs with a controlled clock, so "now" for the
  graph, extraction, reflection, decay, the profile, retrieval and the
  maintenance schedule is the scenario's time, starting on Monday
  `START`; user messages carry the scenario's timestamps. Lucy's replies and
  turn ends are logged at the real time (the orchestrator stamps them), which
  nothing in memory reads. Jobs run when a scenario says so (`jobs`,
  `maintenance`), not on the app's background timers, and the profile rewrite
  after extraction is not delayed.
- `history` steps log user messages directly (each as a complete turn without
  a reply), to build up weeks of conversation cheaply; with `extract = false`
  they are also marked extracted, as if extraction had found nothing in them
  (reflection still reads them).

Keyword grading of answers is deliberately simple and has limits: a reply can
mention a forbidden word while denying it ("you moved away from Lisbon"), or
say the right thing in other words than the required ones. Scenarios list
alternatives (`any`) where wording varies, and forbid only phrases that would
be wrong in any context. There is no LLM judge in the default path.
"""
