# Lucy memory design

Memory is what lets Lucy build context over time: she records what happens, draws conclusions about the user from it, and revises those conclusions as evidence accumulates.

This document describes the design. The decisions behind it are recorded in ADRs [0010](adr/0010-sqlite-memory-store.md), [0011](adr/0011-statement-model.md) and [0012](adr/0012-learning-loop.md). The system around it is described in [`architecture.md`](architecture.md).

Status: implemented in `src/lucy/memory/` and active in `lucy chat`. The sections below describe the design; "As implemented" near the end records where the built system differs, with measurements.

Not yet built: telling Lucy in conversation to correct or forget something (the `lucy memory` commands do this for now), Lucy asking the user to confirm an uncertain inference, answering questions about the past from invalidated statements, merging near-duplicate predicates, and ingesting documents.

## Three layers

| Layer | Contents | Role |
|---|---|---|
| Episodes | Everything said, done and ingested, stored raw | Source of truth, append-only |
| Knowledge graph | Entities connected by statements | What Lucy currently believes, with provenance |
| Core profile | A generated summary of the user, about 500 tokens | Loaded into every turn |

The graph and the profile are derived from episodes. They can be rebuilt from the log when prompts or models improve.

## Entities

An entity is anything statements can be about: a person, place, organisation, project, object, event or topic. The user is an entity too, and the subject of most statements.

Each entity has a canonical name, a list of aliases ("Priya", "my sister"), a type, and a short generated summary that is refreshed during reflection.

## Statements

A statement is one thing Lucy believes, expressed as an edge from a subject entity to another entity or a literal value.

```
statement
  subject → entity          predicate        object → entity or value
  text         "Yogesh prefers short answers in the morning"
  kind         fact | preference | habit | decision | goal | instruction
  origin       stated | inferred
  confidence   0.0–1.0      evidence_count
  valid_from   invalid_at   superseded_by
  sources → episodes
```

- **Predicates are free text.** There is no fixed ontology. Reflection merges near-duplicate predicates over time.
- **`text` is the retrieval surface.** Every statement carries a natural-language sentence that is embedded and indexed, so finding it never depends on the predicate chosen at extraction time.
- **Two time axes.** `valid_from` / `invalid_at` describe when the statement was true in the world. `recorded_at` describes when Lucy learned it.
- **Nothing is overwritten.** A contradicted statement gets an `invalid_at` and a pointer to its replacement, so Lucy knows both what is true now and what used to be.

### Kinds

| Kind | Example | How it is learned | How it is used |
|---|---|---|---|
| Fact | "Your sister Priya lives in Pune" | Stated once | Retrieved when relevant |
| Preference | "Prefers short answers in the morning" | Stated, or inferred from repeated choices | Shapes responses and suggestions |
| Habit | "Gym on Mon/Wed/Fri around 7am" | Inferred from patterns across episodes | Anticipation, scheduling, reminders |
| Decision | "Chose SQLite over Neo4j, for simplicity" | Captured when made | Resurfaced when a similar choice comes up |
| Goal | "Wants to ship the voice loop by January" | Stated | Tracked until done or dropped |
| Instruction | "Don't read URLs aloud" | The user's corrections of Lucy's behaviour | Always loaded, every turn |

Decisions are stored as an entity of type `decision` with statements for the options considered, the option chosen, the reasoning, and later the outcome. This keeps the reasoning attached to the choice.

### Stated and inferred

`origin` records whether the user said it or Lucy concluded it.

- A stated statement starts with high confidence and does not decay.
- An inferred statement starts low, gains confidence with each supporting observation, and decays when it stops being observed.
- A stated statement always outranks an inferred one on the same subject.

## The learning loop

```
turn ends
   │
   ▼
1. Extract ──► 2. Reconcile ──► graph
                                  │
            nightly               ▼
        3. Reflect ◄──────── episodes + graph
        4. Decay
        5. Rewrite profile
```

All steps run in the background lane (ADR 0005) using the `extraction` and `summary` model roles (ADR 0007).

### 1. Extract

After each turn, a job reads the new episodes with a little surrounding conversation and produces candidate statements as structured output: subject, predicate, object, text, kind, origin and any time expressions ("since March", "next Friday") resolved to dates.

Entity mentions are resolved against existing entities in three steps: exact name or alias match, then embedding similarity, then an LLM check for ambiguous cases. Unmatched mentions create new entities.

### 2. Reconcile

Each candidate is compared with existing statements about the same subject, found by embedding similarity. The outcome is one of:

| Outcome | When | Effect |
|---|---|---|
| Add | Nothing similar exists | New statement |
| Reinforce | Same meaning already known | `evidence_count` and confidence rise, new source linked |
| Supersede | Contradicts a current statement | Old one gets `invalid_at` and `superseded_by`; new one added |
| Ignore | Trivial, or already invalidated by a stated correction | Nothing stored |

Example: "I've switched to decaf" supersedes "drinks dark roast", which stays in the graph with an end date.

### 3. Reflect

A nightly job looks across many episodes for what no single turn shows.

- **Habits:** recurring activities at similar times or in similar contexts, using episode timestamps.
- **Inferred preferences:** consistent choices across similar situations.
- **Decision outcomes and goal progress:** open decisions and goals are checked against recent episodes.
- **Housekeeping:** duplicate entities and near-duplicate predicates are merged; entity summaries are refreshed.

Every inferred statement links to the statements and episodes that support it.

### 4. Decay

Inferred statements that have not been observed recently lose confidence on a slow schedule. Decay only lowers ranking; it never deletes. Stated statements, instructions and decisions do not decay.

### 5. Rewrite the profile

The core profile is regenerated from the strongest current statements: identity, key people, active projects and goals, strong preferences and habits. The previous version is kept, so changes to the profile can be reviewed.

### Checking inferences with the user

When an inferred statement would change Lucy's behaviour and its confidence is below a threshold, Lucy asks before acting on it ("You usually skip lunch on Fridays, should I stop suggesting it?"). The answer becomes a stated statement, confirming or superseding the inference.

## Retrieval

Each turn's context gets:

1. **Always:** the core profile and all active instructions.
2. **Per turn:** a search over current statements and entity summaries combining
   - vector similarity on statement `text`,
   - full-text keyword match,
   - one hop of graph expansion from the entities found.

Results are ranked by relevance, confidence and recency, then packed into the memory section's token budget (ADR 0004). Invalidated statements are excluded unless the question is about the past. Past decisions are retrieved when the current turn involves a similar choice.

Recent turns that extraction has not processed yet are covered by the recent-conversation window, not by memory.

## User control

| Request | Behaviour |
|---|---|
| "What do you know about X?" | Answered from the graph |
| "Why do you think that?" | Answered from the statement's sources |
| "That's wrong" | Statement superseded by a stated correction with high confidence |
| "Forget that" | Statements and their source episodes are deleted |

"Forget that" is the only operation that deletes episodes. Deleting only the statement would let the next rebuild extract it again.

As implemented (`SQLiteGraphStore.forget_statements`, one transaction, with a `dry_run` preview):

1. Every episode of each turn that sourced a forgotten statement is deleted, including Lucy's reply and tool events, since they usually repeat the information.
2. The requested statements are deleted, along with any statement left with no source. A statement that still has a source elsewhere survives.
3. A statement that had been superseded by a forgotten one is pointed at the next surviving successor, or reopened as current if there is none. This matches what a rebuild from the remaining log would produce.
4. Traces of those turns are deleted, plus any other trace that contains the forgotten text.
5. Entities left with no statements are deleted. Surviving entities touched by the deletion have their summary cleared.
6. Profile versions written since the earliest forgotten statement are deleted; the profile is regenerated.

Known gap: if the same information was repeated in turns that sourced no forgotten statement, those turns remain. The forget command should first collect related statements and show the preview.

## Storage

One SQLite file, behind the `MemoryStore` port. Schema changes are numbered migrations in `src/lucy/memory/migrations.py`, tracked with `PRAGMA user_version`. The `episodes` table below exists; the other tables are a sketch whose exact columns are settled when the graph is implemented.

```sql
-- Implemented (migration 1). Each row is one persisted event.
CREATE TABLE episodes (
  id INTEGER PRIMARY KEY,
  event_id TEXT NOT NULL UNIQUE,
  ts TEXT NOT NULL,
  session_id TEXT NOT NULL,
  turn_id TEXT,
  type TEXT NOT NULL,          -- user_message, assistant_message, tool_finished, ...
  schema_version INTEGER NOT NULL,
  payload TEXT NOT NULL,       -- the full event as JSON
  extracted_at TEXT            -- NULL until the extraction job has processed it
);

CREATE TABLE entities (
  id INTEGER PRIMARY KEY,
  type TEXT NOT NULL,
  name TEXT NOT NULL,
  aliases TEXT NOT NULL DEFAULT '[]',   -- JSON array
  summary TEXT,
  merged_into INTEGER REFERENCES entities(id)
);

CREATE TABLE statements (
  id INTEGER PRIMARY KEY,
  subject_id INTEGER NOT NULL REFERENCES entities(id),
  predicate TEXT NOT NULL,
  object_id INTEGER REFERENCES entities(id),
  object_value TEXT,
  text TEXT NOT NULL,
  kind TEXT NOT NULL,
  origin TEXT NOT NULL,                 -- stated | inferred
  confidence REAL NOT NULL,
  evidence_count INTEGER NOT NULL DEFAULT 1,
  valid_from TEXT,
  invalid_at TEXT,
  superseded_by INTEGER REFERENCES statements(id),
  recorded_at TEXT NOT NULL,
  last_observed_at TEXT NOT NULL
);

CREATE TABLE statement_sources (
  statement_id INTEGER NOT NULL REFERENCES statements(id),
  episode_id INTEGER NOT NULL REFERENCES episodes(id),
  PRIMARY KEY (statement_id, episode_id)
);

CREATE TABLE profile_versions (
  id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  content TEXT NOT NULL
);
```

The implemented schema (migration 4) follows this sketch with added constraints. The user is entity id 1, created by the migration and never deleted or merged away.

Derived indexes, both rebuildable:

- `sqlite-vec` tables of statement and entity embeddings, one set per embedding model and dimension. Search only uses the current model's vectors; after an embedder swap, a background job re-embeds what is missing.
- FTS5 tables over statement `text` and over entity names and aliases, kept in sync by triggers.

Measured with 20,000 statements and 1,000 entities at 768 dimensions (74 MB file): a full retrieval takes about 19 ms, and rebuilding the whole vector index takes about 6 seconds, excluding the time the embedding model needs.

Graph traversal uses ordinary joins for one hop and recursive queries beyond that.

## Measuring improvement

An eval set of scripted conversations followed by questions runs whenever a prompt or model changes. It covers:

- **Recall:** after these turns, does she know the fact?
- **Update:** after a contradiction, does she give the new value and still know the old one was true before?
- **Inference:** after a repeated pattern, has the habit or preference appeared with supporting evidence?
- **Restraint:** does she avoid storing trivia and avoid asserting weak inferences as fact?
- **Forgetting:** after "forget that", is the information gone and not re-extracted on rebuild?

## As implemented

All LLM roles use `gemma4:12b-mlx` with thinking off; the embedder is `embeddinggemma-2`. Thresholds below were measured for these models and must be re-measured if either changes.

### Extraction and reconciliation

- Runs as the background job `memory.extract` after each turn, one structured-output call per turn plus one per candidate that resembles something already known. About 2 seconds per turn.
- Statements come only from what the user said. Lucy's replies and tool results are context. Extraction never emits inferred statements; inference belongs to reflection.
- A single choice or order produces nothing. A preference is stored only when stated as one ("I always", "I can't stand"). A short-term plan is not a goal and produces nothing.
- A new person or thing must come with a statement linking it to the user; unlinked new entities are logged.
- A change of state is stored as the new state ("lives in Porto"), which supersedes the old one. The old statement ends at the date the user gave, or the date of the message, never the processing time.
- A candidate that is worded as a change ("switched to decaf") gets one dedicated reconcile call: it sees the subject's statements down to a lower similarity (0.76), rewrites the change as the new state in the form of the statement it replaces ("drinks decaf"), and supersedes the old one. A change that only says what ended ("left Brightline") is kept as it is.
- Giving something up is stored as a fact, and closes the goal and related habits.
- Code guard rails override the model: an inference never supersedes a stated statement; a statement is never superseded by one about a different subject; statements from the same turn are not reconciled against each other; near-identical additions become reinforcements.
- Confidence: stated starts at 0.9, inferred at 0.4, and each reinforcement moves it a quarter of the way to 1.0.

### Reflection

- Runs as `memory.reflect` in the daily maintenance pass, when at least 10 new user messages have arrived.
- Reads the user's messages over 60 days, grouped by weekday, and proposes habits and preferences with the messages that support them. Code enforces the evidence: a habit needs 3 messages on 2 different days, a preference needs 2 messages.
- An inferred statement's evidence is its source episodes, so forgetting cascades to inferences.
- Reviews open goals and decisions against recent messages, closing achieved or dropped goals at the date of the message and recording decision outcomes.

### Decay

- Inferred facts, preferences, habits and goals are untouched for 30 days after last being observed, then halve every 30 days, down to 0.05. A weekly habit that stops falls below the 0.6 "uncertain" line about 47 days after it was last seen. A monthly habit that continues keeps rising.
- The result is the same however often decay runs.

### Profile and maintenance

- The profile is rewritten after the extraction backlog drains (60 seconds after the user goes quiet in a running app) and in the daily pass. The model call is skipped when the selected statements are unchanged.
- Daily pass, in order: decay, duplicate-entity merging, reflection, entity summaries, profile. "Daily" means at most once per 24 hours while Lucy is running, checked at start and hourly.
- Duplicate entities are found by name-to-name similarity (0.9) and confirmed by the model before merging.

### Retrieval

- A vector match needs similarity of at least 0.68 and within 0.06 of the best match; keyword matches use content words and need vector support; entity expansion needs 0.75. At most 8 statements are returned.
- Measured on 61 statements and 44 queries: recall at 5 of 0.86, mean reciprocal rank 0.93, and 2.2 irrelevant statements per personal query. Six of seven general-knowledge questions retrieve nothing personal; greetings still retrieve a few.
- Parameters are overridable in the `[retrieval]` config section.

### User control

`lucy memory` provides `status`, `profile`, `search`, `about`, `why`, `correct`, `forget`, `work` and `rebuild`. A correction is also written to the episode log as a user turn, so it survives a rebuild.

Do not run `work`, `correct`, `forget` or `rebuild` while `lucy chat` is open on the same database: the job queue assumes one process.

### Evaluation

`lucy eval` runs 18 scripted scenarios against the configured models on a temporary database with a simulated clock, grades by keywords and graph checks, and saves a JSON report under `data/evals/` for comparison with `--compare`.

- First full run, 2026-11-12, 17 scenarios: 16 passed. The failure was a scenario whose messages stated the habit outright while expecting it to be inferred; the scenario was rewritten and a separate one added for stated habits.
- Second full run, 18 scenarios: 16 passed. Failures: `recall_after_weeks` (passes 2 of 3 when re-run alone) and `update_preference_change`, where "switched to decaf" was stored as a change and superseded nothing. That was fixed in the reconciler.
- Latest full run, 18 scenarios: 17 passed. The failure is `recall_after_weeks`, which is intermittent.
- Results vary between runs, so compare runs and re-run a failing scenario alone before concluding.

### Known limits

- When extraction writes a new state itself but its wording is not similar enough to the old one, the old statement stays current: "works at Corvid Labs" against "works at Fernhill Bank" scores 0.82, under the 0.84 floor, so both employers remain.
- An ending fact can remain beside the new state ("stopped driving to the office" next to "cycles to the office").
- On open questions ("who do I call on Sundays?") the chat model often states an unconfirmed inference as fact; it hedges more reliably on yes/no questions.
- A preference shown only through repeated choices of one kind (always the window seat) was not inferred in testing.
- When ending an activity, only habits and goals that resemble it or share a keyword are closed.
- Forgetting leaves turns that repeated the information without producing a statement; `forget` lists them in its preview.
- The check for invented content in profiles catches invented names and numbers, and gendered pronouns when no statement mentions that gender. It does not work out who a pronoun refers to.
- The chat prompt's history file (`chat_history` beside the database) keeps what the user typed; `lucy memory forget` does not remove text from it.

## Open questions

- Whether ingested documents use the same extraction prompt as conversation or a separate one.
- The confidence threshold and wording for asking the user to confirm an inference.
