"""Schema migrations for the SQLite memory store.

`MIGRATIONS[i]` upgrades a database from schema version `i` to `i + 1`; the
current version is kept in `PRAGMA user_version`. Migrations are append-only:
never edit or reorder a released entry, add a new one instead.
"""

from __future__ import annotations

from typing import Final

MIGRATIONS: Final[tuple[str, ...]] = (
    # 1: the episode log.
    """
    CREATE TABLE episodes (
      id             INTEGER PRIMARY KEY,
      event_id       TEXT NOT NULL UNIQUE,
      ts             TEXT NOT NULL,          -- ISO 8601 UTC
      session_id     TEXT NOT NULL,
      turn_id        TEXT,
      type           TEXT NOT NULL,          -- the event's type discriminator
      schema_version INTEGER NOT NULL,
      payload        TEXT NOT NULL,          -- the full event as JSON (event_to_json)
      extracted_at   TEXT                    -- NULL until memory extraction has processed it
    );
    CREATE INDEX episodes_session ON episodes(session_id, id);
    """,
    # 2: one trace per turn (ADR 0004).
    """
    CREATE TABLE turn_traces (
      turn_id    TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      started_at TEXT NOT NULL,                -- ISO 8601 UTC
      ended_at   TEXT NOT NULL,                -- ISO 8601 UTC
      outcome    TEXT NOT NULL,                -- completed | interrupted | error
      payload    TEXT NOT NULL                 -- the full TurnTrace as JSON
    );
    CREATE INDEX turn_traces_session ON turn_traces(session_id, started_at);
    """,
    # 3: the background job queue (ADR 0005).
    """
    CREATE TABLE jobs (
      id           INTEGER PRIMARY KEY,
      kind         TEXT NOT NULL,
      payload      TEXT NOT NULL,              -- JSON object
      status       TEXT NOT NULL
                   CHECK (status IN ('pending', 'running', 'done', 'failed')),
      attempts     INTEGER NOT NULL DEFAULT 0, -- runs started and counted
      max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
      run_after    TEXT NOT NULL,              -- ISO 8601 UTC, fixed width; not eligible before
      created_at   TEXT NOT NULL,              -- ISO 8601 UTC
      updated_at   TEXT NOT NULL,              -- ISO 8601 UTC
      last_error   TEXT,
      dedupe_key   TEXT
    );
    CREATE INDEX jobs_due ON jobs(status, run_after, id);
    -- At most one pending or running job per dedupe key.
    CREATE UNIQUE INDEX jobs_dedupe_active ON jobs(dedupe_key)
      WHERE dedupe_key IS NOT NULL AND status IN ('pending', 'running');
    """,
    # 4: the knowledge graph (docs/memory.md, ADR 0011) and its full-text index.
    # Vectors live in sqlite-vec tables created on demand per embedding model
    # (see lucy.memory.vector_index); only their registry is created here, so
    # connections without the extension can still open and migrate the file.
    """
    CREATE TABLE entities (
      id          INTEGER PRIMARY KEY,
      type        TEXT NOT NULL CHECK (type <> ''),
      name        TEXT NOT NULL CHECK (name <> ''),
      aliases     TEXT NOT NULL DEFAULT '[]'
                  CHECK (json_valid(aliases) AND json_type(aliases) = 'array'),
      summary     TEXT,
      merged_into INTEGER REFERENCES entities(id),
      created_at  TEXT NOT NULL,                  -- ISO 8601 UTC, fixed width
      updated_at  TEXT NOT NULL,
      CHECK (merged_into IS NULL OR merged_into <> id)
    );
    CREATE INDEX entities_merged_into ON entities(merged_into) WHERE merged_into IS NOT NULL;
    -- The user (USER_ENTITY_ID = 1), named "User" until the real name is learned.
    INSERT INTO entities (id, type, name, aliases, summary, merged_into, created_at, updated_at)
    VALUES (1, 'person', 'User', '[]', NULL, NULL,
            strftime('%Y-%m-%dT%H:%M:%f', 'now') || '000+00:00',
            strftime('%Y-%m-%dT%H:%M:%f', 'now') || '000+00:00');

    CREATE TABLE statements (
      id               INTEGER PRIMARY KEY,
      subject_id       INTEGER NOT NULL REFERENCES entities(id),
      predicate        TEXT NOT NULL CHECK (predicate <> ''),
      object_id        INTEGER REFERENCES entities(id),
      object_value     TEXT,
      text             TEXT NOT NULL CHECK (text <> ''),
      kind             TEXT NOT NULL CHECK (kind IN
                         ('fact', 'preference', 'habit', 'decision', 'goal', 'instruction')),
      origin           TEXT NOT NULL CHECK (origin IN ('stated', 'inferred')),
      confidence       REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
      evidence_count   INTEGER NOT NULL DEFAULT 1 CHECK (evidence_count >= 1),
      valid_from       TEXT,                     -- ISO 8601 UTC, fixed width; NULL = unknown
      invalid_at       TEXT,                     -- NULL = still true
      superseded_by    INTEGER REFERENCES statements(id),
      recorded_at      TEXT NOT NULL,
      last_observed_at TEXT NOT NULL,
      -- An object is an entity, a literal, or absent; never both.
      CHECK (object_id IS NULL OR object_value IS NULL),
      CHECK (superseded_by IS NULL OR (invalid_at IS NOT NULL AND superseded_by <> id)),
      CHECK (valid_from IS NULL OR invalid_at IS NULL OR invalid_at >= valid_from)
    );
    CREATE INDEX statements_subject ON statements(subject_id, kind);
    CREATE INDEX statements_object ON statements(object_id) WHERE object_id IS NOT NULL;
    CREATE INDEX statements_superseded_by ON statements(superseded_by)
      WHERE superseded_by IS NOT NULL;
    CREATE INDEX statements_kind ON statements(kind, origin, last_observed_at);

    CREATE TABLE statement_sources (
      statement_id INTEGER NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
      episode_id   INTEGER NOT NULL REFERENCES episodes(id),
      PRIMARY KEY (statement_id, episode_id)
    ) WITHOUT ROWID;
    CREATE INDEX statement_sources_episode ON statement_sources(episode_id, statement_id);

    CREATE TABLE profile_versions (
      id         INTEGER PRIMARY KEY,
      created_at TEXT NOT NULL,                 -- ISO 8601 UTC, fixed width
      content    TEXT NOT NULL
    );

    -- One row per (embedding model, dimensions) that has vectors; its vectors live
    -- in vec0 tables vec_statements_<id> and vec_entities_<id>.
    CREATE TABLE vector_spaces (
      id         INTEGER PRIMARY KEY,
      model      TEXT NOT NULL,
      dimensions INTEGER NOT NULL CHECK (dimensions > 0),
      created_at TEXT NOT NULL,
      UNIQUE (model, dimensions)
    );

    -- For the learning loop: unextracted episodes, and episodes by turn (forget).
    CREATE INDEX episodes_unextracted ON episodes(id) WHERE extracted_at IS NULL;
    CREATE INDEX episodes_turn ON episodes(turn_id, id);

    -- Full-text indexes over statement text and entity names, kept in sync by triggers.
    CREATE VIRTUAL TABLE statements_fts USING fts5(
      text, content='statements', content_rowid='id',
      tokenize='porter unicode61 remove_diacritics 2'
    );
    CREATE TRIGGER statements_fts_insert AFTER INSERT ON statements BEGIN
      INSERT INTO statements_fts(rowid, text) VALUES (new.id, new.text);
    END;
    CREATE TRIGGER statements_fts_delete AFTER DELETE ON statements BEGIN
      INSERT INTO statements_fts(statements_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END;
    CREATE TRIGGER statements_fts_update AFTER UPDATE OF text ON statements BEGIN
      INSERT INTO statements_fts(statements_fts, rowid, text) VALUES ('delete', old.id, old.text);
      INSERT INTO statements_fts(rowid, text) VALUES (new.id, new.text);
    END;

    CREATE VIRTUAL TABLE entities_fts USING fts5(
      name, aliases, content='entities', content_rowid='id',
      tokenize='porter unicode61 remove_diacritics 2'
    );
    INSERT INTO entities_fts(rowid, name, aliases) SELECT id, name, aliases FROM entities;
    CREATE TRIGGER entities_fts_insert AFTER INSERT ON entities BEGIN
      INSERT INTO entities_fts(rowid, name, aliases) VALUES (new.id, new.name, new.aliases);
    END;
    CREATE TRIGGER entities_fts_delete AFTER DELETE ON entities BEGIN
      INSERT INTO entities_fts(entities_fts, rowid, name, aliases)
        VALUES ('delete', old.id, old.name, old.aliases);
    END;
    CREATE TRIGGER entities_fts_update AFTER UPDATE OF name, aliases ON entities BEGIN
      INSERT INTO entities_fts(entities_fts, rowid, name, aliases)
        VALUES ('delete', old.id, old.name, old.aliases);
      INSERT INTO entities_fts(rowid, name, aliases) VALUES (new.id, new.name, new.aliases);
    END;
    """,
    # 5: key-value metadata for memory maintenance (decay state, profile input
    # hash, summary fingerprints, rejected merges, maintenance schedule). Keys
    # starting with "derived." describe the graph and are cleared by reset_graph.
    """
    CREATE TABLE memory_meta (
      key        TEXT PRIMARY KEY,
      value      TEXT NOT NULL,
      updated_at TEXT NOT NULL                  -- ISO 8601 UTC, fixed width
    ) WITHOUT ROWID;
    """,
)
"""Ordered SQL scripts; the schema version after applying `MIGRATIONS[:n]` is `n`."""
