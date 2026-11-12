# 0010: SQLite as the memory store

Date: 2026-11-11
Status: Accepted

## Context

Memory needs an append-only log, a graph of entities and statements, vector search and keyword search. It serves one user on one machine and must stay readable for many years. Alternatives considered were a dedicated graph database (Neo4j, FalkorDB) and Postgres with extensions.

## Decision

Store all memory in one SQLite file: relational tables for episodes, entities, statements and provenance; `sqlite-vec` for embeddings; FTS5 for keyword search. Access goes only through the `MemoryStore` port.

The episode log is the source of truth and is append-only. The graph and both indexes are derived and rebuildable. The single exception to append-only is a user's "forget that" request, which deletes the statements and their source episodes.

## Consequences

- No server to run; backup is copying one file.
- One person's graph is small enough that joins and recursive queries cover traversal. Multi-hop graph algorithms would be awkward; if they become necessary, a graph engine can be added behind the same port and populated from the log.
- `sqlite-vec` is a young extension. Because vectors are derived, replacing it does not risk data.
- Each vector records its embedding model, so changing embedder is a background re-embed.
