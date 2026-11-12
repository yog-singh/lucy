"""Memory: the episode log, the knowledge graph and retrieval (docs/memory.md).

- `sqlite_store`: the episode log and turn traces (`MemoryStore`, `TraceSink`).
- `graph_store`: entities, statements, provenance and profile versions.
- `retrieval`: the per-turn `MemoryContext` from the graph.
- `models`, `vector_index`, `fts`, `db`, `migrations`: supporting pieces.
"""
