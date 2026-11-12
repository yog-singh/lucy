"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.ports import GraphStore

#: Every `GraphStore` implementation the store contract tests run against. Each
#: factory takes `(path, embedder, *, clock)` and returns a store usable as an
#: async context manager. A second implementation (e.g. Postgres) is added here.
GRAPH_STORE_FACTORIES: dict[str, Callable[..., GraphStore]] = {"sqlite": SQLiteGraphStore}


@pytest.fixture(params=sorted(GRAPH_STORE_FACTORIES))
def make_graph_store(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """The store factory under test; tests using it run once per implementation."""
    return GRAPH_STORE_FACTORIES[request.param]
