"""`SQLiteGraphStore` implements the `GraphStore` protocol, signature for signature.

`runtime_checkable` only checks that the member names exist, so each member's
signature is compared with `inspect`: parameter names, kinds and defaults, and
the annotations (as written, since both modules use postponed annotations). A
protocol default of `...` means "has a default" without fixing its value.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.ports import GraphStore
from tests.graph_support import HashingEmbedder


def protocol_members() -> list[str]:
    names = []
    for name, value in vars(GraphStore).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(value) or isinstance(value, property):
            names.append(name)
    return sorted(names)


MEMBERS = protocol_members()


def test_protocol_lists_what_other_modules_use() -> None:
    expected = {
        "user_entity",
        "find_entities",
        "similar_entities",
        "neighbours",
        "supersede",
        "reinforce",
        "add_statement",
        "unextracted_episodes",
        "mark_extracted",
        "current_profile",
        "save_profile",
        "forget_statements",
        "reset_graph",
        "stats",
        "embedder",
    }
    assert expected <= set(MEMBERS)


@pytest.mark.parametrize("name", MEMBERS)
def test_member_signature_matches(name: str) -> None:
    proto = vars(GraphStore)[name]
    impl = inspect.getattr_static(SQLiteGraphStore, name)
    if isinstance(proto, property):
        assert isinstance(impl, property), f"{name} must be a property"
        assert (
            inspect.signature(impl.fget).return_annotation
            == inspect.signature(proto.fget).return_annotation
        )
        return
    assert inspect.iscoroutinefunction(impl) == inspect.iscoroutinefunction(proto), name
    want, got = inspect.signature(proto), inspect.signature(impl)
    assert list(want.parameters) == list(got.parameters), name
    for param_name, wanted in want.parameters.items():
        actual = got.parameters[param_name]
        assert actual.kind == wanted.kind, (name, param_name)
        assert actual.annotation == wanted.annotation, (name, param_name)
        if wanted.default is Ellipsis:
            assert actual.default is not inspect.Parameter.empty, (name, param_name)
        else:
            assert actual.default == wanted.default, (name, param_name)
    assert got.return_annotation == want.return_annotation, name


async def test_instance_is_a_graph_store(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "lucy.db", HashingEmbedder())
    assert isinstance(store, GraphStore)
    await store.close()
