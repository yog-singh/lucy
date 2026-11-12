"""Import boundaries: `core/` must not import outer layers (ADR 0002), and code
written against the `GraphStore` interface must not import the SQLite store."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
CORE = SRC / "lucy" / "core"
FORBIDDEN = (
    "adapters",
    "server",
    "clients",
    "memory",
    "tools",
    "voice",
    "prompts",
    "evals",
    "app",
)


def imported_modules(tree: ast.Module, package: list[str]) -> list[str]:
    """Absolute names imported by `tree`, resolving relative imports against `package`.

    For `from X import a`, both `X` and `X.a` are returned, so importing a
    forbidden subpackage by name (`from lucy import tools`) is caught.
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            names.append(module)
            names.extend(f"{module}.{alias.name}" for alias in node.names)
    return names


def is_forbidden(module: str) -> bool:
    return any(module == f"lucy.{p}" or module.startswith(f"lucy.{p}.") for p in FORBIDDEN)


def package_of(path: Path) -> list[str]:
    return list(path.parent.relative_to(SRC).parts)


CORE_FILES = sorted(CORE.rglob("*.py"))


def test_core_files_found() -> None:
    assert len(CORE_FILES) > 1


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_core_does_not_import_outer_layers(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    bad = [m for m in imported_modules(tree, package_of(path)) if is_forbidden(m)]
    assert not bad, f"{path.relative_to(SRC)} imports {bad}"


@pytest.mark.parametrize(
    ("source", "forbidden"),
    [
        ("import lucy.adapters.llm_ollama", True),
        ("from lucy.memory import store", True),
        ("from lucy import tools", True),
        ("from lucy.voice.session import VoiceSession", True),
        ("from lucy import app", True),
        ("from ..server import app", True),
        ("from .. import clients", True),
        ("from . import events", False),
        ("from lucy.core.types import Role", False),
        ("import asyncio", False),
    ],
)
def test_checker_catches_absolute_and_relative_forms(source: str, forbidden: bool) -> None:
    names = imported_modules(ast.parse(source), ["lucy", "core"])
    assert any(is_forbidden(n) for n in names) is forbidden


# --- Code written against `GraphStore` does not import the SQLite store ----------

GRAPH_STORE = "lucy.memory.graph_store"
#: The SQLite implementation itself, and the code that constructs the store.
GRAPH_STORE_IMPORTERS_ALLOWED = frozenset(
    {
        "memory/graph_store.py",
        "memory/sqlite_store.py",
        "memory/sqlite_jobs.py",
        "memory/vector_index.py",
        "memory/db.py",
        "memory/migrations.py",
        "memory/fts.py",
        "app.py",
    }
)
GRAPH_STORE_IMPORTER_DIRS_ALLOWED = ("clients/", "evals/")
LUCY_FILES = sorted((SRC / "lucy").rglob("*.py"))


def imports_graph_store(tree: ast.Module, package: list[str]) -> bool:
    return any(
        m == GRAPH_STORE or m.startswith(f"{GRAPH_STORE}.") for m in imported_modules(tree, package)
    )


def may_import_graph_store(path: Path) -> bool:
    rel = path.relative_to(SRC / "lucy").as_posix()
    return rel in GRAPH_STORE_IMPORTERS_ALLOWED or rel.startswith(GRAPH_STORE_IMPORTER_DIRS_ALLOWED)


def test_memory_files_found() -> None:
    assert any(p.parent.name == "memory" for p in LUCY_FILES)


@pytest.mark.parametrize(
    "path",
    [p for p in LUCY_FILES if not may_import_graph_store(p)],
    ids=lambda p: p.relative_to(SRC / "lucy").as_posix(),
)
def test_only_the_store_and_its_constructors_import_graph_store(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    assert not imports_graph_store(tree, package_of(path)), (
        f"{path.relative_to(SRC)} imports {GRAPH_STORE}; depend on lucy.memory.ports instead"
    )


@pytest.mark.parametrize(
    ("source", "imports"),
    [
        ("from lucy.memory.graph_store import SQLiteGraphStore", True),
        ("import lucy.memory.graph_store", True),
        ("from lucy.memory import graph_store", True),
        ("from .graph_store import DERIVED_META_PREFIX", True),
        ("from . import graph_store", True),
        ("from lucy.memory.ports import GraphStore", False),
        ("from . import fts", False),
    ],
)
def test_graph_store_checker_catches_every_form(source: str, imports: bool) -> None:
    assert imports_graph_store(ast.parse(source), ["lucy", "memory"]) is imports
