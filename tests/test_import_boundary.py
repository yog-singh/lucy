"""`core/` must not import adapters, server, clients, memory or tools (ADR 0002)."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
CORE = SRC / "lucy" / "core"
FORBIDDEN = ("adapters", "server", "clients", "memory", "tools")


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
