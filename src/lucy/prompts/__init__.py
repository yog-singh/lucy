"""Prompt texts, stored as Markdown files in this package and versioned with the code."""

from __future__ import annotations

from importlib.resources import files


def load_prompt(name: str) -> str:
    """Return the text of the prompt `<name>.md` in this package.

    Raises `FileNotFoundError` if there is no such prompt.
    """
    return files(__name__).joinpath(f"{name}.md").read_text(encoding="utf-8")
