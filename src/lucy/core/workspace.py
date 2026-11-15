"""Lucy's workspace: a directory of her own for files her code writes and reads (ADR 0021).

Created on demand, owner-only (0700). Every path handed out stays inside it; a
name or relative path that would leave it is refused. The chat model has no
tool that reaches it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

SPEAKERS: Final = "speakers"
"""The speaker book and enrolment clips."""
UTTERANCES: Final = "utterances"
"""Per-utterance voice clips, pruned by `[speakers]` retention."""

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,63}$")


class WorkspaceError(ValueError):
    """A path or name that would leave the workspace, or is not a plain name."""


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().absolute()

    @property
    def root(self) -> Path:
        """The workspace directory, created (0700) on first use."""
        if not self._root.is_dir():
            self._root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self._root, 0o700)
        return self._root

    def dir(self, *names: str) -> Path:
        """The subdirectory `names` (each a plain name), created 0700 on demand."""
        path = self.root
        for name in names:
            path = path / check_name(name)
            if not path.is_dir():
                path.mkdir(mode=0o700, exist_ok=True)
                os.chmod(path, 0o700)
        return self.resolve(path.relative_to(self.root))

    def resolve(self, relative: str | Path) -> Path:
        """`relative` (to the workspace) as an absolute path; refuses escapes."""
        root = self.root.resolve()
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise WorkspaceError(f"{relative} is outside the workspace")
        return path

    def relative(self, path: Path) -> str:
        """`path` relative to the workspace (refuses paths outside it)."""
        root = self.root.resolve()
        resolved = Path(path).resolve()
        if root not in resolved.parents:
            raise WorkspaceError(f"{path} is outside the workspace")
        return resolved.relative_to(root).as_posix()


def check_name(name: str) -> str:
    """`name` if it is a plain file or directory name (no separators, no leading dot)."""
    if not _NAME.match(name) or name in (".", ".."):
        raise WorkspaceError(f"not a plain name: {name!r}")
    return name
