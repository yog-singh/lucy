"""Small helpers shared by the checks: where outputs go and an isolated configuration.

Every check reads a Lucy settings file (default `lucy.example.toml`, never the user's
`lucy.toml` unless passed explicitly) and writes only under `data/checks/<name>/` or
an `--out` directory. Anything that opens a database gets a fresh one there, so the
real `data/lucy.db`, `data/evals/` and `data/chat_history` are never touched.

Run as a script, this module starts the `lucy` command line with such an isolated
configuration: `python checks/common.py lucy DB_PATH -- <lucy arguments>`.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "lucy.example.toml"
PROTECTED = [REPO / "data" / "lucy.db", REPO / "data" / "evals", REPO / "data" / "chat_history"]


def out_dir(name: str, out: str | Path | None = None) -> Path:
    """The output directory for check `name`: `--out` if given, else data/checks/<name>/."""
    path = Path(out) if out else REPO / "data" / "checks" / name
    path = path.resolve()
    for p in PROTECTED:
        if path == p or p in path.parents:
            raise SystemExit(f"refusing to write under {p}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def fresh_db(directory: Path) -> Path:
    """A new, empty database path in `directory` (an earlier run's files are removed)."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory / "lucy.db"


def isolated_settings(config: str | Path, db: Path):  # returns lucy.core.config.Settings
    """Settings from `config` with the memory database (and so the chat history and
    voice timings, which live beside it) moved to `db`."""
    from lucy.core.config import AdapterConfig, load_settings

    settings = load_settings(config)
    memory = settings.memory or AdapterConfig(adapter="sqlite")
    memory = memory.model_copy(update={"options": {**memory.options, "path": str(db)}})
    return settings.model_copy(update={"memory": memory})


def _run_lucy(db: str, argv: list[str]) -> int:
    """`lucy ARGV` with every settings file it loads redirected to the database `db`."""
    import lucy.clients.cli as cli

    cli.load_settings = lambda path: isolated_settings(path, Path(db))  # type: ignore[assignment]
    return cli.main(argv)


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "lucy" or sys.argv[3] != "--":
        raise SystemExit("usage: common.py lucy DB_PATH -- <lucy arguments>")
    sys.exit(_run_lucy(sys.argv[2], sys.argv[4:]))
