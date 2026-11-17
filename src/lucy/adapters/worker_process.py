"""A model worker in its own process, shared by the adapters that run one (ADR 0023, 0024).

Models whose packages stay out of Lucy's environment run in a PEP 723 script started
with `uv run --script`, speaking JSON lines on stdin/stdout. This module holds what
those adapters share: finding `uv`, the minimal child environment, and
`WorkerProcess`, which starts the command in its own process group (so uv's child goes
too), hands each stdout line to a callback on a reader thread, sends the worker's
stderr to Lucy's debug log, and stops the whole group on `stop`, `kill` and at
interpreter exit.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

STOP_GRACE: Final = 3.0
"""Seconds a worker gets to exit after its stdin closes, before it is killed."""
ENV_KEEP: Final = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "XDG_CACHE_HOME",
)
"""Variables passed to a worker (plus any `UV_*`); everything else is left out."""


class WorkerGone(Exception):
    """The worker exited (or its pipe broke)."""


def find_uv() -> str | None:
    """The `uv` executable: `$UV` (set by `uv run`), then PATH, then the usual places."""
    candidates = [os.environ.get("UV"), shutil.which("uv")]
    home = Path.home()
    candidates += [
        str(home / ".local" / "bin" / "uv"),
        str(home / ".cargo" / "bin" / "uv"),
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def worker_environment(*, offline: bool, **extra: str) -> dict[str, str]:
    """The minimal environment for a worker: `ENV_KEEP` and `UV_*` from Lucy's, unbuffered
    output, no telemetry, plus `extra`. `offline` (the model is cached) sets the Hugging
    Face offline variables; otherwise a token, if any, is passed so a download can work."""
    env = {k: v for k, v in os.environ.items() if k in ENV_KEEP or k.startswith("UV_")}
    env.update(PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1", **extra)
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    else:
        for key in ("HF_TOKEN", "HF_TOKEN_PATH"):
            if key in os.environ:
                env[key] = os.environ[key]
    return env


_LIVE: weakref.WeakSet[WorkerProcess] = weakref.WeakSet()


@atexit.register
def stop_all() -> None:
    """Kill every live worker (registered to run at interpreter exit)."""
    for worker in list(_LIVE):
        worker.kill()


class WorkerProcess:
    """One running worker: `on_line(bytes)` gets each stdout line on a reader thread and
    `on_line(b"")` once at end of file. `name` labels its threads and the stderr lines
    logged at debug level to `log`."""

    def __init__(
        self,
        command: Sequence[str],
        env: dict[str, str],
        on_line: Callable[[bytes], None],
        *,
        name: str = "worker",
        log: logging.Logger = logger,
    ) -> None:
        self.name = name
        self._log = log
        self.proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,  # its own process group, so uv's child goes too
        )
        self._on_line = on_line
        self._write_lock = threading.Lock()
        _LIVE.add(self)
        threading.Thread(target=self._read, name=f"lucy-{name}-out", daemon=True).start()
        threading.Thread(target=self._pump, name=f"lucy-{name}-err", daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        with contextlib.suppress(OSError, ValueError):
            for line in self.proc.stdout:
                if line:
                    self._on_line(line)
        self._on_line(b"")  # end of file

    def _pump(self) -> None:
        assert self.proc.stderr is not None
        with contextlib.suppress(OSError, ValueError):
            for raw in self.proc.stderr:
                text = raw.decode("utf-8", "replace").strip()[:500]
                if text:
                    self._log.debug("[%s] %s", self.name, text)

    def send(self, obj: dict[str, Any]) -> None:
        """Write one JSON line; raises `WorkerGone` if the pipe is closed."""
        assert self.proc.stdin is not None
        try:
            with self._write_lock:
                self.proc.stdin.write(json.dumps(obj).encode() + b"\n")
                self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise WorkerGone(f"cannot write to the worker: {type(exc).__name__}") from None

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, grace: float = STOP_GRACE) -> None:
        """Close stdin, wait up to `grace` seconds, then stop the whole process group."""
        with contextlib.suppress(OSError, ValueError):
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=grace)
        self.kill()

    def kill(self) -> None:
        """SIGTERM then SIGKILL the worker's process group (never Lucy's)."""
        pgid = self.proc.pid  # a session leader: its group id is its pid
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not _group_alive(pgid):
                break
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, sig)
            deadline = time.monotonic() + 1.0
            while _group_alive(pgid) and time.monotonic() < deadline:
                time.sleep(0.01)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=1.0)
        _LIVE.discard(self)


def _group_alive(pgid: int) -> bool:
    if pgid == os.getpgrp():
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
