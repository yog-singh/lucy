"""The `insightface` face adapter (ADR 0026, stage 2): InsightFace's SCRFD detector and
ArcFace recogniser, run in a worker process.

InsightFace and its dependencies stay out of Lucy's environment: the models run in
`faces_insightface_worker.py` (shipped next to this module), a PEP 723 script with pinned
packages started with `uv run --script`, which loads the two models once and answers
JSON lines (see the worker's docstring). The worker starts on first use (or `warm_up`)
and is kept running; one thread speaks to it, one request at a time. A request that
takes longer than `timeout` kills the worker and fails; a worker that dies is restarted
once and the request retried. It runs offline: the models are loaded only from
`<models>/<pack>/`, which `lucy vision download` fills (`download_pack`, the only
network call), and the worker gets the offline environment.

The pretrained models are licensed by InsightFace for non-commercial research only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import shutil
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

from lucy.adapters.worker_process import WorkerGone, WorkerProcess, find_uv, worker_environment
from lucy.core import registry
from lucy.core.ports import DetectedFace, FaceRecognizerError
from lucy.core.types import Image

logger = logging.getLogger(__name__)

WORKER_SCRIPT: Final = Path(__file__).with_name("faces_insightface_worker.py")
PACKS: Final = ("buffalo_l", "buffalo_s")
PACK_URL: Final = "https://github.com/deepinsight/insightface/releases/download/v0.7/{pack}.zip"
PACK_FILES: Final = {
    "buffalo_l": ("det_10g.onnx", "w600k_r50.onnx"),
    "buffalo_s": ("det_500m.onnx", "w600k_mbf.onnx"),
}
"""The two models Lucy uses from each pack (detector, recogniser); the rest are not kept."""
LICENCE: Final = (
    "InsightFace's pretrained models are licensed for non-commercial research use only "
    "(https://github.com/deepinsight/insightface#license). Use them on that basis."
)
PROVIDERS: Final = ("cpu", "coreml")


def pack_dir(models: str | Path, pack: str) -> Path:
    return Path(models) / pack


def pack_present(models: str | Path, pack: str) -> bool:
    """Whether both models of `pack` are under `models`."""
    directory = pack_dir(models, pack)
    return all((directory / name).is_file() for name in PACK_FILES.get(pack, ("?",)))


def download_pack(models: str | Path, pack: str) -> Path:
    """Fetch `pack` from InsightFace's GitHub release into `<models>/<pack>/`, keeping only
    the detector and recogniser (the one network call; explicit)."""
    if pack not in PACK_FILES:
        raise ValueError(f"unknown face model pack {pack!r}; choose one of {', '.join(PACKS)}")
    target = pack_dir(models, pack)
    target.mkdir(parents=True, exist_ok=True)
    wanted = PACK_FILES[pack]
    with tempfile.TemporaryDirectory(prefix="lucy-faces-") as tmp:
        archive = Path(tmp) / f"{pack}.zip"
        with urllib.request.urlopen(PACK_URL.format(pack=pack), timeout=60) as response:  # noqa: S310
            with archive.open("wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                name = Path(member.filename).name
                if name in wanted and not member.is_dir():
                    with zf.open(member) as src, (target / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
    missing = [name for name in wanted if not (target / name).is_file()]
    if missing:
        raise FaceRecognizerError(f"the {pack} archive had no {', '.join(missing)}")
    return target


class _Worker(WorkerProcess):
    """The worker process with its replies in a queue, read one at a time."""

    def __init__(self, command: Sequence[str], env: dict[str, str]) -> None:
        self.lines: queue.Queue[bytes] = queue.Queue()
        super().__init__(command, env, self.lines.put, name="faces worker", log=logger)

    def receive(self, timeout: float) -> dict[str, Any] | None:
        """The next JSON line, or None on timeout; raises `WorkerGone` at end of file."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self.lines.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                if self.proc.poll() is not None and self.lines.empty():
                    raise WorkerGone(f"the worker exited (code {self.proc.returncode})") from None
                continue
            if not line:
                self.lines.put(b"")
                raise WorkerGone(f"the worker exited (code {self.proc.poll()})")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.debug("[faces worker stdout] %s", line[:200].decode("utf-8", "replace"))


@registry.register("faces", "insightface")
class InsightFaceRecognizer:
    """`FaceRecognizer` over InsightFace's detector and ArcFace, in a worker process."""

    def __init__(
        self,
        *,
        models: str | Path = "data/models/faces",
        pack: str = "buffalo_l",
        provider: str = "coreml",
        det_size: int = 640,
        timeout: float = 10.0,
        start_timeout: float = 120.0,
        command: Sequence[str] | None = None,
    ) -> None:
        if pack not in PACKS:
            raise ValueError(f"face model pack must be one of {', '.join(PACKS)}")
        if provider not in PROVIDERS:
            raise ValueError(f"face provider must be one of {', '.join(PROVIDERS)}")
        if not 160 <= det_size <= 1280 or det_size % 32:
            raise ValueError("face det_size must be a multiple of 32 between 160 and 1280")
        self.models = Path(models)
        self.pack = pack
        self.provider = provider
        self.det_size = det_size
        self.timeout = timeout
        self.start_timeout = start_timeout
        self._command = list(command) if command is not None else None
        """Replaces `uv run --script <worker>` (tests)."""
        self._worker: _Worker | None = None
        self._next_id = 0
        self._executor: ThreadPoolExecutor | None = None
        self._lock = asyncio.Lock()
        self.last_ready: dict[str, Any] = {}

    @property
    def present(self) -> bool:
        """Whether the pack's models are on disk."""
        return pack_present(self.models, self.pack)

    # --- Worker lifecycle (on the adapter's one thread) -------------------------------

    def _start_blocking(self) -> _Worker:
        if self._worker is not None and self._worker.alive():
            return self._worker
        if self._worker is not None:
            self._worker.kill()
            self._worker = None
        if self._command is not None:
            command = self._command
        else:
            if not self.present:
                raise FaceRecognizerError(
                    f"the face models ({self.pack}) are not in {self.models}; run "
                    "`lucy vision download`"
                )
            uv = find_uv()
            if uv is None:
                raise FaceRecognizerError(
                    "face recognition needs `uv` to run its worker, and it was not found; "
                    "install it from https://docs.astral.sh/uv/"
                )
            command = [uv, "run", "--quiet", "--script", str(WORKER_SCRIPT)]
        args = [
            f"--pack-dir={pack_dir(self.models, self.pack).absolute()}",
            f"--provider={self.provider}",
            f"--det-size={self.det_size}",
        ]
        try:
            worker = _Worker([*command, *args], worker_environment(offline=True))
        except OSError as exc:
            raise FaceRecognizerError(f"cannot start the face worker: {exc}") from exc
        try:
            reply = worker.receive(self.start_timeout)
        except WorkerGone as exc:
            worker.kill()
            raise FaceRecognizerError(
                f"the face worker exited while loading ({exc}); see the debug log"
            ) from None
        if reply is None or "fatal" in reply or not reply.get("ready"):
            worker.kill()
            why = (
                f"it was not ready after {self.start_timeout:.0f}s"
                if reply is None
                else str(reply.get("fatal") or reply)
            )
            raise FaceRecognizerError(f"cannot load the face models: {why}")
        self.last_ready = reply
        logger.info(
            "face worker ready: %s + %s on %s in %.1fs",
            reply.get("detector"),
            reply.get("recognizer"),
            reply.get("providers"),
            reply.get("load_s", 0.0),
        )
        self._worker = worker
        return worker

    def _ask_once(self, data: bytes) -> dict[str, Any]:
        worker = self._start_blocking()
        self._next_id += 1
        rid = self._next_id
        worker.send({"id": rid, "image": base64.b64encode(data).decode("ascii")})
        while True:
            reply = worker.receive(self.timeout)
            if reply is None:
                self._discard_worker()
                raise FaceRecognizerError(
                    f"finding faces took longer than {self.timeout:g}s; the worker was stopped"
                )
            if reply.get("id") != rid:
                continue
            if "error" in reply:
                raise FaceRecognizerError(f"finding faces failed: {reply['error']}")
            return reply

    def _ask_blocking(self, data: bytes) -> dict[str, Any]:
        """One request; if the worker dies, restart it once and retry."""
        try:
            return self._ask_once(data)
        except WorkerGone as exc:
            logger.warning("the face worker died (%s); restarting it", exc)
            self._discard_worker()
        try:
            return self._ask_once(data)
        except WorkerGone as exc:
            self._discard_worker()
            raise FaceRecognizerError(f"the face worker died twice in a row ({exc})") from None

    def _discard_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.kill()

    def _stop_blocking(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.stop()

    async def _call(self, fn: Any, *args: Any) -> Any:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lucy-faces")
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    # --- The port ------------------------------------------------------------------

    async def warm_up(self) -> None:
        """Start the worker and load the models, so the first look does not wait."""
        async with self._lock:
            await self._call(self._start_blocking)

    async def faces(self, image: Image) -> list[DetectedFace]:
        async with self._lock:
            reply = await self._call(self._ask_blocking, image.data)
        return [
            DetectedFace(
                box=tuple(float(v) for v in face["box"]),  # type: ignore[arg-type]
                score=float(face["score"]),
                embedding=tuple(float(v) for v in face["embedding"]),
            )
            for face in reply.get("faces", [])
        ]

    async def aclose(self) -> None:
        """Stop the worker and the thread. Idempotent."""
        async with self._lock:
            if self._executor is not None:
                await self._call(self._stop_blocking)
                executor, self._executor = self._executor, None
                executor.shutdown(wait=False)
            else:
                self._stop_blocking()
