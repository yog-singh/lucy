"""The `opencv` camera adapter (ADR 0026): one still frame from a local camera.

The camera is opened for one capture and released right after, so it is on (and
the Mac's camera light lit) only while Lucy looks. A few frames are read and
discarded first so the exposure settles; the last is scaled down to `max_side`
pixels on its longer side and JPEG-encoded. All OpenCV work runs on a worker
thread. OpenCV (`opencv-python-headless`, the `vision` extra) is imported only
when a frame is taken, so Lucy runs without it until then.

The frame is returned in memory and never written to disk.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Final

from lucy.core import registry
from lucy.core.ports import CameraError
from lucy.core.types import Image

INSTALL_HINT: Final = (
    "Looking needs OpenCV, which is not installed. In the Lucy folder run "
    "`uv sync --extra vision` (or `uv pip install opencv-python-headless`), then restart Lucy."
)
PERMISSION_HINT: Final = (
    "The camera did not open. Check that no other app is using it and that the terminal "
    "running Lucy may use the camera: System Settings > Privacy & Security > Camera, turn "
    "it on for the terminal app, then restart the terminal and Lucy."
)

_LOCK: Final = threading.Lock()
"""One capture at a time per process: two opens of one device fight over it."""


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CameraError(INSTALL_HINT) from exc
    return cv2


@registry.register("camera", "opencv")
class OpenCVCamera:
    """A camera read through OpenCV's `VideoCapture`.

    `device` is the camera index (0: the built-in camera). `max_side` bounds the
    longer side of the frame in pixels (never enlarged). `warmup_frames` are read
    and discarded before the one kept. `jpeg_quality` is 1 to 100.
    """

    def __init__(
        self,
        *,
        device: int = 0,
        max_side: int = 1024,
        warmup_frames: int = 5,
        jpeg_quality: int = 85,
    ) -> None:
        if device < 0:
            raise ValueError("camera device must be 0 or more")
        if not 64 <= max_side <= 4096:
            raise ValueError("camera max_side must be between 64 and 4096")
        if not 0 <= warmup_frames <= 60:
            raise ValueError("camera warmup_frames must be between 0 and 60")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("camera jpeg_quality must be between 1 and 100")
        self.device = device
        self.max_side = max_side
        self.warmup_frames = warmup_frames
        self.jpeg_quality = jpeg_quality

    async def capture(self, *, warmup_frames: int | None = None) -> Image:
        """One JPEG frame; the camera is released before this returns. `warmup_frames`
        overrides the configured count for this capture."""
        warmup = self.warmup_frames if warmup_frames is None else max(0, warmup_frames)
        return await asyncio.to_thread(self._capture, warmup)

    def _capture(self, warmup_frames: int) -> Image:
        cv2 = _import_cv2()
        with _LOCK:
            capture = cv2.VideoCapture(self.device)
            try:
                if not capture.isOpened():
                    raise CameraError(PERMISSION_HINT)
                frame = None
                for _ in range(warmup_frames + 1):
                    ok, read = capture.read()
                    if ok and read is not None:
                        frame = read
                if frame is None:
                    raise CameraError(PERMISSION_HINT)
            finally:
                capture.release()
        return encode_frame(cv2, frame, max_side=self.max_side, quality=self.jpeg_quality)


def encode_frame(cv2: Any, frame: Any, *, max_side: int, quality: int) -> Image:
    """`frame` (a BGR array) scaled to at most `max_side` on its longer side, as JPEG."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CameraError("The camera frame could not be encoded as JPEG.")
    return Image(data=encoded.tobytes(), media_type="image/jpeg")
