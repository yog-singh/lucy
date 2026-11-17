# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "insightface==0.7.3",
#   "onnxruntime==1.23.2",
#   "onnx==1.23.2",
#   "numpy==2.3.5",
#   "opencv-python-headless==5.0.0.93",
# ]
# [tool.uv]
# exclude-newer = "2026-11-17T00:00:00Z"
# ///
"""Worker for Lucy's `insightface` face adapter: InsightFace detection and ArcFace in its own env.

Started by `lucy.adapters.faces_insightface` as `uv run --script faces_insightface_worker.py`,
so InsightFace and its dependencies never enter Lucy's environment (ADR 0026). Not
imported by Lucy. It loads two models from `--pack-dir` (the SCRFD detector `det_*.onnx`
and the ArcFace recogniser `w600k_*.onnx`, put there by `lucy vision download`) and never
touches the network. Then it serves requests, one at a time, as JSON lines:

  stdout, once:   {"ready": true, "load_s", "providers", "detector", "recognizer"}
                  or {"fatal": "..."} and exit 2 if the models cannot be loaded
  stdin:          {"id": N, "image": "<base64 JPEG or PNG>"}
  stdout:         {"id": N, "faces": [{"box": [l, t, r, b] (fractions), "score",
                   "embedding": [512 floats, L2-normalised]}], "seconds"}
                  or {"id": N, "error": "..."}

Faces come largest first. Everything the libraries print goes to stderr: the protocol is
written to a private copy of the original stdout.
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path


def _protocol_stream():
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # library prints go to stderr
    return out


def _load(pack_dir: Path, provider: str, det_size: int):
    from insightface.model_zoo import get_model

    det_files = sorted(pack_dir.glob("det_*.onnx"))
    rec_files = sorted(pack_dir.glob("w600k_*.onnx"))
    if not det_files or not rec_files:
        raise FileNotFoundError(f"no det_*.onnx and w600k_*.onnx in {pack_dir}")
    providers = ["CPUExecutionProvider"]
    if provider == "coreml":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    detector = get_model(str(det_files[0]), providers=providers)
    recognizer = get_model(str(rec_files[0]), providers=providers)
    detector.prepare(0, input_size=(det_size, det_size), det_thresh=0.5)
    recognizer.prepare(0)
    return detector, recognizer, det_files[0].name, rec_files[0].name


def _faces(detector, recognizer, data: bytes):
    import cv2
    import numpy as np
    from insightface.app.common import Face

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("the image could not be decoded")
    height, width = img.shape[:2]
    boxes, kpss = detector.detect(img, max_num=0, metric="default")
    found = []
    for i in range(boxes.shape[0]):
        face = Face(bbox=boxes[i, 0:4], kps=kpss[i] if kpss is not None else None)
        face.det_score = float(boxes[i, 4])
        recognizer.get(img, face)
        left, top, right, bottom = (float(v) for v in boxes[i, 0:4])
        box = [
            max(0.0, left / width),
            max(0.0, top / height),
            min(1.0, right / width),
            min(1.0, bottom / height),
        ]
        area = (box[2] - box[0]) * (box[3] - box[1])
        embedding = [round(float(v), 6) for v in face.normed_embedding]
        found.append((area, {"box": box, "score": face.det_score, "embedding": embedding}))
    found.sort(key=lambda item: item[0], reverse=True)
    return [f for _, f in found]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", required=True, type=Path)
    parser.add_argument("--provider", default="cpu", choices=("cpu", "coreml"))
    parser.add_argument("--det-size", default=640, type=int)
    args = parser.parse_args()
    out = _protocol_stream()

    def say(obj) -> None:
        out.write(json.dumps(obj) + "\n")
        out.flush()

    started = time.perf_counter()
    try:
        detector, recognizer, det_name, rec_name = _load(
            args.pack_dir, args.provider, args.det_size
        )
    except Exception as exc:
        say({"fatal": f"{type(exc).__name__}: {exc}"})
        return 2
    say(
        {
            "ready": True,
            "load_s": round(time.perf_counter() - started, 3),
            "providers": detector.session.get_providers(),
            "detector": det_name,
            "recognizer": rec_name,
        }
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            request = json.loads(line)
            rid = request.get("id")
            t0 = time.perf_counter()
            faces = _faces(detector, recognizer, base64.b64decode(request["image"]))
            say({"id": rid, "faces": faces, "seconds": round(time.perf_counter() - t0, 4)})
        except Exception as exc:
            say({"id": rid, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
