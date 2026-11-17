"""`lucy vision`: download the face models, list and forget enrolled faces (ADR 0026).

`download` is the only network call face recognition makes: it fetches the InsightFace
model pack into `[vision.faces] models`; Lucy loads the models only from there.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from lucy.core.config import Settings
from lucy.core.workspace import Workspace


def add_parser(commands: Any, common: argparse.ArgumentParser) -> None:
    vision = commands.add_parser(
        "vision", parents=[common], help="face models and enrolled faces for the camera"
    )
    actions = vision.add_subparsers(dest="vision_action", required=True, metavar="ACTION")
    download = actions.add_parser(
        "download",
        parents=[common],
        help="download the face model pack (buffalo_l ~190 MB kept, buffalo_s ~16 MB)",
    )
    download.add_argument(
        "--pack", choices=("buffalo_l", "buffalo_s"), help="default: [vision.faces] pack"
    )
    actions.add_parser("list", parents=[common], help="enrolled faces")
    forget = actions.add_parser("forget", parents=[common], help="forget an enrolled face")
    forget.add_argument("name")


def run(
    settings: Settings,
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    download: Callable[[Path, str], Path] | None = None,
) -> int:
    """Run one `lucy vision` action; `download` is for tests."""
    from lucy.vision.faces import FaceBook

    faces = settings.vision.faces
    action = args.vision_action
    if action == "download":
        from lucy.adapters.faces_insightface import LICENCE, download_pack

        pack = getattr(args, "pack", None) or faces.pack
        models = Path(faces.models)
        print(f"Downloading the InsightFace {pack} face models into {models / pack}", file=err)
        print(f"Licence: {LICENCE}", file=err)
        try:
            path = (download or download_pack)(models, pack)
        except Exception as exc:
            print(f"lucy vision: download failed: {exc}", file=err)
            return 1
        print(path, file=out)
        return 0
    book = FaceBook(Workspace(settings.workspace.path))
    if action == "list":
        people = book.people()
        if not people:
            print("No faces enrolled.", file=out)
        for name, count in people.items():
            print(f"{name}\t{count} picture(s)", file=out)
        return 0
    if action == "forget":
        name = " ".join(str(args.name).split())
        if not book.remove(name):
            print(f"lucy vision: no face is enrolled for {name}", file=err)
            return 1
        print(f"Forgot {name}'s face.", file=out)
        return 0
    print(f"lucy vision: unknown action {action}", file=err)
    return 2
