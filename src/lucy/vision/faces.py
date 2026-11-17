"""The face book and face matching (ADR 0026, stage 2).

The book lives in the workspace's `faces/` as `book.npz`: each enrolled person's name
and several face embeddings (one per enrolment frame). No photo is kept, only the
embeddings. Matching is cosine similarity of a face's (normalised) embedding against
each person's embeddings, the best per person, accepted from `threshold` up.

A face match is a tag, never an authorisation: a photo of an enrolled person matches
as well as the person does. Nothing may gate an action on it.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

from lucy.core.ports import DetectedFace
from lucy.core.workspace import Workspace

FACES: Final = "faces"
"""The workspace directory of the face book."""
BOOK_NAME: Final = "book.npz"

Position = Literal["left", "centre", "right"]


@dataclass(frozen=True)
class Person:
    """One face in view: `name` when it matched an enrolled person (else None),
    `score` the best similarity found (None with nobody enrolled), `position` where it
    is in the frame as the camera sees it."""

    name: str | None
    score: float | None
    position: Position


def position_of(face: DetectedFace) -> Position:
    """Left, centre or right third of the frame, by the centre of the face's box."""
    middle = (face.box[0] + face.box[2]) / 2
    if middle < 1 / 3:
        return "left"
    if middle > 2 / 3:
        return "right"
    return "centre"


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


class FaceBook:
    """Enrolled people's face embeddings, stored at `<workspace>/faces/book.npz`."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._lock = threading.Lock()
        self._people: dict[str, np.ndarray] | None = None

    @property
    def path(self) -> Path:
        return self._workspace.dir(FACES) / BOOK_NAME

    def _load(self) -> dict[str, np.ndarray]:
        if self._people is None:
            people: dict[str, np.ndarray] = {}
            path = self.path
            if path.is_file():
                with np.load(path, allow_pickle=False) as data:
                    names = [str(n) for n in data["names"]]
                    owners = data["owners"]
                    embeddings = data["embeddings"].astype(np.float32)
                for index, name in enumerate(names):
                    people[name] = embeddings[owners == index]
            self._people = people
        return self._people

    def _save(self) -> None:
        people = self._load()
        names = list(people)
        owners = np.concatenate(
            [np.full(len(people[n]), i, dtype=np.int32) for i, n in enumerate(names)]
            or [np.zeros(0, dtype=np.int32)]
        )
        vectors = [people[n] for n in names]
        embeddings = np.concatenate(vectors) if vectors else np.zeros((0, 0), np.float32)
        target = self.path
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".book-", suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as out:
                np.savez(
                    out, names=np.array(names, dtype=str), owners=owners, embeddings=embeddings
                )
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def people(self) -> dict[str, int]:
        """Enrolled names and how many embeddings each has."""
        with self._lock:
            return {name: len(vectors) for name, vectors in self._load().items()}

    def find(self, name: str) -> str | None:
        """The enrolled name matching `name`, ignoring case."""
        folded = name.casefold()
        return next((n for n in self.people() if n.casefold() == folded), None)

    def enroll(self, name: str, embeddings: Sequence[Sequence[float]]) -> None:
        """Store `embeddings` for `name`, replacing any it had."""
        if not embeddings:
            raise ValueError("no face embeddings to enrol")
        vectors = np.stack([_unit(e) for e in embeddings])
        with self._lock:
            people = self._load()
            existing = next((n for n in people if n.casefold() == name.casefold()), None)
            if existing is not None:
                del people[existing]
            people[name] = vectors
            self._save()

    def remove(self, name: str) -> bool:
        """Forget `name` (ignoring case); False if it was not enrolled."""
        with self._lock:
            people = self._load()
            existing = next((n for n in people if n.casefold() == name.casefold()), None)
            if existing is None:
                return False
            del people[existing]
            self._save()
            return True

    def scores(self, embedding: Sequence[float]) -> list[tuple[str, float]]:
        """Each enrolled person's best cosine similarity with `embedding`, best first."""
        probe = _unit(embedding)
        with self._lock:
            people = self._load()
            ranked = [
                (name, float(np.max(vectors @ probe)))
                for name, vectors in people.items()
                if len(vectors) and vectors.shape[1] == probe.shape[0]
            ]
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def match(self, faces: Sequence[DetectedFace], *, threshold: float) -> list[Person]:
        """A `Person` per face. Each name is given to at most one face (the best
        scoring); the others with that name count as not recognised."""
        candidates = [self.scores(face.embedding) for face in faces]
        best_for: dict[str, tuple[int, float]] = {}
        for index, ranked in enumerate(candidates):
            if ranked and ranked[0][1] >= threshold:
                name, score = ranked[0]
                if name not in best_for or score > best_for[name][1]:
                    best_for[name] = (index, score)
        people = []
        for index, (face, ranked) in enumerate(zip(faces, candidates, strict=True)):
            top = ranked[0] if ranked else None
            named = top is not None and best_for.get(top[0], (-1, 0.0))[0] == index
            people.append(
                Person(
                    name=top[0] if named and top is not None else None,
                    score=round(top[1], 3) if top is not None else None,
                    position=position_of(face),
                )
            )
        return people


_COUNT: Final = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


NONE_ENROLLED: Final = "No faces are enrolled yet; the user can ask you to remember a face."


def describe_people(people: Sequence[Person], *, enrolled: bool = True) -> str:
    """One line for the chat model, such as "People in view: Yogesh (left), one person
    I don't recognise." Positions are as the camera sees them. With `enrolled` False
    (the face book is empty) unknown faces are only counted, and the line says that
    nobody is enrolled, so the model offers enrolment instead of saying it cannot
    recognise people."""
    if not people:
        return "People in view: nobody I can make out."
    parts = [f"{p.name} ({p.position})" for p in people if p.name is not None]
    unknown = sum(1 for p in people if p.name is None)
    if unknown:
        count = _COUNT[unknown] if unknown < len(_COUNT) else str(unknown)
        noun = "person" if unknown == 1 else "people"
        parts.append(f"{count} {noun} I don't recognise" if enrolled else f"{count} {noun}")
    line = "People in view: " + ", ".join(parts) + "."
    if not enrolled and unknown:
        line += " " + NONE_ENROLLED
    return line
