"""The rules for a person's name, shared by voice and face enrolment and by
`[people] owner` (ADR 0021, 0026, 0027), so one person has one name everywhere."""

from __future__ import annotations

import re
from typing import Final

from lucy.core.workspace import WorkspaceError, check_name

MAX_NAME: Final = 32
_PLAIN: Final = re.compile(r"^[A-Za-z][A-Za-z -]{0,31}$")

PLACEHOLDERS: Final = frozenset(
    {
        "my", "me", "myself", "mine", "i", "user", "the user", "voice", "your voice",
        "owner", "speaker", "unknown", "guest", "you", "yourself", "lucy", "name",
        "my name", "someone", "somebody", "person", "the person", "face", "my face",
        "your face", "chehra",
    }
)  # fmt: skip
NOT_A_NAME: Final = (
    "{value!r} is not a person's name, so the enrolment was not started. Call this tool "
    "again with the person's actual name: for the user's own {what}, their name if it is "
    "in what you know about them; otherwise ask the user what name to enrol the {what} "
    "under, and call the tool once they answer."
)
_THING_WORDS: Final = frozenset({"voice", "voices", "face", "faces"})


def _is_placeholder(name: str) -> bool:
    folded = name.casefold()
    return (
        folded in PLACEHOLDERS
        or not _THING_WORDS.isdisjoint(folded.split())
        or folded.startswith(("my ", "the ", "your "))
    )


def clean_name(value: object, *, what: str = "voice") -> str:
    """The name as given, trimmed, spaces collapsed, and checked: a short plain name
    (letters, spaces, hyphens), not a placeholder or phrase such as "my voice". An
    all-lowercase name is title-cased ("asha" -> "Asha"); any other capitalisation is
    kept as given ("McKay", "DeShawn"). Raises `ValueError` with a message for the
    model. The same rules name voices and faces (`what`), so one person has one name
    for both (ADR 0026)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"name is required: the name of the person whose {what} to enrol")
    name = " ".join(value.split())
    if _is_placeholder(name):
        raise ValueError(NOT_A_NAME.format(value=name, what=what))
    if len(name) > MAX_NAME or not _PLAIN.match(name):
        raise ValueError(
            f"{value!r} is not a usable name: give a short plain name such as 'Asha', "
            "letters and spaces only"
        )
    if name.islower():
        name = name.title()
    try:
        check_name(name)  # it becomes a directory name in the workspace
    except WorkspaceError as exc:
        raise ValueError(str(exc)) from exc
    return name
