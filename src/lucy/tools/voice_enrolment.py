"""The voice enrolment tools (ADR 0021, amendment): `enroll_voice` (act tier) for a
name not yet enrolled, `replace_voice` (dangerous: asks first) for one that is.

Neither records anything. A successful call only REQUESTS an enrolment session: the
voice client sees the call finish (`ToolStarted` / `ToolFinished` on the bus, ADR
0003) and, once the turn has been spoken, leads the session itself in code
(`lucy.voice.enrol`). The result tells the model so, so that it adds nothing.

Outside a voice client, with `[speakers]` off, or without the model weights, the
tools refuse with what the user should do instead.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from lucy.core.config import Settings
from lucy.core.events import TimerFired
from lucy.core.reminders import TimerReply
from lucy.core.types import Risk, ToolOutput
from lucy.core.workspace import WorkspaceError, check_name
from lucy.tools.common import error

ENROL_TOOLS: Final = frozenset({"enroll_voice", "replace_voice"})
ENROLMENT_TIMER: Final = "voice_enrolment"
"""`TimerFired.name` the voice client publishes to have the outcome said and logged."""

MAX_NAME: Final = 32
_PLAIN: Final = re.compile(r"^[A-Za-z][A-Za-z -]{0,31}$")

STARTED: Final = (
    "The voice enrolment for {name} is starting now and Lucy leads it herself in code, "
    "right after your reply. Reply with just 'Okay.' and nothing else: no instructions, "
    "no 'I am ready', no description of the steps."
)
PLACEHOLDERS: Final = frozenset(
    {
        "my", "me", "myself", "mine", "i", "user", "the user", "voice", "your voice",
        "owner", "speaker", "unknown", "guest", "you", "yourself", "lucy", "name",
        "my name", "someone", "somebody", "person", "the person",
    }
)  # fmt: skip
NOT_A_NAME: Final = (
    "{value!r} is not a person's name, so the enrolment was not started. Call this tool "
    "again with the person's actual name: for the user's own voice, their name if it is "
    "in what you know about them; otherwise ask the user what name to enrol the voice "
    "under, and call the tool once they answer."
)


def _is_placeholder(name: str) -> bool:
    folded = name.casefold()
    return (
        folded in PLACEHOLDERS
        or "voice" in folded.split()
        or "voices" in folded.split()
        or folded.startswith(("my ", "the ", "your "))
    )


def clean_name(value: object) -> str:
    """The name as given, trimmed, spaces collapsed, and checked: a short plain name
    (letters, spaces, hyphens), not a placeholder or phrase such as "my voice". An
    all-lowercase name is title-cased ("asha" -> "Asha"); any other capitalisation is
    kept as given ("McKay", "DeShawn"). Raises `ValueError` with a message for the
    model."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("name is required: the name of the person whose voice to enrol")
    name = " ".join(value.split())
    if _is_placeholder(name):
        raise ValueError(NOT_A_NAME.format(value=name))
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


class _EnrolBase:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "The person's name, such as 'Yogesh' or 'Asha'; never 'my voice', 'me' "
                    "or 'user'. For the user's own voice, their actual name."
                ),
            }
        },
        "required": ["name"],
    }

    def __init__(
        self,
        settings: Settings,
        *,
        speakers: Callable[[], dict[str, int]] | None = None,
        voice_client: bool = False,
    ) -> None:
        self._settings = settings
        self._speakers = speakers
        self.voice_client = voice_client
        """Set by the app: whether a voice client (`lucy voice`, `lucy listen`) runs it."""

    def _enrolled(self) -> dict[str, int]:
        if self._speakers is None:
            from lucy.clients.speakers_cli import make_identifier

            self._speakers = make_identifier(self._settings).speakers
        assert self._speakers is not None
        return self._speakers()

    def _existing(self, name: str) -> str | None:
        """The enrolled name matching `name` (ignoring case), if any."""
        return next((n for n in self._enrolled() if n.casefold() == name.casefold()), None)

    def _blocker(self) -> str | None:
        config = self._settings.speakers
        if not config.enabled:
            return (
                "Voice enrolment is off: speaker identification is disabled. The user can "
                "turn it on with enabled = true under [speakers] in lucy.toml, then restart Lucy."
            )
        if not Path(config.weights).is_file():
            return (
                "Voice enrolment needs the speaker model, which is not downloaded. The user "
                "can run `lucy speakers download` in a terminal, then restart Lucy."
            )
        if not self.voice_client:
            return (
                "Voice enrolment needs the microphone: this is a text chat. The user can ask "
                "again in `lucy voice` or `lucy listen`."
            )
        return None

    def _check(self, args: dict[str, Any]) -> tuple[str | None, ToolOutput | None]:
        try:
            name = clean_name(args.get("name"))
        except ValueError as exc:
            return None, error(str(exc))
        blocker = self._blocker()
        if blocker is not None:
            return None, error(blocker)
        return name, None


class EnrollVoiceTool(_EnrolBase):
    """Enrol a new voice (risk tier act): requests the voice client's enrolment session."""

    name = "enroll_voice"
    description = (
        "Start enrolling a person's voice so Lucy can recognise who is speaking. Lucy then "
        "leads the recording herself. Only for a name not enrolled yet."
    )
    risk = Risk.ACT

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        name, refusal = self._check(args)
        if refusal is not None:
            return refusal
        assert name is not None
        existing = self._existing(name)
        if existing is not None:
            return error(
                f"A voice is already enrolled for {existing}. To record it again, call the "
                f"replace_voice tool with name {existing!r}; it asks the user first."
            )
        return ToolOutput(content=STARTED.format(name=name))


class ReplaceVoiceTool(_EnrolBase):
    """Re-enrol an enrolled voice (risk tier dangerous: the user confirms first)."""

    name = "replace_voice"
    description = (
        "Record an already enrolled person's voice again, replacing the old enrolment. "
        "Asks the user first, then Lucy leads the recording herself."
    )
    risk = Risk.DANGEROUS

    async def preview(self, args: dict[str, Any]) -> ToolOutput:
        name, refusal = self._check(args)
        if refusal is not None:
            return refusal
        assert name is not None
        existing = self._existing(name)
        if existing is None:
            return error(
                f"No voice is enrolled for {name}. Call the enroll_voice tool to enrol it."
            )
        return ToolOutput(content=f"Replace the enrolled voice for {existing}?")

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        name, refusal = self._check(args)
        if refusal is not None:
            return refusal
        assert name is not None
        existing = self._existing(name)
        if existing is None:
            return error(f"No voice is enrolled for {name}.")
        return ToolOutput(content=STARTED.format(name=existing))


async def handle_enrolment(event: TimerFired, turn_id: str) -> TimerReply | None:
    """The orchestrator's handler for `ENROLMENT_TIMER`: say (and so log) the outcome
    the voice client worked out."""
    text = event.payload.get("text")
    return TimerReply(text=text) if isinstance(text, str) and text else None
