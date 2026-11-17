"""The face enrolment tools (ADR 0026, stage 2): `enroll_face` (act tier) for a name not
yet enrolled, `replace_face` (dangerous: asks first) for one that is.

Unlike voice enrolment, the tool does the capture itself: `VisionService.enrol_face`
takes a few frames about half a second apart, each of which must hold exactly one
clear face, and stores only their embeddings (no photo). So it works in `lucy chat`
and in voice alike. Names follow the speaker name rules (`clean_name`), so one person
has one name for voice and face. A face match is a tag, never an authorisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from lucy.core.config import Settings
from lucy.core.types import Risk, ToolOutput
from lucy.tools.common import error
from lucy.tools.voice_enrolment import clean_name

if TYPE_CHECKING:
    from lucy.vision.service import VisionService

FACE_TOOLS: Final = frozenset({"enroll_face", "replace_face"})
VISION_TOOLS: Final = frozenset({"look", *FACE_TOOLS})
"""The tools that use the `VisionService`."""


class _FaceBase:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "The person's name, such as 'Yogesh' or 'Asha'; never 'my face', 'me' "
                    "or 'user'. For the user's own face, their actual name."
                ),
            }
        },
        "required": ["name"],
    }

    def __init__(self, settings: Settings, *, vision_service: VisionService | None = None) -> None:
        self._settings = settings
        self.vision_service = vision_service
        """Set by the app when `[vision]` is enabled."""

    def _blocker(self) -> str | None:
        vision = self._settings.vision
        if not vision.enabled or not vision.faces.enabled:
            return (
                "Face enrolment is off. The user can turn it on with enabled = true under "
                "[vision] and [vision.faces] in lucy.toml, then restart Lucy."
            )
        service = self.vision_service
        if service is None:
            return "Face enrolment is not available in this Lucy session."
        if service.faces_off is not None:
            return f"Face enrolment is not available: {service.faces_off}"
        if not service.faces_ready:
            return "Face enrolment is not available in this Lucy session."
        return None

    def _check(self, args: dict[str, Any]) -> tuple[str | None, ToolOutput | None]:
        try:
            name = clean_name(args.get("name"), what="face")
        except ValueError as exc:
            return None, error(str(exc))
        blocker = self._blocker()
        if blocker is not None:
            return None, error(blocker)
        return name, None

    def _existing(self, name: str) -> str | None:
        assert self.vision_service is not None and self.vision_service.book is not None
        return self.vision_service.book.find(name)

    async def _enrol(self, name: str, *, replaced: bool) -> ToolOutput:
        from lucy.vision.service import VisionError

        assert self.vision_service is not None
        try:
            count = await self.vision_service.enrol_face(name)
        except VisionError as exc:
            return error(f"The face was not enrolled: {exc} Nothing was stored.")
        verb = "replaced" if replaced else "enrolled"
        return ToolOutput(
            content=(
                f"{name}'s face is {verb} ({count} pictures; only the face measurements are "
                "kept, no photos). Lucy will now say who she sees when she looks."
            )
        )


class EnrollFaceTool(_FaceBase):
    """Enrol a new face (risk tier act): captures a few frames now."""

    name = "enroll_face"
    description = (
        "Learn a person's face through the camera so Lucy can say who she sees. Faces are "
        "remembered only with this tool, so when the user asks you to remember, learn or "
        "recognise their face or someone's, such as 'remember my face' or 'mera chehra "
        "yaad rakho', call it; never say you cannot remember faces. For the user's own "
        "face, pass their name from what you know about them, and ask for it only when "
        "you do not know it. Takes a few pictures right away; the person should be alone "
        "in front of the camera, facing it. Only for a name not enrolled yet."
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
                f"A face is already enrolled for {existing}. To take it again, call the "
                f"replace_face tool with name {existing!r}; it asks the user first."
            )
        return await self._enrol(name, replaced=False)


class ReplaceFaceTool(_FaceBase):
    """Re-enrol an enrolled face (risk tier dangerous: the user confirms first)."""

    name = "replace_face"
    description = (
        "Take an already enrolled person's face again, replacing the old enrolment. Asks "
        "the user first, then takes a few pictures right away."
    )
    risk = Risk.DANGEROUS

    async def preview(self, args: dict[str, Any]) -> ToolOutput:
        name, refusal = self._check(args)
        if refusal is not None:
            return refusal
        assert name is not None
        existing = self._existing(name)
        if existing is None:
            return error(f"No face is enrolled for {name}. Call the enroll_face tool to enrol it.")
        return ToolOutput(content=f"Replace the enrolled face for {existing}?")

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        name, refusal = self._check(args)
        if refusal is not None:
            return refusal
        assert name is not None
        existing = self._existing(name)
        if existing is None:
            return error(f"No face is enrolled for {name}.")
        return await self._enrol(existing, replaced=True)
