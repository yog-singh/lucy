"""The `look` tool (read tier, ADR 0026): Lucy looks through the camera when asked.

It takes one camera frame and has the vision model answer `question` about it
(`lucy.vision.VisionService`); the chat model gets the answer as text. The app
attaches the service (`vision_service`) when `[vision]` is enabled; without it
the tool refuses with what the user should do. With face matching on (stage 2)
the result starts with who is in view, such as "People in view: Yogesh (left),
one person I don't recognise." With nobody enrolled it says so instead ("People in
view: one person. No faces are enrolled yet; the user can ask you to remember a
face."), so the model offers enrolment rather than saying it cannot recognise people.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from lucy.core.config import Settings
from lucy.core.types import Risk, ToolOutput
from lucy.tools.common import error, required_text

if TYPE_CHECKING:
    from lucy.vision.service import VisionService

GENERAL: Final = "Describe what you see."


class LookTool:
    name = "look"
    description = (
        "Look through the camera and say what is visible. Use it whenever the user asks "
        "what you can see, what they are holding or showing, what something in front of "
        "the camera is, or how something looks. Takes one photo; nothing is stored."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What to find out from the picture, in English, such as 'what is the "
                    "user holding' or 'what colour is the shirt'. Leave it out for a "
                    "general description."
                ),
            }
        },
    }
    risk = Risk.READ

    def __init__(self, settings: Settings, *, vision_service: VisionService | None = None) -> None:
        self._settings = settings
        self.vision_service = vision_service
        """Set by the app when `[vision]` is enabled and its role is configured."""

    def _blocker(self) -> str | None:
        config = self._settings.vision
        if not config.enabled:
            return (
                "Looking is off: the camera is not set up. The user can turn it on with "
                "enabled = true under [vision] in lucy.toml, add a 'vision' model role under "
                "[llm.roles], then restart Lucy."
            )
        if config.role not in self._settings.llm.roles:
            return (
                f"Looking needs a model that can see, but [llm.roles] has no {config.role!r} "
                "role. The user can add one in lucy.toml, then restart Lucy."
            )
        if self.vision_service is None:
            return "Looking is not available in this Lucy session."
        return None

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        blocker = self._blocker()
        if blocker is not None:
            return error(blocker)
        assert self.vision_service is not None
        from lucy.vision.faces import describe_people
        from lucy.vision.service import VisionError

        question = required_text(args, "question") or GENERAL
        try:
            sight = await self.vision_service.see(question)
        except VisionError as exc:
            return error(f"Could not look: {exc}")
        seen = f"Seen through the camera just now: {sight.description}"
        if not self.vision_service.faces_ready:
            return ToolOutput(content=seen)
        if sight.faces_note is not None:
            return ToolOutput(content=f"People in view: unknown ({sight.faces_note}).\n{seen}")
        # Positions are as the camera sees them; a match is a tag, never proof of identity.
        book = self.vision_service.book
        enrolled = book is None or bool(book.people())
        people = describe_people(sight.people, enrolled=enrolled)
        return ToolOutput(content=f"{people}\n{seen}")
