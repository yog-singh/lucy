"""Deciding whether the user confirmed a dangerous tool call (ADR 0006).

Plain code, no model: a pending request is read from the episode log, and the
user's answer is matched against a short fixed list of whole utterances. Anything
not on the lists, including "ok" and "sure" (which speech recognition produces
from noise), counts as saying something else.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from typing import Final, Literal

from lucy.core.events import (
    AnyEvent,
    ConfirmationOutcome,
    ConfirmationRequested,
    ConfirmationResolved,
    PlaybackInterrupted,
    TurnEnded,
    UserMessage,
)

Answer = Literal["yes", "no", "other"]

YES: Final[frozenset[str]] = frozenset(
    {
        "yes",
        "yes please",
        "yeah",
        "yep",
        "confirm",
        "do it",
        "go ahead",
        "yes do it",
        "yes go ahead",
    }
)
NO: Final[frozenset[str]] = frozenset(
    {"no", "nope", "cancel", "stop", "dont", "never mind", "no thanks"}
)
_FILLERS: Final = ("lucy", "please", "thanks")
_PUNCTUATION: Final = re.compile(f"[{re.escape(string.punctuation)}’‘“”…]")


def normalise(text: str) -> str:
    """Casefolded, without punctuation or extra spaces, and without one leading
    and one trailing "lucy", "please" or "thanks" ("don't" becomes "dont")."""
    words = _PUNCTUATION.sub("", text.casefold()).split()
    if len(words) > 1 and words[0] in _FILLERS:
        words = words[1:]
    if len(words) > 1 and words[-1] in _FILLERS and " ".join(words) not in YES | NO:
        words = words[:-1]
    return " ".join(words)


def classify(text: str) -> Answer:
    """Yes or no only when the whole utterance is one of the fixed answers."""
    said = normalise(text)
    if said in YES:
        return "yes"
    if said in NO:
        return "no"
    return "other"


def pending_confirmation(
    episodes: Sequence[AnyEvent], message: UserMessage
) -> ConfirmationRequested | None:
    """The request `message` answers, if any.

    `episodes` are the session's recent events, oldest first, and may already
    include `message`. A request is pending only if it is the last request, is
    not resolved, `message` is the first user message after it, `message` was
    typed or spoken (not a memory correction), and `message` arrived after the
    turn that asked the question had completed.
    """
    if message.source not in ("text", "voice"):
        return None
    requests = [i for i, e in enumerate(episodes) if isinstance(e, ConfirmationRequested)]
    if not requests:
        return None
    index = requests[-1]
    request = episodes[index]
    assert isinstance(request, ConfirmationRequested)
    later = episodes[index + 1 :]
    if any(isinstance(e, ConfirmationResolved) and e.request_id == request.id for e in later):
        return None
    first_user = next((e for e in later if isinstance(e, UserMessage)), None)
    if first_user is not None and first_user.id != message.id:
        return None  # an earlier message already answered (or ignored) the question
    if first_user is None and any(e.id == message.id for e in episodes[:index]):
        return None  # the message came before the question
    ended = next(
        (
            e
            for e in later
            if isinstance(e, TurnEnded) and e.turn_id == request.turn_id and e.reason == "completed"
        ),
        None,
    )
    if ended is None or message.ts <= ended.ts:
        return None
    return request


def outcome_for(
    request: ConfirmationRequested, episodes: Sequence[AnyEvent], message: UserMessage
) -> ConfirmationOutcome:
    """How `message` resolves `request`, before any re-check of the target: the
    question was cut off, it expired, or the answer's yes, no or something else."""
    if any(isinstance(e, PlaybackInterrupted) and e.turn_id == request.turn_id for e in episodes):
        return "unheard"
    if message.ts > request.expires_at:
        return "expired"
    return {"yes": "confirmed", "no": "declined", "other": "superseded"}[classify(message.text)]
