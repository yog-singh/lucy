"""Telling romanised Hindi from English in Latin-script text (pure, no model).

The user's Hindi reaches Lucy romanised (from speech recognition), so the script cannot
tell a Hindi message from an English one. This counts common romanised Hindi function
words (and a few very common everyday words) instead. Used by `checks/reply_language`
to score messages and replies.

Limits: it says nothing about whether the Hindi is good; spellings outside the list
count as English; English nouns Hindi speakers use (meeting, weather) count as English,
so Hindi and Hindi-English mixed text look alike (both are Hindi here).
"""

from __future__ import annotations

import re
from typing import Final

HINDI_WORDS: Final = frozenset(
    """
    hai hain ka ki ke ko mein se nahi nahin na kya aap aapka aapki aapke aapko main mai
    aur par pe tha thi hoga hogi honge kar karo karna karen karein karte karti kiya
    raha rahi rahe hoon hun hu ho hota hoti hote yeh ye woh wo voh kal aaj abhi bhi toh ji
    haan han accha acha achha achchha theek thik bahut kuch koi mera meri mere mujhe hum
    humein tum tumhara tumhe apna apni apne liye lekin magar kaise kaisa kaisi kab kahan
    kyun kyon kaun jo sab wala wali wale gaya gayi gaye diya di dena dijiye dekh dekho
    bata batao bataiye bataya sakte sakti sakta chahiye hua hui hue waqt samay din raat
    subah shaam dopahar baje baj ek teen chaar paanch dus bees tees sirf phir fir yahan
    wahan jab tab agar ab aisa aise waise thoda zyada jaldi baad pehle saath iske uske
    inka unka unki unke iska uski jaise matlab bilkul shukriya dhanyavaad namaste
    kaafi rahega rahegi rahenge thodi thode isse usse jata jati jate liya maine aapne
    kehte karke wajah baare wapas
    """.split()
)
"""Common romanised Hindi words. English "me" (Hindi "mein" said short) is left out."""

AMBIGUOUS: Final = frozenset(
    """
    main mai par pe ho jo kal han ab tab di na din teen bees dus tees ek sab hum ye wo
    fir hu hun bata baj
    """.split()
)
"""List words that are also English words, names or common abbreviations ("the main
reason", "Kal Penn", "teen", "par for the course"): they never make text Hindi alone."""

WORD: Final = re.compile(r"[a-z]+(?:'[a-z]+)?")
SURE_AT: Final = 2
"""Unambiguous Hindi words from which a sentence can count as Hindi."""
HINDI_AT: Final = 0.2
"""Share of list words from which a sentence with enough sure words counts as Hindi."""
TOO_SHORT: Final = 4
"""Fewer words than this, without a verdict from the rules above, is too short to tell."""


def words(text: str) -> list[str]:
    """The lower-case Latin words of `text` (digits and other scripts dropped)."""
    return WORD.findall(text.lower())


def hindi_share(text: str) -> float:
    """The share of words in `text` that are on the romanised Hindi list (0 if none)."""
    ws = words(text)
    return sum(w in HINDI_WORDS for w in ws) / len(ws) if ws else 0.0


def latin_language(text: str) -> str | None:
    """'hi' if Latin-script `text` reads as Hindi or Hindi-English mixed, 'en' if as
    English, None if it is too short to tell ("Ok.", "Kal Penn.", a number).

    Hindi needs either at least two unambiguous list words making up at least a fifth
    of the words ("Aapki meeting kal 3 baje hai."), or at least half the words on the
    list with one of them unambiguous ("Haan.", "Ji.", "Sun liya.", "Kal do meetings
    hain."). So an English
    sentence with a loanword or a name ("I had chai with Kal Penn", "The main reason
    is cost") stays English. Otherwise a sentence of fewer than four words is too
    short to tell, and a longer one is English."""
    ws = words(text)
    if not ws:
        return None
    listed = [w for w in ws if w in HINDI_WORDS]
    sure = sum(w not in AMBIGUOUS for w in listed)
    if sure >= SURE_AT and len(listed) / len(ws) >= HINDI_AT:
        return "hi"
    if sure and len(listed) * 2 >= len(ws):
        return "hi"
    return None if len(ws) < TOO_SHORT else "en"
