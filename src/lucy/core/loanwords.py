"""English loanwords common in Hindi speech, with their usual Devanagari spelling.

One table, used in both directions (ADR 0024):

- English to Devanagari (`lucy.core.devanagari`): a Latin word left inside a Hindi
  sentence is respelled for the `hi-IN` voice ("meeting" becomes "मीटिंग").
- Devanagari to English (`lucy.voice.translit`): on screen, a loanword the model wrote
  in Devanagari is shown in its English spelling ("कैलेंडर" as "calendar", not
  "kaileindar"), while real Hindi words are transliterated as before.

`LOANWORDS` maps the lower-case English word to Devanagari. Where several English
spellings share one Devanagari form (ok/okay), `CANONICAL` names the one shown on
screen. `CAPITALISED` lists the words written with a capital (weekdays, months,
names). Words that are also everyday Hindi words are left out, so a Hindi word is
never shown as English: "bus" (बस, "enough"), "mail" (मेल). Function words such as
"the" or "and" are not loanwords: they live in
`devanagari.EXCEPTIONS` only and are never shown back in English.
"""

from __future__ import annotations

from typing import Final

LOANWORDS: Final[dict[str, str]] = {
    # weekdays and months
    "monday": "मंडे", "tuesday": "ट्यूज़डे", "wednesday": "वेडनेसडे", "thursday": "थर्सडे",
    "friday": "फ्राइडे", "saturday": "सैटरडे", "sunday": "संडे", "january": "जनवरी",
    "february": "फ़रवरी", "march": "मार्च", "april": "अप्रैल", "may": "मई", "june": "जून",
    "july": "जुलाई", "august": "अगस्त", "september": "सितंबर", "october": "अक्टूबर",
    "november": "नवंबर", "december": "दिसंबर",
    # names
    "lucy": "लूसी", "google": "गूगल", "gmail": "जीमेल", "whatsapp": "व्हाट्सऐप",
    # time and planning
    "meeting": "मीटिंग", "meetings": "मीटिंग्स", "calendar": "कैलेंडर",
    "reminder": "रिमाइंडर", "reminders": "रिमाइंडर्स", "alarm": "अलार्म", "timer": "टाइमर",
    "time": "टाइम", "schedule": "शेड्यूल", "appointment": "अपॉइंटमेंट", "event": "इवेंट",
    "events": "इवेंट्स", "weekend": "वीकेंड", "deadline": "डेडलाइन", "plan": "प्लान",
    "minute": "मिनट", "minutes": "मिनट्स", "date": "डेट", "week": "वीक",
    "late": "लेट", "free": "फ़्री", "busy": "बिज़ी", "cancel": "कैंसल", "list": "लिस्ट",
    "note": "नोट", "notes": "नोट्स", "task": "टास्क", "update": "अपडेट",
    # communication
    "email": "ईमेल", "message": "मैसेज", "phone": "फ़ोन", "call": "कॉल",
    "mobile": "मोबाइल", "video": "वीडियो", "chat": "चैट", "reply": "रिप्लाई",
    "contact": "कॉन्टैक्ट", "number": "नंबर", "photo": "फ़ोटो",
    # politeness and replies
    "ok": "ओके", "okay": "ओके", "please": "प्लीज़", "sorry": "सॉरी", "thanks": "थैंक्स",
    "thank": "थैंक", "welcome": "वेलकम", "hello": "हेलो", "bye": "बाय", "sure": "श्योर",
    "good": "गुड", "morning": "मॉर्निंग", "night": "नाइट",
    # work and study
    "office": "ऑफ़िस", "boss": "बॉस", "team": "टीम", "project": "प्रोजेक्ट",
    "report": "रिपोर्ट", "presentation": "प्रेज़ेंटेशन", "interview": "इंटरव्यू",
    "job": "जॉब", "work": "वर्क", "school": "स्कूल", "college": "कॉलेज", "class": "क्लास",
    "exam": "एग्ज़ाम", "teacher": "टीचर", "homework": "होमवर्क",
    # health
    "doctor": "डॉक्टर", "dentist": "डेंटिस्ट", "hospital": "हॉस्पिटल",
    "medicine": "मेडिसिन", "tablet": "टैबलेट", "gym": "जिम", "workout": "वर्कआउट",
    "yoga": "योगा", "walk": "वॉक", "checkup": "चेकअप",
    # technology
    "app": "ऐप", "online": "ऑनलाइन", "file": "फ़ाइल", "laptop": "लैपटॉप",
    "computer": "कंप्यूटर", "internet": "इंटरनेट", "wifi": "वाईफ़ाई", "battery": "बैटरी",
    "charge": "चार्ज", "charger": "चार्जर", "password": "पासवर्ड", "link": "लिंक",
    "website": "वेबसाइट", "download": "डाउनलोड", "camera": "कैमरा", "screen": "स्क्रीन",
    # money and shopping
    "bank": "बैंक", "account": "अकाउंट", "payment": "पेमेंट", "bill": "बिल",
    "order": "ऑर्डर", "delivery": "डिलीवरी", "shopping": "शॉपिंग", "card": "कार्ड",
    # travel
    "flight": "फ़्लाइट", "ticket": "टिकट", "train": "ट्रेन", "taxi": "टैक्सी",
    "car": "कार", "station": "स्टेशन", "airport": "एयरपोर्ट", "hotel": "होटल",
    "traffic": "ट्रैफ़िक", "trip": "ट्रिप",
    # home, food and free time
    "weather": "वेदर", "temperature": "टेम्परेचर", "birthday": "बर्थडे", "party": "पार्टी",
    "dinner": "डिनर", "lunch": "लंच", "breakfast": "ब्रेकफ़ास्ट", "coffee": "कॉफ़ी",
    "movie": "मूवी", "song": "सॉन्ग", "music": "म्यूज़िक", "game": "गेम", "match": "मैच",
    "news": "न्यूज़", "book": "बुक", "tv": "टीवी", "room": "रूम", "light": "लाइट",
    "family": "फ़ैमिली", "friend": "फ़्रेंड", "friends": "फ़्रेंड्स",
}  # fmt: skip
"""Lower-case English word to its usual Devanagari spelling in Hindi."""

CANONICAL: Final[dict[str, str]] = {"ओके": "ok"}
"""The English spelling shown for a Devanagari form that several entries share."""

CAPITALISED: Final[frozenset[str]] = frozenset(
    """
    monday tuesday wednesday thursday friday saturday sunday january february march
    april may june july august september october november december lucy google gmail
    whatsapp
    """.split()
)
"""Words shown with a capital first letter (weekdays, months, names)."""

TO_ENGLISH: Final[dict[str, str]] = {}
"""Devanagari to the English spelling shown on screen (capitalised where usual)."""
for _english, _devanagari in LOANWORDS.items():
    _shown = CANONICAL.get(_devanagari, _english)
    TO_ENGLISH[_devanagari] = _shown.capitalize() if _shown in CAPITALISED else _shown
