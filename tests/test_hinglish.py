"""Tests for lucy.core.hinglish: telling romanised Hindi from English."""

from __future__ import annotations

from lucy.core.hinglish import hindi_share, latin_language, words

# Sentences of Lucy's replies saved by checks/reply_language (run "after2"), split as
# the speech player splits them, plus English sentences with Hindi-looking words.
# Expected: "hi" (Hindi or Hindi-English mixed), "en", or None (too short to tell).
HINDI = [
    "Main bilkul theek hoon, shukriya!",
    "Aap kaise hain?",
    "Kabhi kabhi din sach mein bahut heavy ho sakte hain.",
    "Ab thoda relax karo.",
    "Japan ki capital Tokyo hai.",
    "Baarish ka hona ek natural process hai jo water cycle se juda hota hai.",
    "Isse evaporation kehte hain.",
    "In boondon ko clouds ya badal bante hain.",
    "Tokyo mein abhi subah ke 5:46 baj rahe hain.",
    "Delhi mein abhi 24 degrees hain aur thodi haze hai.",
    "Din mein maximum 28 degrees tak ja sakta hai aur raat ko 14 degrees tak gir sakta hai.",
    "Mausam kaafi clear rahega.",
    "Kal, 18 November ko aapke calendar mein teen events hain.",
    "Kal do meetings hain.",
    "Ek Team meeting hai 10am se 11am tak aur dusri Budget meeting hai 1pm se 2pm tak.",
    "Theek hai, main aapko kal subah 7:30 baje yaad dila doongi ki 2 goliyan leni hain.",
    "Dentist appointment Friday 20 November ko 4 baje ke liye add ho gaya hai.",
    "Paris ka khana kaafi famous hai, khaas kar wahan ke pastries aur breads.",
    "French cuisine mein cheese aur wine ka bhi bahut bada role hai.",
    "Pappu: Papa, aaj maine 10 rupaye bacha liye.",
    "Papa: Woh kaise?",
    "Toh phir kya kiya?",
    "Neend nahi aa rahi toh thoda relax karne ki koshish karo.",
    "Thoda sa garam paani ya milk peena bhi accha ho sakta hai.",
    "Agar aap kam chai pite hain aur usme chini kam hai, toh yeh thik hai.",
    "Theek hai.",
    "Haan, kal Delhi mein baarish hone ke chances kaafi kam hain, bas 10 percent chance hai.",
    "Mausam thoda cloudy rahega.",
    "Sun liya.",
    "Team meeting subah 10 baje hai aur Budget meeting dopahar 1 baje hai.",
    'Doctor ne kaha, "Nahi, lekin aapko thoda kam karna chahiye."',
    "Haan.",
    "Ji.",
    "Aapki meeting 18 November ko hai.",
    "Bill 1250 rupaye ka hai.",
]
ENGLISH = [
    "The capital of Japan is Tokyo.",
    "I'm doing well, thank you for asking.",
    "It's 5:44 am in Tokyo.",
    "It is 24 degrees in Delhi today, feeling like 25 degrees.",
    "You have three events on Wednesday, November 18.",
    "I've set a reminder for you to take 2 tablets tomorrow, Wednesday, November 18th, at 7:30 am.",
    "Can you tell me what time you'd like the reminder for tomorrow?",
    "The dentist appointment is scheduled for Friday, November 20th, at 4 pm.",
    "Try to go to bed and wake up at the same time every day, even on weekends.",
    "Let me know if you'd like to talk about anything else.",
    "I had chai with Kal Penn.",
    "The main reason is cost.",
    "Ask Sab about the main hall on par with the teen centre.",
    "Ho Chi Minh City is in Vietnam.",
    "We went to Bodh Gaya and had a namaste from the monks there.",
    "Na, that is the chemical symbol for sodium.",
]
SHORT = ["Sure.", "Ok.", "Done.", "You're welcome.", "No problem.", "I understand.", "7:30."]
CASES = [(s, "hi") for s in HINDI] + [(s, "en") for s in ENGLISH] + [(s, None) for s in SHORT]

KNOWN_MISSES = {
    # English with a quoted Hindi phrase reads as mixed: its numbers would be said in
    # Hindi.
    'The calendar tool didn\'t understand "kal subah 7:30 baje" as a time.': "en",
}


def test_tags_the_saved_replies() -> None:
    wrong = {s: want for s, want in CASES if latin_language(s) != want}
    assert len(CASES) >= 55
    assert wrong == {}, f"{len(CASES) - len(wrong)}/{len(CASES)} right; wrong: {wrong}"


def test_known_misses() -> None:
    for text, want in KNOWN_MISSES.items():
        assert latin_language(text) != want


def test_edges() -> None:
    assert latin_language("") is None
    assert latin_language("1250") is None
    assert latin_language("Kal Penn.") is None  # a name: too short, not Hindi
    assert latin_language("Kal subah.") == "hi"
    assert words("It's 7:30, Ji!") == ["it's", "ji"]
    assert hindi_share("Theek hai, sure.") == 2 / 3
    assert hindi_share("") == 0.0
