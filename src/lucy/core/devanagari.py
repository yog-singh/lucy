"""A rough English-to-Devanagari respelling, for Latin words left inside a Hindi sentence.

Magpie TTS with `hi-IN` reads Devanagari well but spells Latin-script words letter by
letter ("M-E-E-T-I-N-G"). The chat model is told to write English loanwords in Devanagari
inside Hindi replies, and mostly does; this module catches the rest just before speech
(ADR 0024). It is not transliteration: a short exception list covers the words most
likely to appear, all-capital words are read letter by letter with Hindi letter names
("PM" becomes "पीएम"), and anything else goes through simple spelling rules (vowel
groups, a silent final e, consonant digraphs, n before a consonant as anusvara), which
gives something close enough to the English sound to beat spelling it out. Only the
speech changes; the text shown and stored stays as written.
"""

from __future__ import annotations

import re
from typing import Final

from lucy.core.loanwords import LOANWORDS

FUNCTION_WORDS: Final[dict[str, str]] = {
    "you": "यू", "the": "द", "and": "एंड", "of": "ऑफ़", "to": "टू", "is": "इज़", "at": "ऐट",
    "in": "इन", "on": "ऑन", "for": "फ़ॉर", "a": "अ", "i": "आई", "am": "एएम", "pm": "पीएम",
}  # fmt: skip
"""English function words and abbreviations: respelled for speech, never shown back."""
EXCEPTIONS: Final[dict[str, str]] = {**LOANWORDS, **FUNCTION_WORDS}

LETTER_NAMES: Final = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (
    "ए बी सी डी ई एफ़ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड"
).split(), strict=True))  # fmt: skip

CONSONANTS: Final = {
    "sh": "श", "ch": "च", "th": "थ", "ph": "फ़", "kh": "ख", "gh": "ग", "ck": "क",
    "qu": "क्व", "wh": "व", "b": "ब", "c": "क", "d": "ड", "f": "फ़", "g": "ग", "h": "ह",
    "j": "ज", "k": "क", "l": "ल", "m": "म", "n": "न", "p": "प", "q": "क", "r": "र",
    "s": "स", "t": "ट", "v": "व", "w": "व", "x": "क्स", "z": "ज़", "y": "य",
}  # fmt: skip
# Vowel groups: (independent form, sign after a consonant; "" is the inherent vowel).
VOWELS: Final = {
    "igh": ("आइ", "ाइ"), "ee": ("ई", "ी"), "ea": ("ई", "ी"), "oo": ("ऊ", "ू"),
    "ai": ("ए", "े"), "ay": ("ए", "े"), "ou": ("आउ", "ाउ"), "ow": ("आउ", "ाउ"),
    "oa": ("ओ", "ो"), "au": ("ऑ", "ॉ"), "aw": ("ऑ", "ॉ"), "oi": ("ऑय", "ॉय"),
    "oy": ("ऑय", "ॉय"), "ie": ("ई", "ी"), "ei": ("ए", "े"), "a": ("ऐ", "ै"),
    "e": ("ए", "े"), "i": ("इ", "ि"), "o": ("ऑ", "ॉ"), "u": ("अ", ""),
}  # fmt: skip
# Long vowels before a consonant and a silent final e ("time", "late", "phone").
MAGIC_E: Final = {"a": ("ए", "े"), "i": ("आइ", "ाइ"), "o": ("ओ", "ो"), "u": ("यू", "्यू"),
                  "e": ("ई", "ी")}  # fmt: skip
HALANT: Final = "्"
LATIN_WORD: Final = re.compile(r"[A-Za-z][A-Za-z']*")
WORD: Final = re.compile(r"[A-Za-zऀ-ॿ][A-Za-z'ऀ-ॿ]*")


def respell_word(word: str) -> str:
    """One Latin word in Devanagari (exception list, letter names, or the rules)."""
    lower = word.lower().replace("'", "")
    if lower in EXCEPTIONS:
        return EXCEPTIONS[lower]
    if word.isupper() and len(word) <= 5:
        return "".join(LETTER_NAMES.get(c, "") for c in word)
    return _rules(lower)


def _rules(w: str) -> str:
    units: list[tuple[str, str]] = []  # ("c", letter) or ("v", group)
    i = 0
    while i < len(w):
        if w[i] == "y" and i > 0 and (i + 1 == len(w) or w[i + 1] not in "aeiou"):
            units.append(("v", "y"))  # a vowel: "happy", "style"
            i += 1
            continue
        for size in (3, 2, 1):
            part = w[i : i + size]
            if part in VOWELS:
                units.append(("v", part))
                break
            if part in CONSONANTS:
                if not (units and units[-1] == ("c", part)):  # "tt" is one t
                    units.append(("c", part))
                break
        else:
            size = 1
        i += size
    # A silent final e after vowel-consonant lengthens the vowel.
    if len(units) >= 3 and units[-1] == ("v", "e") and units[-2][0] == "c" and units[-3][0] == "v":
        units.pop()
        if units[-2][1] in MAGIC_E:
            units[-2] = ("V", units[-2][1])
    out: list[str] = []
    for k, (kind, part) in enumerate(units):
        prev = units[k - 1] if k else None
        nxt = units[k + 1] if k + 1 < len(units) else None
        if kind == "c":
            if part in "nm" and nxt and nxt[0] == "c" and prev and prev[0] != "c":
                out.append("ं")
                continue
            letter = CONSONANTS[part]
            if part == "c" and nxt and nxt[1][:1] in ("e", "i", "y"):
                letter = "स"
            out.append(letter)
            if nxt and nxt[0] == "c":
                out.append(HALANT)
            continue
        after_consonant = prev is not None and prev[0] == "c"
        if part == "y":
            out.append("ी" if after_consonant else "ई")
            continue
        if part == "e" and after_consonant and nxt == ("c", "r") and k + 2 == len(units):
            continue  # "-er" is the inherent vowel: "teacher"
        independent, sign = MAGIC_E[part] if kind == "V" else VOWELS[part]
        out.append(sign if after_consonant else independent)
    return "".join(out)


def respell_latin_words(text: str) -> tuple[str, int, int]:
    """`text` with each Latin word respelled in Devanagari, the number of Latin words,
    and the number of words (Latin or Devanagari; digits are left alone and not counted)."""
    latin = sum(1 for _ in LATIN_WORD.finditer(text))
    total = sum(1 for _ in WORD.finditer(text))
    return LATIN_WORD.sub(lambda m: respell_word(m.group()), text), latin, total


# --- Romanised Hindi to Devanagari (a fallback for replies the model romanised) ---
#
# The persona asks for Hindi replies in Devanagari, but some still come back in Latin
# letters ("Kal aapke calendar mein teen events hain."). Sent as `en-US`, Magpie reads
# them as English words; sent as `hi-IN`, it spells them letter by letter. So a
# Latin-script sentence that `lucy.core.hinglish.latin_language` judges to be Hindi is
# converted here and sent as `hi-IN` (ADR 0024). Per word: a table of common romanised
# Hindi words with their usual spelling variants; then English loanwords (the exception
# list); then a capitalised word inside the sentence, or one that looks English (c, q, x,
# "-tion", "-ing"), through the English rules above; anything else through simple
# romanised-Hindi rules (aa, ee, oo, ai, au, kh, gh, ch, chh, th, dh, sh, bh, ph, a final
# a as आ, an n before a consonant or at the end after a long vowel as anusvara). Latin
# letters cannot tell dental from retroflex consonants (t, d) or always show vowel
# length, so the rules guess dental and the spelling as written: imperfect by design.

ROMAN_HINDI: Final[dict[str, str]] = {
    w: d
    for line in """
    hai है|hain हैं|hei है|ka का|ki की|ke के|ko को|mein में|mai मैं|main मैं|se से|aur और
    ya या|tak तक|ho हो|koi कोई|toh तो|to तो|liye लिए|liya लिया|kabhi कभी|kar कर|bahut बहुत
    bohot बहुत|boht बहुत|bahot बहुत|bht बहुत|sakte सकते|sakta सकता|sakti सकती|thodi थोड़ी
    thoda थोड़ा|thode थोड़े|kuch कुछ|kuchh कुछ|ek एक|der देर|baarish बारिश|barish बारिश
    jab जब|par पर|pe पे|kal कल|baat बात|baatein बातें|baaten बातें|nahi नहीं|nahin नहीं
    nhi नहीं|nai नहीं|din दिन|bhari भारी|abhi अभी|phir फिर|fir फिर|theek ठीक|thik ठीक
    subah सुबह|subha सुबह|jaise जैसे|jaisa जैसा|jaisi जैसी|zyada ज़्यादा|jyada ज़्यादा
    ziada ज़्यादा|ab अब|halki हल्की|halka हल्का|halke हल्के|isse इससे|usse उससे
    sukoon सुकून|sukun सुकून|hoon हूँ|hun हूँ|hu हूँ|ye ये|yeh यह|wo वो|woh वह|vo वो
    voh वह|kaafi काफ़ी|kafi काफ़ी|wahan वहाँ|wahaan वहाँ|yahan यहाँ|yahaan यहाँ|raha रहा
    rahi रही|rahe रहे|kya क्या|neend नींद|nind नींद|peena पीना|pee पी|peete पीते
    peene पीने|piyein पिएँ|jate जाते|jaate जाते|jata जाता|jaata जाता|jati जाती|jaati जाती
    ja जा|jao जाओ|jaiye जाइए|jaaiye जाइए|dopahar दोपहर|bhi भी|kaise कैसे|kaisa कैसा
    kaisi कैसी|hone होने|hona होना|hota होता|hoti होती|hote होते|hoga होगा|hogi होगी
    honge होंगे|bhap भाप|bhaap भाप|ban बन|upar ऊपर|hum हम|aapke आपके|aapka आपका
    aapki आपकी|aapko आपको|aap आप|aapne आपने|lekin लेकिन|dimaag दिमाग़|dimag दिमाग़
    agar अगर|kam कम|sach सच|door दूर|dur दूर|ise इसे|use उसे|kehte कहते|kahte कहते
    hawa हवा|badal बादल|baadal बादल|pani पानी|paani पानी|saath साथ|sath साथ|lagti लगती
    lagta लगता|lagte लगते|lag लग|aaj आज|aj आज|chal चल|ne ने|karta करता|karti करती
    karte करते|karo करो|karna करना|karni करनी|karne करने|karen करें|karein करें
    karenge करेंगे|kiya किया|ki की|wajah वजह|vajah वजह|si सी|sa सा|aksar अक्सर
    milta मिलता|milti मिलती|mil मिल|milkar मिलकर|karke करके|suraj सूरज|sooraj सूरज
    garmi गर्मी|nadiyon नदियों|samundaron समुंदरों|udta उड़ता|ud उड़|thandi ठंडी|thand ठंड
    thanda ठंडा|boondein बूँदें|badi बड़ी|bada बड़ा|bade बड़े|zameen ज़मीन|jameen ज़मीन
    girne गिरने|gir गिर|teen तीन|sham शाम|shaam शाम|maine मैंने|diya दिया|di दी|unka उनका
    unki उनकी|unke उनके|khana खाना|log लोग|bataiye बताइए|batao बताओ|bata बता|bataya बताया
    batayein बताएँ|bataun बताऊँ|padhna पढ़ना|padh पढ़|sunna सुनना|sun सुन|suno सुनो|jo जो
    kisi किसी|yaad याद|sehat सेहत|accha अच्छा|acha अच्छा|achha अच्छा|achchha अच्छा
    acchi अच्छी|achhi अच्छी|achchhi अच्छी|acche अच्छे|achhe अच्छे|is इस|us उस|kitni कितनी
    kitna कितना|kitne कितने|dikkat दिक़्क़त|cheeni चीनी|chini चीनी|dood दूध|doodh दूध
    khaas ख़ास|khas ख़ास|chahiye चाहिए|chaahiye चाहिए|pet पेट|puchna पूछना|pucha पूछा
    poochna पूछना|chote छोटे|chhote छोटे|chhota छोटा|chota छोटा|chhoti छोटी|choti छोटी
    baar बार|aadmi आदमी|dhyan ध्यान|dhyaan ध्यान|kijiye कीजिए|bajkar बजकर|baje बजे
    baj बज|hue हुए|hua हुआ|hui हुई|garam गरम|rakh रख|rakhen रखें|rakhein रखें|rakhni रखनी
    rakhte रखते|rakho रखो|dekhiye देखिए|dekho देखो|dekh देख|aankhein आँखें|aankhen आँखें
    band बंद|bandh बंद|lete लेते|lene लेने|le ले|lo लो|shukriya शुक्रिया|bhara भरा
    mukhya मुख्य|kaaran कारण|karan कारण|unme उनमें|unmein उनमें|maujood मौजूद|jud जुड़
    jise जिसे|do दो|jaante जानते|jaanta जानता|jaanti जानती|mere मेरे|mera मेरा|meri मेरी
    yaar यार|toot टूट|gaya गया|gayi गई|gaye गए|karoge करोगे|daal डाल|daalne डालने
    dunga दूँगा|dungi दूँगी|doonga दूँगा|aaramdayak आरामदायक|kitab किताब|kitaab किताब
    rehna रहना|rahna रहना|fayde फ़ायदे|fayda फ़ायदा|faayda फ़ायदा|kaam काम|shayad शायद
    lekar लेकर|soch सोच|likh लिख|padegi पड़ेगी|padega पड़ेगा|padta पड़ता|saans साँस
    chahte चाहते|chahti चाहती|chahen चाहें|awaaz आवाज़|awaz आवाज़|kahani कहानी
    kahaani कहानी|sunau सुनाऊँ|sunaun सुनाऊँ|sunao सुनाओ|iska इसका|iski इसकी|iske इसके
    uska उसका|uski उसकी|uske उसके|inka इनका|tab तब|wese वैसे|waise वैसे|cheezon चीज़ों
    cheez चीज़|parhez परहेज़|khali ख़ाली|khaali ख़ाली|kadak कड़क|kyunki क्योंकि
    kyonki क्योंकि|kyuki क्योंकि|jalan जलन|hisaab हिसाब|hisab हिसाब|rahegi रहेगी
    rahega रहेगा|rahenge रहेंगे|sirf सिर्फ़|niche नीचे|neeche नीचे|isi इसी|raat रात|rat रात
    mausam मौसम|mausum मौसम|saaf साफ़|saf साफ़|dila दिला|bilkul बिल्कुल|sahab साहब
    kaha कहा|bas बस|zaroorat ज़रूरत|zarurat ज़रूरत|rasta रास्ता|dikh दिख|aa आ|wali वाली
    wala वाला|wale वाले|aise ऐसे|aisa ऐसा|aisi ऐसी|de दे|saari सारी|sari सारी|sara सारा
    isliye इसलिए|tha था|thi थी|the थे|jiski जिसकी|jiska जिसका|jaag जाग|zyadatar ज़्यादातर
    nuksan नुक़सान|nuksaan नुक़सान|kami कमी|asar असर|honi होनी|ji जी|haan हाँ|han हाँ
    ha हाँ|lagbhag लगभग|kab कब|kahan कहाँ|kahaan कहाँ|kyun क्यों|kyon क्यों|kyu क्यों
    kaun कौन|sab सब|apna अपना|apni अपनी|apne अपने|tum तुम|tumhe तुम्हें|tumhara तुम्हारा
    mujhe मुझे|mujhse मुझसे|humein हमें|hume हमें|magar मगर|waqt वक़्त|samay समय|dus दस
    das दस|paanch पाँच|panch पाँच|chaar चार|char चार|saat सात|aath आठ|nau नौ|chhe छह
    bees बीस|tees तीस|sau सौ|hazaar हज़ार|hazar हज़ार|pehle पहले|pahle पहले|baad बाद
    jaldi जल्दी|matlab मतलब|dhanyavaad धन्यवाद|dhanyavad धन्यवाद|namaste नमस्ते
    wapas वापस|vapas वापस|baare बारे|bare बारे|ghar घर|paas पास|pas पास|sahi सही
    galat ग़लत|khush ख़ुश|dost दोस्त|naam नाम|hafte हफ़्ते|hafta हफ़्ता|mahina महीना
    mahine महीने|saal साल|parso परसों|parson परसों|roz रोज़|har हर|sabse सबसे|aage आगे
    peeche पीछे|andar अंदर|bahar बाहर|khatam ख़त्म|khatm ख़त्म|shuru शुरू|yahi यही
    wahi वही|sabhi सभी|dobara दोबारा|zaroor ज़रूर|zarur ज़रूर|dena देना|dijiye दीजिए
    doon दूँ|kahin कहीं|hi ही|na ना|mat मत|ji जी|dikhao दिखाओ|dikha दिखा|dikhaiye दिखाइए
    yaad याद|dilana दिलाना|goliyan गोलियाँ|goli गोली|leni लेनी|lena लेना|thaka थका
    thaki थकी|thake थके|wahaan वहाँ|chalo चलो|chaliye चलिए|socha सोचा|samajh समझ
    pata पता|pta पता|sakoge सकोगे|sakenge सकेंगे|pyaar प्यार|pyar प्यार|zindagi ज़िंदगी
    duniya दुनिया|sawaal सवाल|sawal सवाल|jawab जवाब|jawaab जवाब|madad मदद|baje बजे
    ghante घंटे|ghanta घंटा|minat मिनट|aadha आधा|sawa सवा|dedh डेढ़|dhai ढाई|paune पौने
    bajne बजने|subeh सुबह|raaste रास्ते|khaana खाना|peene पीने|paani पानी|chai चाय
    chaay चाय|chay चाय|doodh दूध|roti रोटी|sabzi सब्ज़ी|aam आम|kela केला|bachche बच्चे
    bacche बच्चे|maa माँ|papa पापा|bhai भाई|behen बहन|beta बेटा|beti बेटी|ladka लड़का
    ladki लड़की|aadat आदत|tabiyat तबीयत|tabiyat तबीयत|dard दर्द|sar सिर|sir सिर
    bimaar बीमार|bimar बीमार|dawai दवाई|dawa दवा|aaram आराम|araam आराम|neeche नीचे
    zameen ज़मीन|aasmaan आसमान|dhoop धूप|garmiyon गर्मियों|sardi सर्दी|sardiyon सर्दियों
    barsaat बरसात|baarishein बारिशें|hawayein हवाएँ|khaas ख़ास|naya नया|nayi नई|naye नए
    purana पुराना|purani पुरानी|bada बड़ा|lamba लंबा|lambi लंबी|chhutti छुट्टी
    chutti छुट्टी|kaamkaaj कामकाज|seekh सीख|sikh सीख|padhai पढ़ाई|likhna लिखना
    bolna बोलना|bolo बोलो|boliye बोलिए|kaho कहो|kahiye कहिए|suniye सुनिए|samjha समझा
    samjhe समझे|samjhi समझी|lagega लगेगा|lagegi लगेगी|achanak अचानक|shaayad शायद
    japan जापान|paris पेरिस|delhi दिल्ली|dilli दिल्ली|tokyo टोक्यो|mumbai मुंबई|bombay बॉम्बे
    france फ़्रांस|india इंडिया|bharat भारत|london लंदन|america अमेरिका|kolkata कोलकाता
    bangalore बैंगलोर|chennai चेन्नई|pune पुणे|santa संता|banta बंता|celsius सेल्सियस
    """.split("\n")
    for pair in line.split("|")
    if pair.strip()
    for w, d in [pair.split()]
}  # fmt: skip
"""Common romanised Hindi words (with spelling variants) and their usual spelling."""

ROMAN_CONSONANTS: Final = {
    "chh": "छ", "cch": "च्छ", "kh": "ख", "gh": "घ", "ch": "च", "jh": "झ", "th": "थ",
    "dh": "ध", "sh": "श", "bh": "भ", "ph": "फ", "rh": "ढ़", "k": "क", "g": "ग", "j": "ज",
    "t": "त", "d": "द", "n": "न", "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र",
    "l": "ल", "v": "व", "w": "व", "s": "स", "h": "ह", "f": "फ़", "z": "ज़", "q": "क़",
    "x": "क्स", "c": "क",
}  # fmt: skip
ROMAN_VOWELS: Final = {
    "aa": ("आ", "ा"), "ai": ("ऐ", "ै"), "au": ("औ", "ौ"), "ee": ("ई", "ी"), "ii": ("ई", "ी"),
    "oo": ("ऊ", "ू"), "uu": ("ऊ", "ू"), "ei": ("ए", "े"), "a": ("अ", ""), "i": ("इ", "ि"),
    "u": ("उ", "ु"), "e": ("ए", "े"), "o": ("ओ", "ो"),
}  # fmt: skip
LONG_FINAL: Final = {"a": ("आ", "ा"), "i": ("ई", "ी"), "u": ("ऊ", "ू")}
"""A word-final a, i or u is long in romanised Hindi: "tha" था, "thi" थी, "tu" तू."""
ENGLISH_LOOK: Final = re.compile(
    r"c(?!h)|q|x|tion|sion|ck|ing$|ea|ou|[^aeiou]y$|w[^aeiouh]|ies$|ent$|er$|or$"
    r"|gh$|w$|([b-df-hj-np-tv-z])\1$|(?<![aeiou])[aeiou]{1,2}[^aeiouhy]e$"
)
"""Spellings that romanised Hindi rarely uses: such a word is respelled as English."""


def roman_word(word: str) -> str:
    """One romanised word in Devanagari: the Hindi table, then English loanwords and
    English-looking words by the English rules, otherwise the romanised-Hindi rules."""
    lower = word.lower().replace("'", "")
    if lower in ROMAN_HINDI:
        return ROMAN_HINDI[lower]
    if lower in EXCEPTIONS or (word.isupper() and len(word) > 1):
        return respell_word(word)
    if ENGLISH_LOOK.search(lower):
        return respell_word(word)
    return _roman_rules(lower)


def _roman_rules(w: str) -> str:
    units: list[tuple[str, str]] = []  # ("c", group) or ("v", group)
    i = 0
    while i < len(w):
        for size in (3, 2, 1):
            part = w[i : i + size]
            if part in ROMAN_VOWELS and size < 3:
                units.append(("v", part))
                break
            if part in ROMAN_CONSONANTS:
                units.append(("c", part))
                break
        else:
            size = 1
        i += size
    out: list[str] = []
    for k, (kind, part) in enumerate(units):
        prev = units[k - 1] if k else None
        nxt = units[k + 1] if k + 1 < len(units) else None
        last = k + 1 == len(units)
        if kind == "c":
            # n (or m before b, p) after a vowel and before a consonant: anusvara.
            if part == "n" and prev and prev[0] == "v" and nxt and nxt[0] == "c":
                out.append("ं")
                continue
            if part == "n" and last and prev and prev[0] == "v" and prev[1] not in ("a", "i", "u"):
                out.append("ं")  # "hain", "mein", "hoon"
                continue
            if part == "m" and prev and prev[0] == "v" and nxt and nxt[1] in ("b", "p", "bh", "ph"):
                out.append("ं")
                continue
            out.append(ROMAN_CONSONANTS[part])
            if nxt and nxt[0] == "c":
                out.append(HALANT)
            continue
        after_consonant = prev is not None and prev[0] == "c"
        if last and part in LONG_FINAL and k > 0:
            independent, sign = LONG_FINAL[part]
        else:
            independent, sign = ROMAN_VOWELS[part]
        out.append(sign if after_consonant else independent)
    return "".join(out)


def romanised_to_devanagari(text: str) -> str:
    """Romanised Hindi `text` in Devanagari, word by word (`roman_word`); digits,
    punctuation and spacing are left as they are."""
    return LATIN_WORD.sub(lambda m: roman_word(m.group()), text)
