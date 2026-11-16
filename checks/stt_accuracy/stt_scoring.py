"""Scoring for the speech recognition accuracy check: text normalisation, word error
rate, the truth file and matching enrolment clips to their pool sentences.

No model is loaded here, so the unit tests (tests/test_check_stt_accuracy.py) can
import it freely.

Normalisation, applied identically to the truth and to every recogniser's output
before any scoring:

1. Lower case; curly apostrophes become straight ones.
2. "%" becomes " percent", "&" becomes " and ", hyphens, slashes and colons become
   spaces (so "7:30" is "7 30" and "well-known" is "well known").
3. Every other character except letters, digits and apostrophes inside a word is
   dropped (punctuation does not count). "Lucy's" stays one word.
4. Number words become digits: "twenty five" -> "25", "two hundred" -> "200",
   "three thousand four hundred" -> "3400", "first" -> "1st", "fifth" -> "5th".
   Number words join only where English grammar joins them (tens before units, a
   multiplier after a number), so "seven thirty" is "7 30", matching "7:30". "and"
   is never absorbed ("one hundred and five" is "100 and 5"). Digits with commas
   ("1,500") lose the commas.
5. A short list of spelling variants (SPELLING) is mapped to one form, mostly
   American to British, because Parakeet writes American spelling and the truth
   is written in Indian/British English ("enroll" -> "enrol").

Word error rate reuses `distance` from checks/voice_e2e/wer.py (Levenshtein over
words); corpus WER is total edits over total truth words.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "voice_e2e"))
from wer import distance  # noqa: E402

UNITS = {
    "zero": 0, "oh": None, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}  # fmt: skip
UNITS.pop("oh")  # "oh" is too often the interjection; left as a word
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100_000, "million": 1_000_000,
          "crore": 10_000_000, "billion": 1_000_000_000}  # fmt: skip
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
    "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
    "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "thirtieth": 30,
}  # fmt: skip
SPELLING = {
    "enroll": "enrol", "enrolls": "enrols",
    "neighbor": "neighbour", "neighbors": "neighbours", "color": "colour",
    "colors": "colours", "favorite": "favourite", "center": "centre", "meter": "metre",
    "kilometer": "kilometre", "theater": "theatre", "organize": "organise",
    "realize": "realise", "recognize": "recognise", "program": "programme",
    "ok": "okay", "gray": "grey",
}  # fmt: skip


def _suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _numbers(words: list[str]) -> list[str]:
    """Replace runs of number words with digits (see the module docstring)."""
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w not in UNITS and w not in TENS and w not in ORDINALS:
            out.append(w)
            i += 1
            continue
        total, current = 0, 0
        last = None  # "unit", "teen", "tens", "scale"
        ordinal = False
        j = i
        while j < len(words):
            x = words[j]
            if x in ORDINALS:
                v = ORDINALS[x]
                if last in (None, "scale") or (last == "tens" and v < 10):
                    current += v
                    ordinal = True
                    j += 1
                break
            if x in TENS and last in (None, "scale"):
                current += TENS[x]
                last = "tens"
            elif x in UNITS and (last in (None, "scale") or (last == "tens" and UNITS[x] < 10)):
                current += UNITS[x]
                last = "unit" if UNITS[x] < 10 else "teen"
            elif x in SCALES and last in ("unit", "teen", "tens"):
                scale = SCALES[x]
                if scale == 100:
                    current *= 100
                else:
                    total += current * scale
                    current = 0
                last = "scale"
            else:
                break
            j += 1
        n = total + current
        out.append(f"{n}{_suffix(n)}" if ordinal else str(n))
        i = j
    return out


def normalise(text: str) -> list[str]:
    """The words of `text` after normalisation (see the module docstring)."""
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = t.replace("%", " percent").replace("&", " and ")
    t = re.sub(r"[-/:–—]", " ", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    words = [w.strip("'") for w in t.split()]
    words = [SPELLING.get(w, w) for w in words if w]
    return _numbers(words)


def normal_text(text: str) -> str:
    return " ".join(normalise(text))


def edits(truth: str, output: str) -> tuple[int, int]:
    """(word edits, truth word count) after normalisation."""
    ref = normalise(truth)
    return distance(ref, normalise(output)), len(ref)


def wer(truth: str, output: str) -> float:
    e, n = edits(truth, output)
    return e / max(1, n)


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> float:
    """Total edits over total truth words for (truth, output) pairs."""
    total_e = total_n = 0
    for truth, output in pairs:
        e, n = edits(truth, output)
        total_e, total_n = total_e + e, total_n + n
    return total_e / max(1, total_n)


def exact(truth: str, output: str) -> bool:
    return normalise(truth) == normalise(output)


def length_bucket(seconds: float) -> str:
    if seconds < 2.0:
        return "<2s"
    return "2-5s" if seconds <= 5.0 else ">5s"


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100); nan for no values."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(-(-q * len(s) // 100)) - 1))
    return s[k]


# --- Indic scoring ---------------------------------------------------------------
#
# For Hindi (and other Indic) references, applied identically to reference and output:
#
# 1. Unicode NFD, then nukta signs dropped (क़ = क, ड़ = ड: annotators and models
#    disagree on them) and chandrabindu written as anusvara (हाँ = हां), then NFC.
# 2. Digits of any script become ASCII digits (१५ = 15). Number words are NOT turned
#    into digits ("पंद्रह" against "15" is an error); the Monsoon lattices often list
#    both forms, which covers most cases.
# 3. Latin letters lower case; zero-width joiners and non-joiners dropped; every other
#    character that is not a letter, combining mark or digit (punctuation, danda,
#    symbols) becomes a space.
#
# References are lattices: a list of spans, each a list of accepted written forms
# (one form may be several words or none). Word edits are the minimum Levenshtein
# distance over every path through the lattice (as OIWER does); the reference word
# count is that of the best path. Two reference sets are scored:
#   "lattice": every accepted form, as the dataset intends: Latin-script English
#              spellings ("welcome" for वेलकम) and romanised Hindi ("hai") both count.
#   "native":  forms containing Latin letters are dropped from a span that also has a
#              non-Latin form, so only native-script output scores well; a span with
#              only Latin forms keeps them. An English word in Latin script in the
#              output therefore counts as an error here even where the speaker said
#              that English word.
# Character error rate: character (code point) edits between the output and the best
# word path's text, with spaces removed (so joining or splitting compounds costs
# nothing), over the reference's characters.

_NUKTAS = {"़", "়", "਼", "઼", "଼", "఼", "಼"}
_CHANDRABINDU = {"ँ": "ं", "ঁ": "ং", "ఁ": "ం", "ಁ": "ಂ"}
_LATIN = re.compile(r"[A-Za-z]")


def indic_normalise(text: str) -> list[str]:
    """The words of `text` after the Indic normalisation (see above)."""
    import unicodedata

    t = unicodedata.normalize("NFD", text)
    out = []
    for ch in t:
        if ch in _NUKTAS:
            continue
        ch = _CHANDRABINDU.get(ch, ch)
        cat = unicodedata.category(ch)
        if cat == "Nd":
            out.append(str(unicodedata.digit(ch)))
        elif cat[0] in "LM":
            out.append(ch.lower())
        elif cat == "Cf":  # zero-width joiner and non-joiner: dropped, not a break
            continue
        else:
            out.append(" ")
    return unicodedata.normalize("NFC", "".join(out)).split()


Lattice = list[list[str]]


def text_lattice(text: str) -> Lattice:
    """A plain reference string as a lattice: one span per word, one form each."""
    return [[w] for w in indic_normalise(text)]


def native_lattice(lattice: Lattice) -> Lattice:
    """Drop Latin-script forms from spans that also have a non-Latin form."""
    out = []
    for span in lattice:
        native = [f for f in span if not _LATIN.search(f)]
        out.append(native or list(span))
    return out


def _lev_row(ref: Sequence[str], hyp: Sequence[str]) -> list[int]:
    """Edit distance of `ref` against every prefix of `hyp` (len(hyp) + 1 values)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev


def lattice_align(lattice: Lattice, output: str) -> tuple[int, list[str]]:
    """(word edits, best reference path's words) of `output` against `lattice`."""
    hyp = indic_normalise(output)
    n = len(hyp)
    inf = 10**9
    # best[j]: (edits, reference words) for the spans so far against hyp[:j]
    best: list[tuple[int, list[str]]] = [(j, []) for j in range(n + 1)]
    for span in lattice:
        forms = {tuple(indic_normalise(f)) for f in span} or {()}
        nxt: list[tuple[int, list[str]]] = [(inf, [])] * (n + 1)
        for form in forms:
            for j0 in range(n + 1):
                base, words = best[j0]
                if base >= inf:
                    continue
                row = _lev_row(form, hyp[j0:])
                for b, cost in enumerate(row):
                    if base + cost < nxt[j0 + b][0]:
                        nxt[j0 + b] = (base + cost, words + list(form))
        best = nxt
    return best[n]


def char_distance(a: str, b: str) -> int:
    return _lev_row(list(a), list(b))[-1]


@dataclass(frozen=True)
class IndicScore:
    word_edits: int
    ref_words: int
    char_edits: int
    ref_chars: int

    @property
    def exact(self) -> bool:
        return self.word_edits == 0


def indic_score(lattice: Lattice, output: str) -> IndicScore:
    """Word and character edits of `output` against `lattice` (see above)."""
    e, ref = lattice_align(lattice, output)
    ref_chars = "".join(ref)
    hyp_chars = "".join(indic_normalise(output))
    return IndicScore(e, len(ref), char_distance(ref_chars, hyp_chars), len(ref_chars))


# Romanised scoring, for configurations that must write Hindi in Latin script: every
# form of every span and the output go through translit.fold (Devanagari written in
# Latin, then the spelling fold for romanised Hindi; see translit.py), and the folded
# lattice is scored as above. With fold=False only the transliteration is applied.


def roman_lattice(lattice: Lattice, fold: bool = True) -> Lattice:
    from translit import fold as _fold
    from translit import to_latin

    f = _fold if fold else (lambda t: " ".join(indic_normalise(to_latin(t))))
    return [sorted({f(form) for form in span}) for span in lattice]


def roman_score(lattice: Lattice, output: str, fold: bool = True) -> IndicScore:
    """`indic_score` with reference forms and output romanised (and folded)."""
    from translit import fold as _fold
    from translit import to_latin

    out = _fold(output) if fold else " ".join(indic_normalise(to_latin(output)))
    return indic_score(roman_lattice(lattice, fold), out)


def has_latin(text: str) -> bool:
    return bool(_LATIN.search(text))


# --- Truth file ----------------------------------------------------------------


@dataclass(frozen=True)
class Truth:
    """One entry of truth.jsonl. `clip` is relative to the eval directory. A skipped
    clip has `truth` None and is not scored."""

    clip: str
    truth: str | None
    skipped: bool = False
    reason: str = ""
    source: str = "label"  # "label" (label.py) or "enrol_auto" (auto_enrol.py)
    heard: str | None = None
    lattice: list[list[str]] | None = None  # Indic references: accepted forms per span
    lang: str = "en"


def load_truth(path: Path) -> dict[str, Truth]:
    """Entries by clip; a later line for the same clip replaces an earlier one."""
    out: dict[str, Truth] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["clip"]] = Truth(**d)
    return out


def append_truth(path: Path, entry: Truth) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")
        f.flush()


# --- Enrolment matching ---------------------------------------------------------

MAX_MATCH_WER = 0.5
"""A transcript further than this from its best pool sentence is not matched."""
MIN_MARGIN = 0.2
"""The best sentence must beat the second best by at least this much WER."""


@dataclass(frozen=True)
class EnrolMatch:
    clip: str
    heard: str
    sentence: str | None  # None: ambiguous, left unlabelled
    score: float  # WER of the transcript against the best sentence
    margin: float  # second-best WER minus best WER
    why: str = ""


def match_enrolment(heard: dict[str, str], pool: Sequence[str]) -> list[EnrolMatch]:
    """Match each clip's transcript to the closest pool sentence by WER. A match is
    refused when the best WER exceeds MAX_MATCH_WER, the margin over the second best
    is below MIN_MARGIN, or two clips pick the same sentence."""
    first: list[EnrolMatch] = []
    for clip, text in heard.items():
        scored = sorted((wer(s, text), s) for s in pool)
        best, sentence = scored[0]
        margin = scored[1][0] - best if len(scored) > 1 else 1.0
        why = ""
        if best > MAX_MATCH_WER:
            why = f"best WER {best:.2f} > {MAX_MATCH_WER}"
        elif margin < MIN_MARGIN:
            why = f"margin {margin:.2f} < {MIN_MARGIN}"
        first.append(EnrolMatch(clip, text, None if why else sentence, best, margin, why))
    picked = [m.sentence for m in first if m.sentence]
    out = []
    for m in first:
        if m.sentence and picked.count(m.sentence) > 1:
            m = EnrolMatch(m.clip, m.heard, None, m.score, m.margin, "sentence matched twice")
        out.append(m)
    return out


# --- Clips -----------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    clip: str  # relative to the eval directory, e.g. "clips/<time>_<id>.wav"
    heard: str | None  # what Lucy recognised live, if recorded


def list_clips(eval_dir: Path) -> list[Clip]:
    """The live clips (clips/*.wav, in time order, with heard.jsonl's text where the
    message id matches) followed by the enrolment clips (enrol_*.wav)."""
    heard: dict[str, str] = {}
    heard_file = eval_dir / "heard.jsonl"
    if heard_file.exists():
        for line in heard_file.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                heard[d["id"]] = d.get("heard") or ""
    out = []
    for path in sorted((eval_dir / "clips").glob("*.wav")):
        message_id = path.stem.split("_", 1)[-1]
        out.append(Clip(f"clips/{path.name}", heard.get(message_id)))
    out += [Clip(p.name, None) for p in sorted(eval_dir.glob("enrol_*.wav"))]
    return out


def side_by_side(rows: Sequence[dict], outputs: dict[str, dict[str, str]]) -> str:
    """A plain-text comparison of configurations on clips that may have no truth.

    `rows` are dicts with "clip", "audio_s", "truth" (or None) and "heard" (what Lucy
    recognised live, or None); `outputs` maps configuration -> clip -> text. Each clip
    is a header line, marked "DIFFERS" when the configurations' normalised outputs are
    not all equal, then one line per text. A summary line counts the differing clips."""
    width = max([len(n) for n in outputs] + [len("truth"), len("heard")])
    lines: list[str] = []
    differ = 0
    for r in rows:
        texts = {name: by_clip.get(r["clip"], "") for name, by_clip in outputs.items()}
        differs = len({normal_text(t) for t in texts.values()}) > 1
        differ += differs
        mark = "  DIFFERS" if differs else ""
        lines.append(f"{r['clip']}  ({r['audio_s']:.1f} s){mark}")
        if r.get("truth") is not None:
            lines.append(f"  {'truth':<{width}}  {r['truth']}")
        if r.get("heard") is not None:
            lines.append(f"  {'heard':<{width}}  {r['heard']}")
        lines += [f"  {name:<{width}}  {text}" for name, text in texts.items()]
        lines.append("")
    head = f"{len(rows)} clips; configurations differ (after normalisation) on {differ}\n"
    return head + "\n" + "\n".join(lines)
