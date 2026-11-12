"""Word error rate with light normalisation (shared by the e2e scripts)."""

import re

NUM = {"twenty five": "25", "six thirty": "630", "three forty five": "345", "fourteenth": "14th"}


def norm(text: str) -> list[str]:
    t = text.lower().replace("-", " ")
    t = re.sub(r"(\d+):(\d+)", r"\1\2", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    t = " ".join(t.split())
    for k, v in NUM.items():
        t = t.replace(k, v)
    return t.split()


def distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r = norm(ref)
    return distance(r, norm(hyp)) / max(1, len(r))


def corpus_wer(pairs) -> float:
    errors = sum(distance(norm(r), norm(h)) for r, h in pairs)
    return errors / max(1, sum(len(norm(r)) for r, _ in pairs))
