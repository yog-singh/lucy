"""A cheap guard against text invented from silence and noise by indic-transcribe-flex.

Pure Python, no dependency; used by the `indic_flex` STT adapter (ADR 0023) and by
checks/stt_accuracy, which measured it. `junk_reason(text, audio_s, reply, level_dbfs)`
says why a transcript should be thrown away ("" to keep it). `reply` is the worker's
reply for the utterance: `tokens` generated, `max_new_tokens` (the cap used) and
`mean_logprob` of the chosen tokens. Checks, in order:

  quiet     the utterance's RMS level is under `Guard.min_dbfs`;
  cap       generation stopped at its token cap (a loop, not speech);
  rate      more than `token_slack + max_tokens_per_s x seconds` tokens;
  unsure    mean token log-probability under `Guard.min_mean_logprob`;
  redacted  the text is only the word "REDACTED" (the model says it on noise; a
            placeholder it seems to have learnt from redacted training transcripts).
A transcript that is kept goes through `collapse_repeats` (a run of 1 to 6 words said
`max_repeats` times or more in a row is kept once): real speech repeats a word three
times ("ji ji ji"), loops repeat it dozens of times.

The thresholds were set from the runs in checks/README.md (stt_accuracy, Latin-only
flex), on the same clips they are reported on, so they are not independently validated.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Guard:
    """The guard's thresholds (defaults as measured)."""

    min_dbfs: float = -55.0
    """Quietest real clip measured: -52.2 dBFS."""
    max_tokens_per_s: float = 25.0
    """Real clips at most 19.4 tokens per second; loops 60 to 250."""
    token_slack: float = 10.0
    """Tokens allowed on top of the rate, so one-word clips are not penalised."""
    min_mean_logprob: float = -1.0
    """Real speech down to -0.70 (romanized) and -1.27 (mixed) in the runs."""
    max_repeats: int = 4
    """A run of words repeated this many times or more in a row is kept once."""

    def __post_init__(self) -> None:
        if not -120.0 <= self.min_dbfs <= 0.0:
            raise ValueError("min_dbfs must be between -120 and 0")
        if self.max_tokens_per_s <= 0 or self.token_slack < 0:
            raise ValueError("max_tokens_per_s must be positive and token_slack not negative")
        if self.min_mean_logprob > 0:
            raise ValueError("min_mean_logprob must be 0 or negative (a log-probability)")
        if self.max_repeats < 2:
            raise ValueError("max_repeats must be at least 2")

    def token_cap(self, audio_s: float, ceiling: int = 512) -> int:
        """The most tokens worth generating for `audio_s` seconds (at least 1)."""
        return max(1, min(ceiling, int(self.token_slack + self.max_tokens_per_s * audio_s)))


DEFAULT_GUARD = Guard()

_REDACTED = re.compile(r"\W*redacted\W*", re.IGNORECASE)


def junk_reason(
    text: str,
    audio_s: float,
    reply: Mapping[str, Any],
    level_dbfs: float,
    guard: Guard = DEFAULT_GUARD,
) -> str:
    """Why `text` should be dropped ("quiet", "cap", "rate", "unsure", "redacted"), or ""."""
    if not text.strip():
        return ""
    if level_dbfs < guard.min_dbfs:
        return "quiet"
    tokens = reply.get("tokens", 0) or 0
    cap = reply.get("max_new_tokens")
    if cap and tokens >= cap:
        return "cap"
    if tokens > guard.token_slack + guard.max_tokens_per_s * audio_s:
        return "rate"
    if (reply.get("mean_logprob") or 0.0) < guard.min_mean_logprob:
        return "unsure"
    if _REDACTED.fullmatch(text.strip()):
        return "redacted"
    return ""


def collapse_repeats(text: str, times: int = 4, longest: int = 6) -> str:
    """`text` with any run of 1 to `longest` words repeated `times`+ times in a row kept once.

    A decoding loop ("install it install it install it ...") keeps its first copy."""
    words = text.split()
    key = [re.sub(r"\W+", "", w.lower()) for w in words]
    for n in range(1, longest + 1):
        i = 0
        while i + n * times <= len(words):
            gram = key[i : i + n]
            k = 1
            while key[i + k * n : i + (k + 1) * n] == gram:
                k += 1
            if k >= times and any(gram):
                del words[i + n : i + k * n]
                del key[i + n : i + k * n]
            i += 1
    return " ".join(words)


def dbfs(samples: Sequence[float]) -> float:
    """RMS level of float samples in dBFS (-inf for digital silence or no samples)."""
    n = len(samples)
    if n == 0:
        return -math.inf
    rms = math.sqrt(sum(float(x) * float(x) for x in samples) / n)
    return 20 * math.log10(rms) if rms > 0 else -math.inf
