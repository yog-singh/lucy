"""The junk guard for indic-transcribe-flex, as the check uses it.

The guard itself (`junk_reason`, `collapse_repeats`, the thresholds) lives in
`lucy.adapters.stt_indic_flex_guard`, the one copy that Lucy's `indic_flex` speech
adapter and this check share; see its docstring for the rules. This module adds what
only the check needs: `GUARD` (the default thresholds as a dict, for the report),
`repeated` and `clip_dbfs` (the level of a clip file). Since the guard moved into
Lucy it also drops a transcript that is only "REDACTED".
"""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

from lucy.adapters.stt_indic_flex_guard import DEFAULT_GUARD, collapse_repeats, dbfs, junk_reason

__all__ = ["GUARD", "clip_dbfs", "collapse_repeats", "dbfs", "junk_reason", "repeated"]

GUARD: dict[str, float] = dataclasses.asdict(DEFAULT_GUARD)


def repeated(text: str, times: int) -> bool:
    """True if some run of 1 to 4 words occurs `times` times in a row."""
    words = re.findall(r"\w+", text.lower())
    for n in range(1, 5):
        for i in range(len(words) - n * times + 1):
            gram = words[i : i + n]
            if all(words[i + k * n : i + (k + 1) * n] == gram for k in range(1, times)):
                return True
    return False


def clip_dbfs(path: Path) -> float:
    import numpy as np

    from lucy.voice.audio import to_input_format

    s = np.asarray(to_input_format(path), dtype=np.float64)
    if not len(s):
        return -math.inf
    rms = float(np.sqrt(np.mean(s * s)))
    return 20 * math.log10(rms) if rms > 0 else -math.inf
