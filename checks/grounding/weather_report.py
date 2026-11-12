"""Scores the w_ cases (weather: a lookup, no invented numbers, asks when no place is known)
from a rows file. usage: uv run python checks/grounding/weather_report.py data/checks/grounding/rows_LABEL.json
"""

import json
import re
import sys

from new_cases import NEW_CASES

rows = json.load(open(sys.argv[1]))
for r in rows:
    case = NEW_CASES[r["case"]]
    wx = [c[1] for c in r["calls"] if c[0] == "weather"]
    others = [c[0] for c in r["calls"] if c[0] != "weather"]
    text = r["text"]
    outs = (
        " ".join(w.get("_out", "") for w in wx)
        + " "
        + " ".join(res for (_, calls, _) in case["prior"] for (_, _, res) in calls)
    )
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    invented = [n for n in nums if n not in outs and n.lstrip("-") not in outs]
    want = case["want"]
    if want == "weather":
        ok = bool(wx) and not invented
    elif want == "ask":
        ok = not wx and not others and "?" in text
    elif want == "no_weather":
        ok = not wx
    else:  # followup: answer about Saturday from earlier result or a new call
        ok = ("saturday" in text.lower() or "14" in text) and not invented
    print(
        f"{'OK ' if ok else 'BAD'} {r['case']:15s} r{r['rep']} {r['total']:5.1f}s wx={[{k: v for k, v in w.items() if not k.startswith('_')} for w in wx]} "
        f"wx_s={[w.get('_secs') for w in wx]} other={others} invented={invented} | {text[:230]!r}"
    )
