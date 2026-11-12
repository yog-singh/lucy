"""Scores the rq_ cases (news: is `recent` set on the first web_search?) from a rows file.

usage: uv run python checks/grounding/recent_report.py data/checks/grounding/rows_LABEL.json
"""

import json
import sys

from new_cases import NEW_CASES

rows = json.load(open(sys.argv[1]))
ok = 0
for r in rows:
    ws = [c[1] for c in r["calls"] if c[0] == "web_search"]
    want = NEW_CASES[r["case"]]["want_recent"]
    first = (ws[0].get("recent") is True) if ws else None
    ok += first == want
    print(
        f"{r['case']:18s} want={want!s:5s} first={first!s:5s} all={[w.get('recent') for w in ws]} other={[c[0] for c in r['calls'] if c[0] != 'web_search']}"
    )
print(f"first search set as wanted: {ok}/{len(rows)}")
