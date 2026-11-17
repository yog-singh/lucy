"""Pure helpers for the tts_listen check: timing summary, machine-check flags, the page.

No model code here: run.py feeds it the rows it measured and writes what comes back.
"""

from __future__ import annotations

import html
import statistics
from collections.abc import Sequence
from typing import Any


def rtf(total_s: float, duration_s: float) -> float | None:
    """Real-time factor: synthesis time over audio length (below 1 is faster than real time)."""
    return total_s / duration_s if duration_s > 0 else None


def summarise(rows: Sequence[dict[str, Any]], phase: str = "idle") -> dict[str, dict[str, Any]]:
    """Per column (model/voice): medians of first audio, total time and RTF for `phase`."""
    out: dict[str, dict[str, Any]] = {}
    columns = dict.fromkeys(r["column"] for r in rows if r.get("phase") == phase)
    for column in columns:
        mine = [r for r in rows if r["column"] == column and r.get("phase") == phase]
        ok = [r for r in mine if "error" not in r]
        rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
        out[column] = {
            "n": len(ok),
            "errors": len(mine) - len(ok),
            "first_median_s": statistics.median(r["first_s"] for r in ok) if ok else None,
            "first_max_s": max(r["first_s"] for r in ok) if ok else None,
            "total_median_s": statistics.median(r["total_s"] for r in ok) if ok else None,
            "rtf_median": statistics.median(rtfs) if rtfs else None,
        }
    return out


def compare_loaded(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per column: the median first audio and total time of the sentences timed both idle
    and with the chat model loaded, over the same sentences, and the ratio loaded/idle."""
    out: dict[str, dict[str, Any]] = {}
    loaded = [r for r in rows if r.get("phase") == "loaded" and "error" not in r]
    for column in dict.fromkeys(r["column"] for r in loaded):
        pairs = []
        for r in (x for x in loaded if x["column"] == column):
            idle = next(
                (
                    x
                    for x in rows
                    if x["column"] == column
                    and x["id"] == r["id"]
                    and x.get("phase") == "idle"
                    and "error" not in x
                ),
                None,
            )
            if idle:
                pairs.append((idle, r))
        if not pairs:
            continue
        med = {
            f"{k}_{p}": statistics.median(pair[i][k] for pair in pairs)
            for k in ("first_s", "total_s")
            for i, p in ((0, "idle"), (1, "loaded"))
        }
        out[column] = {
            "n": len(pairs),
            **med,
            "total_ratio": med["total_s_loaded"] / med["total_s_idle"]
            if med["total_s_idle"]
            else None,
        }
    return out


def simulate_playback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sentences played back to back in real time, each starting when its audio is
    ready (`ready_s`: wall clock from the first request) and the one before it has
    finished: count and total of the waits after the first sentence started."""
    t_play, waits = None, []
    for r in rows:
        if "error" in r:
            continue
        if t_play is None:
            t_play = r["ready_s"]
        elif r["ready_s"] > t_play:
            waits.append(r["ready_s"] - t_play)
            t_play = r["ready_s"]
        t_play += r["duration_s"]
    return {"underruns": len(waits), "underrun_s": sum(waits)}


def repeated_ngram(words: Sequence[str], n: int = 1) -> str | None:
    """A run of `n` to 4 words said twice in a row (a "stutter"), or None."""
    for size in range(n, 5):
        for i in range(len(words) - 2 * size + 1):
            if words[i : i + size] == words[i + size : i + 2 * size]:
                return " ".join(words[i : i + size])
    return None


def flags(
    ref_words: Sequence[str], hyp_words: Sequence[str], word_err: float, char_err: float | None
) -> list[str]:
    """Gross-failure flags for one transcript, not a quality score.

    dropped: under 60% of the reference's words came back; repeated: a stutter the
    reference does not have, or over 150% of its words; garbled: half or more of the
    words wrong (and, where a character error is known, 35% or more of the characters)."""
    out = []
    n_ref, n_hyp = max(1, len(ref_words)), len(hyp_words)
    if n_hyp < 0.6 * n_ref:
        out.append("dropped")
    stutter = repeated_ngram(list(hyp_words))
    if (stutter and f"{stutter} {stutter}" not in " ".join(ref_words)) or n_hyp > 1.5 * n_ref + 1:
        out.append("repeated")
    if word_err >= 0.5 and (char_err is None or char_err >= 0.35):
        out.append("garbled")
    return out


def _fmt(value: float | None, digits: int = 2, unit: str = "") -> str:
    return "–" if value is None else f"{value:.{digits}f}{unit}"


_STYLE = """
body{font:15px/1.4 -apple-system,system-ui,sans-serif;margin:16px;color:#111;background:#fff}
h1{font-size:20px}h2{font-size:16px;margin-top:28px}
table{border-collapse:collapse;margin:8px 0}
th,td{border:1px solid #ccc;padding:6px;vertical-align:top;text-align:left}
th{background:#f3f3f3}
td.text{max-width:300px}
audio{width:210px;display:block}
small{color:#555;display:block;font-size:11px;margin-top:2px}
.flag{color:#b00;font-weight:600}
.wrap{overflow-x:auto}
@media (prefers-color-scheme:dark){body{background:#111;color:#eee}th{background:#222}
th,td{border-color:#444}small{color:#aaa}.flag{color:#f77}}
"""


def render_page(
    title: str,
    sentences: Sequence[dict[str, Any]],
    columns: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    notes: Sequence[str] = (),
) -> str:
    """One self-contained HTML page: per-model facts on top, then sentences x models,
    each cell an <audio> player on the relative WAV path with its timings in small print.

    `columns`: {"column", "title", "facts": {label: text}}; `rows`: the measured rows
    (phase "idle" ones are shown) with "wav" relative to the page and optional "asr",
    "flags"."""
    e = html.escape
    summary = summarise(rows)
    labels = list(dict.fromkeys(k for c in columns for k in c.get("facts", {})))
    head = ["<tr><th>Model</th>"] + [f"<th>{e(c['title'])}</th>" for c in columns] + ["</tr>"]
    facts = []
    for label in labels:
        cells = "".join(f"<td>{e(str(c.get('facts', {}).get(label, '–')))}</td>" for c in columns)
        facts.append(f"<tr><th>{e(label)}</th>{cells}</tr>")
    med = "".join(
        f"<td>{_fmt(summary.get(c['column'], {}).get('first_median_s'), 2, ' s')}</td>"
        for c in columns
    )
    facts.append(f"<tr><th>Median time to first sound</th>{med}</tr>")
    rtfs = "".join(
        f"<td>{_fmt(summary.get(c['column'], {}).get('rtf_median'))}</td>" for c in columns
    )
    facts.append(f"<tr><th>Median real-time factor</th>{rtfs}</tr>")

    by_key = {(r["column"], r["id"]): r for r in rows if r.get("phase") == "idle"}
    body = []
    for s in sentences:
        cells = [
            f'<td class="text"><b>{e(s["id"])}</b> <small>{e(s["kind"])}</small>'
            f'<span lang="{e(s["lang"])}">{e(s["text"])}</span></td>'
        ]
        for c in columns:
            r = by_key.get((c["column"], s["id"]))
            if r is None:
                cells.append("<td>–</td>")
                continue
            if "error" in r:
                cells.append(
                    f'<td><span class="flag">error</span><small>{e(r["error"])}</small></td>'
                )
                continue
            info = (
                f"first {r['first_s']:.2f} s · total {r['total_s']:.2f} s · "
                f"audio {r['duration_s']:.2f} s · RTF {_fmt(r.get('rtf'))}"
            )
            extra = ""
            if r.get("asr") is not None:
                extra += f"<small>heard: {e(r['asr'])}</small>"
            if r.get("flags"):
                extra += f'<small class="flag">{e(", ".join(r["flags"]))}</small>'
            cells.append(
                f'<td><audio controls preload="none" src="{e(r["wav"])}"></audio>'
                f"<small>{info}</small>{extra}</td>"
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    sent_head = "<tr><th>Sentence</th>" + "".join(f"<th>{e(c['title'])}</th>" for c in columns)
    note_html = "".join(f"<li>{e(n)}</li>" for n in notes)
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{e(title)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{e(title)}</h1>"
        + (f"<ul>{note_html}</ul>" if note_html else "")
        + '<h2>Models</h2><div class="wrap"><table>'
        + "".join(head)
        + "".join(facts)
        + '</table></div><h2>Sentences</h2><div class="wrap"><table>'
        + sent_head
        + "</tr>"
        + "".join(body)
        + "</table></div></body></html>\n"
    )
