"""Does Lucy reply in the language of the user's latest message, Hindi in Latin letters?

usage (from the repo root): uv run python checks/reply_language/reply_language.py
    [--config FILE] [--label LABEL] [--out DIR] [--reps N] [--cases a,b]
    [--silent-after SECONDS]
       uv run python checks/reply_language/reply_language.py --summarise ROWS.json ...
Short scripted conversations (English, romanised Hindi, mixed, switches each way, very
short messages, tool requests, a confirmation) through the real app, prompt assembly,
gateway and chat model, each in a fresh session, on a fresh database in OUT/LABEL/
(default data/checks/reply_language/run/). Tools are the ones enabled in the config,
with the network ones (wikipedia, web_search, fetch_page, weather) replaced by fixed
results and the calendar tools on the FAKE in-memory calendar from the tests; time and
the reminder tools are the real ones (reminders go to the check's own database). Pass a
config with MCP servers, speaker recognition and learning off. Every turn is scored
(`scoring.py`): Devanagari characters, language against the expected one, the expected
tool called (or none), empty reply, fallback retries, and how the reply ends
(checks/endings). See checks/README.md.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_CONFIG, fresh_db, isolated_settings, out_dir  # noqa: E402
from scoring import (  # noqa: E402
    devanagari,
    hindi_share,
    language_ok,
    latin_words,
    script_ok,
    verdict,
    words,
)

ANY = "?"  # tool expectation: any or none
# Each case is a list of turns (user text, expected reply language, expected tool):
# a tool name means that tool must be called, None means no tool, ANY means either.
CASES: dict[str, list[tuple[str, str, Any]]] = {
    "en_fact": [("What is the capital of Japan?", "en", None)],
    "en_smalltalk": [("How are you doing today?", "en", None)],
    "en_time": [("What time is it in Tokyo right now?", "en", "time")],
    "en_weather": [("What's the weather like in Delhi today?", "en", "weather")],
    "en_calendar": [("What's on my calendar tomorrow?", "en", "calendar")],
    "en_numbers": [("Remind me tomorrow at 7:30 am to take 2 tablets.", "en", "set_reminder")],
    "hi_smalltalk": [("Lucy, aap kaisi ho?", "hi", None)],
    "hi_tired": [("Aaj ka din bahut thaka dene wala tha.", "hi", None)],
    "hi_fact": [("Japan ki capital kya hai?", "hi", None)],
    "hi_explain": [("Mujhe batao ki baarish kyun hoti hai.", "hi", None)],
    "hi_time": [("Tokyo mein abhi kitne baje hain?", "hi", "time")],
    "hi_weather": [("Aaj Delhi ka mausam kaisa hai?", "hi", "weather")],
    "hi_calendar": [("Kal mera calendar dikhao.", "hi", "calendar")],
    "mixed_meeting": [("Kal ki meeting ka time kya hai?", "mixed", "calendar")],
    "hi_numbers": [
        ("Kal subah 7:30 baje mujhe yaad dila dena ki 2 goliyan leni hain.", "hi", "set_reminder")
    ],
    "mixed_add": [("Friday ko 4 baje dentist appointment add kar do.", "mixed", ANY)],
    "hi_confirm": [
        ("Kal mere calendar mein kya hai?", "hi", "calendar"),
        ("Workout wala event delete kar do.", "hi", "delete_calendar_event"),
    ],
    "switch_en_hi": [
        ("What's the capital of France?", "en", None),
        ("Tell me one interesting thing about it.", "en", ANY),
        ("Accha, wahan ka khana kaisa hota hai?", "hi", ANY),
    ],
    "switch_hi_en": [
        ("Aap kaisi ho?", "hi", None),
        ("Mujhe ek chhota sa joke sunao.", "hi", None),
        ("What's a good book to read this weekend?", "en", ANY),
    ],
    "short_haan_hi": [("Mujhe neend nahi aa rahi, kya karun?", "hi", None), ("haan", "hi", None)],
    "short_theek_hi": [
        ("Chai peena sehat ke liye accha hai kya?", "hi", None),
        ("theek hai", "hi", None),
    ],
    "short_thanks_hi": [
        ("Kal Delhi mein baarish hogi kya?", "hi", "weather"),
        ("thanks", "hi", None),
    ],
    "short_ok_en": [("Any tips for sleeping better?", "en", None), ("ok", "en", None)],
    "short_haan_en": [("Is green tea healthy?", "en", None), ("haan", "en", None)],
    "short_thanks_en": [("How many legs does a spider have?", "en", None), ("thanks", "en", None)],
}
WEATHER = (
    "Weather from Open-Meteo for Delhi, India, observed at 11:00 local time today.\n"
    "Now: 24°C, feels like 25°C, haze, wind 8 km/h from the north-west, humidity 52%.\n"
    "Today: high 28°C, low 14°C, mainly clear, chance of rain 0%.\n"
    "Tomorrow: high 27°C, low 13°C, partly cloudy, chance of rain 10%."
)
CANNED = {
    "weather": WEATHER,
    "wikipedia": "No Wikipedia article matched that query.",
    "web_search": "No results found for that query.",
    "fetch_page": "The page could not be fetched.",
}


class Canned:
    """A network tool replaced by a fixed result (real name, description, parameters)."""

    def __init__(self, real: Any) -> None:
        self.name, self.description, self.parameters = real.name, real.description, real.parameters
        self.risk = real.risk
        self.sends_out = getattr(real, "sends_out", False)

    async def run(self, args: dict[str, Any]):
        from lucy.core.ports import ToolOutput

        return ToolOutput(content=CANNED[self.name])


class RetryCounter(logging.Handler):
    """Counts the gateway's silent-stream retries and fallbacks (its warnings)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "produced no output" in record.getMessage():
            self.count += 1


async def turn(app, sid, text, limit):
    """(model text before any confirmation, code's confirmation text, tools, error)."""
    from lucy.core.events import (
        ConfirmationRequested,
        TextDelta,
        ToolFinished,
        ToolStarted,
        TurnEnded,
        UserMessage,
    )

    state: dict[str, Any] = {"before": "", "confirm": "", "tools": [], "calls": []}

    async def run():
        async with app.bus.subscribe(
            TextDelta, ToolStarted, ToolFinished, ConfirmationRequested, TurnEnded
        ) as evs:
            await app.bus.publish(UserMessage(session_id=sid, text=text, source="text"))
            async for e in evs:
                if e.session_id != sid:
                    continue
                if isinstance(e, TextDelta):
                    if not state["confirm"]:
                        state["before"] += e.text
                elif isinstance(e, ToolStarted):
                    state["tools"].append(e.call.name)
                    state["calls"].append(
                        {"id": e.call.id, "name": e.call.name, "args": e.call.arguments}
                    )
                elif isinstance(e, ToolFinished):
                    for call in state["calls"]:
                        if call.get("id") == e.result.call_id:
                            call["result"] = e.result.content[:200]
                elif isinstance(e, ConfirmationRequested):
                    state["confirm"] = e.question
                    state["tools"].append(e.call.name)  # asked before it starts
                    state["calls"].append(
                        {"id": e.call.id, "name": e.call.name, "args": e.call.arguments}
                    )
                elif isinstance(e, TurnEnded):
                    return e.error if e.reason == "error" else None

    try:
        error = await asyncio.wait_for(run(), limit)
    except TimeoutError:
        error = f"no end within {limit:.0f} s"
    return state["before"], state["confirm"], state["tools"], state["calls"], error


def score_turn(
    reply: str, confirm: str, lang: str, want_tool: Any, tools: list[str]
) -> dict[str, Any]:
    """Scores of one turn. A turn where only the code spoke (its English confirmation
    question, no model text) has no language score (`lang_ok` None) and is not empty."""
    from endings.endings import classify

    if want_tool == ANY:
        tool_ok = True
    elif want_tool is None:
        tool_ok = not tools
    else:
        tool_ok = want_tool in tools
    return {
        "devanagari": devanagari(reply),
        "hindi_share": round(hindi_share(reply), 3),
        "verdict": verdict(reply),
        "lang_ok": None if confirm and not reply.strip() else language_ok(lang, reply),
        "script_ok": None if not reply.strip() else script_ok(lang, reply),
        "latin_in_deva": latin_words(reply) if devanagari(reply) else 0,
        "tool_ok": tool_ok,
        "empty": not reply.strip() and not confirm,
        "ending": classify(reply),
        "words": len(words(reply)),
    }


def summarise(label: str, rows: list[dict[str, Any]]) -> None:
    """Print the totals of `rows`, re-scoring each reply with the current scorer."""
    for r in rows:
        r.update(score_turn(r["reply"], r["confirm"], r["lang"], r["want_tool"], r["tools"]))
    n = len(rows)
    by_lang = {}
    scored = [r for r in rows if r["lang_ok"] is not None]
    for lang in ("en", "hi", "mixed"):
        sub = [r for r in scored if r["lang"] == lang]
        by_lang[lang] = f"{sum(r['lang_ok'] for r in sub)}/{len(sub)}"
    tooled = [r for r in rows if r["want_tool"] != ANY]
    scripted = [r for r in rows if r["script_ok"] is not None]
    hindi_replies = [r for r in rows if r["devanagari"] and r["lang"] != "en"]
    en = [r for r in rows if r["lang"] == "en" and not r["empty"]]
    print(
        f"SUMMARY {label}: {n} turns; language ok {sum(r['lang_ok'] for r in scored)}/{len(scored)} "
        f"(en {by_lang['en']}, hi {by_lang['hi']}, mixed {by_lang['mixed']}); "
        f"script ok {sum(r['script_ok'] for r in scripted)}/{len(scripted)} "
        f"(en {sum(r['script_ok'] for r in scripted if r['lang'] == 'en')}/"
        f"{sum(r['lang'] == 'en' for r in scripted)}, hi+mixed "
        f"{sum(r['script_ok'] for r in scripted if r['lang'] != 'en')}/"
        f"{sum(r['lang'] != 'en' for r in scripted)}); Devanagari in "
        f"{sum(r['devanagari'] > 0 for r in rows)} turns; Latin words per Devanagari "
        f"reply {mean([r['latin_in_deva'] for r in hindi_replies] or [0]):.2f} "
        f"({sum(r['latin_in_deva'] > 0 for r in hindi_replies)}/{len(hindi_replies)} "
        f"with any); "
        f"tool ok {sum(r['tool_ok'] for r in tooled)}/{len(tooled)}; "
        f"empty {sum(r['empty'] for r in rows)}; retries {sum(r['retries'] for r in rows)}; "
        f"errors {sum(r['error'] is not None for r in rows)}; "
        f"offer+menu endings {sum(r['ending'] in ('offer', 'menu') for r in rows)}; "
        f"mean words English {mean(r['words'] for r in en) if en else 0:.1f}, "
        f"Hindi/mixed {mean([r['words'] for r in rows if r['lang'] != 'en'] or [0]):.1f}"
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", help="output directory (default: data/checks/reply_language/)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cases", help="comma-separated case names (default: all)")
    ap.add_argument("--silent-after", type=float, default=15.0)
    ap.add_argument("--summarise", nargs="+", metavar="ROWS", help="summarise rows files only")
    args = ap.parse_args()
    if args.summarise:
        for path in args.summarise:
            summarise(path, json.loads(Path(path).read_text()))
        return

    from confirm_once.confirm_once import Holder, events, quick_silence

    from lucy.app import LucyApp
    from lucy.tools import build_tools
    from lucy.tools.calendar_changes import DeleteCalendarEventTool, EditCalendarEventTool
    from lucy.tools.calendar_dates import system_zone
    from lucy.tools.calendar_events import AddCalendarEventTool, CalendarTool
    from tests.test_tools_calendar import FakeStore

    tag = args.label
    names = args.cases.split(",") if args.cases else list(CASES)
    db = fresh_db(out_dir("reply_language", args.out) / tag)
    holder = Holder()
    zone = system_zone()

    def build(settings):
        s = settings.tools.calendar
        kw = {"store": holder, "local_zone": zone}
        changes = {
            "changes_enabled": bool(
                {"edit_calendar_event", "delete_calendar_event"} & set(settings.tools.enabled)
            )
        }
        fake = {
            "calendar": lambda: CalendarTool(s, **kw, **changes),
            "add_calendar_event": lambda: AddCalendarEventTool(s, **kw, **changes),
            "edit_calendar_event": lambda: EditCalendarEventTool(s, **kw),
            "delete_calendar_event": lambda: DeleteCalendarEventTool(s, **kw),
        }
        out = []
        for real in build_tools(settings):
            if real.name in CANNED:
                out.append(Canned(real))
            elif real.name in fake:
                out.append(fake[real.name]())
            else:
                out.append(real)
        return out

    counter = RetryCounter()
    logging.getLogger("lucy.core.gateway").addHandler(counter)
    limit = 4 * args.silent_after + 30
    rows = []
    async with LucyApp(
        quick_silence(isolated_settings(args.config, db), args.silent_after),
        background=False,
        tool_builder=build,
    ) as app:
        for rep in range(args.reps):
            for name in names:
                holder.store = FakeStore(events())
                sid = f"{tag}-{name}-{rep}"
                for i, (text, lang, want_tool) in enumerate(CASES[name]):
                    before = counter.count
                    t0 = time.monotonic()
                    reply, confirm, tools, calls, err = await turn(app, sid, text, limit)
                    row = {
                        "case": name,
                        "rep": rep,
                        "turn": i,
                        "q": text,
                        "lang": lang,
                        "want_tool": want_tool,
                        "reply": reply,
                        "confirm": confirm,
                        "tools": tools,
                        "calls": calls,
                        "error": err,
                        "retries": counter.count - before,
                        "secs": round(time.monotonic() - t0, 1),
                        **score_turn(reply, confirm, lang, want_tool, tools),
                    }
                    rows.append(row)
                    flags = (
                        ("L" if row["lang_ok"] is False else ".")
                        + ("." if row["tool_ok"] else "T")
                        + ("S" if row["script_ok"] is False else ".")
                        + ("E" if row["empty"] else ".")
                        + ("R" if row["retries"] else ".")
                    )
                    print(
                        f"[{flags}] {name}#{i} {lang}->{row['verdict'] or '-'} "
                        f"({row['hindi_share']:.2f}, Latin {row['latin_in_deva']}) tools={tools} "
                        f"q={text!r}\n"
                        f"      a={reply.strip()!r}"
                        + "".join(
                            f"\n      call {c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                            f" -> {c.get('result', '')[:100]!r}"
                            for c in calls
                        )
                        + (f"\n      confirm={confirm!r}" if confirm else "")
                        + (f"\n      error={err!r}" if err else ""),
                        flush=True,
                    )
    path = db.parent / "rows.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    summarise(tag, rows)
    print(f"rows: {path}")


if __name__ == "__main__":
    asyncio.run(main())
