"""Live evaluation of memory extraction + reconciliation against real local models.

usage (from the repo root): uv run python checks/extract_eval/extract_eval.py [--config FILE]
    [--label LABEL] [--out DIR] [--scenarios AB] [--turns N]
Feeds a scripted conversation (invented details) through the real pipeline
into a fresh database per scenario (OUT/LABEL-A/, OUT/LABEL-B/; default OUT is
data/checks/extract_eval/) and scores the result against hand-written expectations.
The extraction model and embedder are the config's `extraction` role and [embedder].
See checks/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from lucy.app import BUILTIN_ADAPTER_MODULES
from lucy.core import registry
from lucy.core.config import load_settings
from lucy.core.events import AssistantMessage, TurnEnded, UserMessage
from lucy.core.gateway import LLMGateway
from lucy.memory.extraction import Extractor
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.sqlite_store import SQLiteMemoryStore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DEFAULT_CONFIG, fresh_db, out_dir  # noqa: E402

TODAY = datetime(2026, 11, 12, 18, 0, tzinfo=UTC)

# (date, user text, assistant reply, expectations)
# expectation: dict(words=[...all must appear, case-insensitive...], kind=..., optional
#   valid_from="YYYY-MM"/"YYYY-MM-DD" prefix, valid_until=prefix, outcome=..., optional=True)
E = dict
TURNS = [
    (
        "2026-11-02T08:10",
        "Hi Lucy! How's your day going?",
        "Hello! All good here. How can I help?",
        [],
    ),
    (
        "2026-11-02T08:12",
        "My name's Rohan, by the way.",
        "Nice to meet you, Rohan!",
        [E(name="Rohan")],
    ),
    (
        "2026-11-02T08:15",
        "I work as a physiotherapist at Brookfield Clinic in Bristol.",
        "That sounds rewarding.",
        [E(words=["physio"], kind="fact"), E(words=["Brookfield"], kind="fact")],
    ),
    (
        "2026-11-02T08:20",
        "What's the weather like in Bristol tomorrow?",
        "I can't check the weather yet, sorry.",
        [],
    ),
    (
        "2026-11-03T19:00",
        "My sister Anika lives in Edinburgh, she's a vet.",
        "Edinburgh is lovely. Does she enjoy being a vet?",
        [
            E(words=["Anika", "sister"], kind="fact"),
            E(words=["Anika", "Edinburgh"], kind="fact"),
            E(words=["Anika", "vet"], kind="fact"),
        ],
    ),
    (
        "2026-11-03T19:05",
        "I drink a lot of dark roast coffee, usually three cups a day.",
        "That's quite a lot of coffee!",
        [
            E(words=["dark roast"], kind=("preference", "habit")),
            E(words=["three cups"], kind=("preference", "habit"), optional=True),
        ],
    ),
    (
        "2026-11-03T19:07",
        "Please don't read web links out loud, it's really annoying.",
        "Understood, I won't read links aloud.",
        [E(words=["link"], kind="instruction")],
    ),
    ("2026-11-03T19:08", "Haha, that's funny. Thanks!", "You're welcome!", []),
    (
        "2026-11-04T07:30",
        "I've been learning the cello since September.",
        "How is it going so far?",
        [E(words=["cello"], kind=("fact", "habit"), valid_from="2026-09")],
    ),
    (
        "2026-11-04T07:35",
        "If I had more time I'd probably take up rock climbing.",
        "Rock climbing is a great workout.",
        [],
    ),
    (
        "2026-11-05T12:00",
        "I want to run the Bath half marathon next April.",
        "Exciting! Have you started training?",
        [E(words=["Bath", "half marathon", "2027"], kind="goal")],
    ),
    (
        "2026-11-05T12:05",
        "Can you explain how compound interest works?",
        "Interest is added to the principal, so later interest is earned on interest too.",
        [],
    ),
    (
        "2026-11-06T21:00",
        "I was choosing between a Kindle and a Kobo for reading, and went with the Kobo"
        " because it lets me borrow library books.",
        "Good choice, library lending is handy.",
        [
            E(words=["Kindle"], kind="decision"),
            E(words=["Kobo"], kind="decision"),
            E(words=["Kobo", "chose"], kind="decision"),
            E(words=["library"], kind="decision"),
        ],
    ),
    (
        "2026-11-06T21:10",
        "My dog Pepper is a three-year-old beagle.",
        "Beagles are so friendly!",
        [E(words=["Pepper", "dog"], kind="fact"), E(words=["Pepper", "beagle"], kind="fact")],
    ),
    (
        "2026-11-07T09:00",
        "What time is it in Tokyo right now?",
        "I don't have a clock tool yet.",
        [],
    ),
    (
        "2026-11-07T09:05",
        "I prefer short answers in the morning, I'm usually in a rush.",
        "Got it, I'll keep mornings brief.",
        [
            E(words=["short", "morning"], kind="preference"),
            E(words=["Lucy", "short"], kind="instruction", optional=True),
        ],
    ),
    (
        "2026-11-07T09:10",
        "Anika is coming to visit next Friday.",
        "Lovely, enjoy the visit!",
        [
            E(
                words=["Anika", "visit"],
                kind=("fact", "event", "goal"),
                valid_from="2026-11-13",
                optional=True,
            )
        ],
    ),
    ("2026-11-08T22:00", "Ugh, I'm so tired today.", "Sorry to hear that. Get some rest!", []),
    (
        "2026-11-09T18:00",
        "I go to the gym every Monday and Thursday after work.",
        "Nice routine.",
        [E(words=["gym"], kind="habit")],
    ),
    (
        "2026-11-09T18:05",
        "Would a standing desk help with back pain, do you think?",
        "It can help some people, especially if you alternate sitting and standing.",
        [],
    ),
    (
        "2026-11-10T08:00",
        "Like I said, I'm a physio over at Brookfield.",
        "Right, at Brookfield Clinic.",
        [E(words=["physio"], kind="fact", outcome=("reinforce", "duplicate"))],
    ),
    (
        "2026-11-10T08:05",
        "I've switched to decaf, the dark roast was wrecking my sleep.",
        "Decaf it is. Hope you sleep better.",
        [E(words=["decaf"], kind=("preference", "habit"), outcome="supersede")],
    ),
    (
        "2026-11-10T20:00",
        "My partner Leo is a software engineer at Monzo.",
        "Sounds like a busy household!",
        [
            E(words=["Leo", "partner"], kind="fact"),
            E(words=["Leo", "engineer"], kind="fact"),
            E(words=["Leo", "Monzo"], kind="fact"),
        ],
    ),
    ("2026-11-10T20:02", "Lol okay. Anyway.", "Anything else?", []),
    (
        "2026-11-11T08:00",
        "Call me Ro from now on.",
        "Sure thing, Ro.",
        [E(words=["Ro"], kind="instruction")],
    ),
    ("2026-11-11T08:05", "Let me think about what to cook tonight.", "Take your time.", []),
    (
        "2026-11-12T10:00",
        "Anika actually moved to Glasgow last month.",
        "Oh, how is she finding Glasgow?",
        [E(words=["Anika", "Glasgow"], kind="fact", valid_from="2026-10", outcome="supersede")],
    ),
    (
        "2026-11-12T10:05",
        "Tell me a joke.",
        "Why did the cello go to therapy? Too many strings attached.",
        [],
    ),
]


def matches(exp: dict, item: dict) -> bool:
    text = item["text"].lower()
    if not all(w.lower() in text for w in exp.get("words", [])):
        return False
    return True


def check(exp: dict, item: dict) -> list[str]:
    problems = []
    kinds = exp.get("kind")
    if kinds:
        kinds = (kinds,) if isinstance(kinds, str) else kinds
        if item["kind"] not in kinds:
            problems.append(f"kind {item['kind']} not {kinds}")
    if item["origin"] != "stated":
        problems.append(f"origin {item['origin']}")
    if "valid_from" in exp and not (item["valid_from"] or "").startswith(exp["valid_from"]):
        problems.append(f"valid_from {item['valid_from']} not {exp['valid_from']}")
    if "valid_until_or_from" in exp:
        p = exp["valid_until_or_from"]
        if not (
            (item["valid_until"] or "").startswith(p) or (item["valid_from"] or "").startswith(p)
        ):
            problems.append(f"date {item['valid_from']}/{item['valid_until']} not {p}")
    if "outcome" in exp:
        outs = (exp["outcome"],) if isinstance(exp["outcome"], str) else exp["outcome"]
        if item["outcome"] not in outs:
            problems.append(f"outcome {item['outcome']} not {outs}")
    return problems


# --- Scenario B: the problems found by the reflection evaluation -------------
# The user's name is not known until the end (B-last).
TURNS_B = [
    # A.1 single choices and orders: nothing
    (
        "2026-11-01T09:00",
        "I'll take the window seat.",
        "Booked: window seat on the 10:05 to Leeds.",
        [],
    ),
    ("2026-11-01T12:30", "Pad thai from Thai Orchid again please.", "Ordered your pad thai.", []),
    ("2026-11-01T19:00", "Let's go with the vegetarian lasagne tonight.", "Lasagne it is.", []),
    (
        "2026-11-01T19:30",
        "Get me the large flat white, oat milk.",
        "Ordered a large oat flat white.",
        [],
    ),
    # the exact messages the reflection evaluation stored as preferences or inferences
    (
        "2026-11-01T20:00",
        "Ordering the usual pad thai from Thai Orchid for lunch. Roughly how many calories is that?",
        "About 600 to 800 calories.",
        [],
    ),
    (
        "2026-11-01T20:05",
        "Booking the train to Edinburgh for next month. I'll take a window seat.",
        "Window seat noted for the booking.",
        [],
    ),
    (
        "2026-11-01T20:10",
        "Flight to Lisbon for the conference: I'll take the window seat again.",
        "Window seat it is.",
        [],
    ),
    (
        "2026-11-01T20:15",
        "Lidl run done. Any ideas for a vegetarian dinner?",
        "How about a chickpea curry?",
        [],
    ),
    ("2026-11-01T20:20", "Shorter please, just the headlines.", "Headlines only.", []),
    ("2026-11-01T20:25", "Just got off the phone with Mum, she says hi.", "Say hi back!", []),
    # A.1 stated preferences: kept
    (
        "2026-11-02T09:00",
        "I always pick an aisle seat on planes, I hate being stuck by the window.",
        "Noted, aisle seats.",
        [E(words=["aisle"], kind="preference")],
    ),
    (
        "2026-11-02T09:05",
        "I can't stand coriander.",
        "No coriander, got it.",
        [E(words=["coriander"], kind="preference")],
    ),
    (
        "2026-11-02T09:10",
        "I prefer emails to phone calls for anything work-related.",
        "Understood.",
        [E(words=["email"], kind="preference")],
    ),
    # A.2 plans: nothing; goals: kept
    ("2026-11-02T18:00", "I'm going to cycle to work next week.", "Good luck with the ride!", []),
    (
        "2026-11-02T18:05",
        "I'm planning to repaint the kitchen this weekend.",
        "Sounds like a project.",
        [],
    ),
    (
        "2026-11-03T08:00",
        "I want to get my pilot's licence by 2028.",
        "That's a great goal.",
        [E(words=["pilot"], kind="goal")],
    ),
    (
        "2026-11-03T08:05",
        "I'm working towards running a 5k in under 20 minutes.",
        "You can do it.",
        [E(words=["5k"], kind="goal")],
    ),
    # A.3 new entities: what it is and how it relates to the user
    (
        "2026-11-03T20:00",
        "We have a tortoise called Biscuit, he's about forty years old.",
        "Forty! Tortoises live a long time.",
        [E(words=["Biscuit", "tortoise"], kind="fact"), E(words=["Biscuit", "forty"], kind="fact")],
    ),
    (
        "2026-11-03T20:05",
        "Our car is an old blue Volvo we call Bertha.",
        "Bertha sounds reliable.",
        [E(words=["Bertha", "Volvo"], kind="fact")],
    ),
    (
        "2026-11-03T20:10",
        "Mrs Okafor next door keeps bees, she gave us a jar of honey.",
        "How kind of her.",
        [E(words=["Okafor", "neighbo"], kind="fact"), E(words=["Okafor", "bees"], kind="fact")],
    ),
    (
        "2026-11-03T20:15",
        "Juniper, my goddaughter, just started school.",
        "What a milestone.",
        [
            E(words=["Juniper", "goddaughter"], kind="fact"),
            E(words=["Juniper", "school"], kind="fact", optional=True),
        ],
    ),
    # A.4 changes of state: with and without a date
    (
        "2026-11-04T10:00",
        "My best friend Tobias lives in Lisbon.",
        "Lisbon is beautiful.",
        [E(words=["Tobias", "friend"], kind="fact"), E(words=["Tobias", "Lisbon"], kind="fact")],
    ),
    (
        "2026-11-04T10:05",
        "I work at Hartley & Sons as an accountant.",
        "Nice.",
        [E(words=["Hartley"], kind="fact"), E(words=["accountant"], kind="fact")],
    ),
    (
        "2026-11-06T10:00",
        "Tobias moved from Lisbon to Porto last month.",
        "Does he like Porto?",
        [
            E(
                words=["Tobias", "lives", "Porto"],
                kind="fact",
                valid_from="2026-10",
                outcome="supersede",
            )
        ],
    ),
    (
        "2026-11-07T10:00",
        "I left Hartley & Sons, I'm at Nordwind now.",
        "Congratulations on the new job!",
        [E(words=["Nordwind"], kind="fact", outcome="supersede")],
    ),
    # B.5/B.6 a goal and a related habit, then giving up
    (
        "2026-11-07T19:00",
        "I want to be fluent in Italian by next summer.",
        "Bellissimo!",
        [E(words=["Italian"], kind="goal")],
    ),
    (
        "2026-11-07T19:05",
        "I practise Italian on Duolingo every evening.",
        "Great habit.",
        [E(words=["Italian"], kind="habit")],
    ),
    (
        "2026-11-09T21:00",
        "I've given up on Italian, I just don't have the time.",
        "That's fair.",
        [E(words=["Italian"], kind="fact", outcome="supersede")],
    ),
    # D.9 the name learned late
    ("2026-11-10T08:00", "By the way, I'm Mira.", "Nice to meet you, Mira!", [E(name="Mira")]),
    ("2026-11-10T08:05", "What's the capital of Peru?", "Lima.", []),
]


async def run_scenario(name, turns, extractor, graph, memory, sink):
    rows = []
    totals = dict(
        expected=0,
        matched=0,
        missed=0,
        spurious=0,
        problems=0,
        failures=0,
        optional_hit=0,
        nothing_turns=0,
        nothing_ok=0,
        dropped=0,
    )
    seconds = []
    for i, (when, user, reply, expected) in enumerate(turns, start=1):
        ts = datetime.fromisoformat(when).replace(tzinfo=UTC)
        turn = f"{name}{i}"
        await memory.append_episode(UserMessage(session_id=name, turn_id=turn, ts=ts, text=user))
        await memory.append_episode(
            AssistantMessage(session_id=name, turn_id=turn, ts=ts, text=reply)
        )
        await memory.append_episode(
            TurnEnded(session_id=name, turn_id=turn, ts=ts, reason="completed")
        )
        t0 = time.perf_counter()
        try:
            report = await asyncio.wait_for(extractor.run(1), 900)
        except Exception as exc:  # noqa: BLE001
            totals["failures"] += 1
            print(f"{name}{i} FAILED: {type(exc).__name__}: {exc}")
            [eps] = await extractor.ready_turns(1) or [[]]
            if eps:
                await extractor.skip_turn(eps, "eval failure")
            rows.append(dict(turn=i, user=user, error=str(exc)))
            continue
        dt = time.perf_counter() - t0
        seconds.append(dt)
        tr = report.turns[0]
        cands = tr.extraction.statements if tr.extraction else []
        cand_items = []
        order = getattr(tr, "reconciled", None) or [
            c for c in cands if not any(d.candidate is c for d in tr.dropped)
        ]
        for c, res in zip(order, tr.results, strict=False):
            cand_items.append(
                dict(
                    text=c.text,
                    kind=c.kind,
                    origin=c.origin,
                    valid_from=c.valid_from,
                    valid_until=c.valid_until,
                    outcome=res.outcome.value,
                    subject=f"{c.subject.name}/{c.subject.type}",
                    predicate=c.predicate,
                    obj=(
                        f"{c.object_entity.name}/{c.object_entity.type}"
                        if c.object_entity
                        else c.object_value
                    ),
                    reason=res.reason,
                    stored=(res.statement.text if res.statement else None),
                    target=res.target_id,
                )
            )
        totals["dropped"] += len(tr.dropped)
        used = set()
        lines = []
        if tr.renamed_user:
            lines.append(f"   renamed user -> {tr.renamed_user}")
        for exp in expected:
            if "name" in exp:
                totals["expected"] += 1
                if tr.renamed_user == exp["name"]:
                    totals["matched"] += 1
                else:
                    totals["missed"] += 1
                    lines.append(f"   MISSED name {exp['name']}")
                continue
            hit = next(
                (j for j, it in enumerate(cand_items) if j not in used and matches(exp, it)), None
            )
            if exp.get("optional"):
                if hit is not None:
                    used.add(hit)
                    totals["optional_hit"] += 1
                    probs = check(exp, cand_items[hit])
                    lines.append(f"   ok(optional) {cand_items[hit]['text']!r} {probs or ''}")
                continue
            totals["expected"] += 1
            if hit is None:
                totals["missed"] += 1
                lines.append(f"   MISSED {exp['words']} ({exp.get('kind')})")
            else:
                used.add(hit)
                totals["matched"] += 1
                probs = check(exp, cand_items[hit])
                if probs:
                    totals["problems"] += 1
                lines.append(
                    f"   ok {cand_items[hit]['text']!r} {('PROBLEMS ' + str(probs)) if probs else ''}"
                )
        for j, it in enumerate(cand_items):
            if j not in used:
                totals["spurious"] += 1
                lines.append(
                    f"   EXTRA {it['text']!r} [{it['kind']}/{it['origin']} {it['outcome']}]"
                )
        for d in tr.dropped:
            lines.append(f"   DROPPED {d.candidate.text!r}: {d.reason}")
        if not expected:
            totals["nothing_turns"] += 1
            if not cand_items:
                totals["nothing_ok"] += 1
        print(f"{name}{i} {dt:5.1f}s  {user}")
        for c in cand_items:
            print(
                f"     . [{c['kind']}/{c['origin']}] {c['text']!r} subj={c['subject']} pred={c['predicate']!r} obj={c['obj']!r} from={c['valid_from']} until={c['valid_until']} -> {c['outcome']} t={c['target']} ({c['reason']})"
            )
        for line in lines:
            print(line)
        rows.append(
            dict(
                turn=i,
                user=user,
                seconds=dt,
                candidates=cand_items,
                dropped=[(d.candidate.text, d.reason) for d in tr.dropped],
                renamed=tr.renamed_user,
            )
        )
    print(f"\n[{name}] ENTITIES")
    for e in await graph.list_entities():
        print(f"  {e.id:3d} {e.type:12s} {e.name!r} aliases={e.aliases}")
    print(f"\n[{name}] STATEMENTS (all)")
    for s in await graph.list_statements(current_only=False):
        flag = (
            "" if s.invalid_at is None else f" [ended {s.invalid_at.date()} -> {s.superseded_by}]"
        )
        vf = f" from {s.valid_from.date()}" if s.valid_from else ""
        print(
            f"  {s.id:3d} subj={s.subject_id} obj={s.object_id} {s.kind.value}/{s.origin.value} c={s.confidence:.2f} n={s.evidence_count} {s.text!r}{vf}{flag}"
        )
    print(f"\n[{name}] TOTALS", json.dumps(totals))
    print(f"[{name}] seconds/turn: mean={sum(seconds) / len(seconds):.2f} max={max(seconds):.2f}")
    sink[name] = dict(totals=totals, rows=rows, seconds=seconds)


async def checks_b(graph) -> None:
    """End-state checks for scenario B."""
    from lucy.memory.models import USER_ENTITY_ID

    allst = await graph.list_statements(current_only=False)
    cur = [s for s in allst if s.invalid_at is None]
    by_id = {s.id: s for s in allst}
    out = []

    def ok(label, cond, detail=""):
        out.append(f"  {'PASS' if cond else 'FAIL'} {label} {detail}")

    # A.1: nothing inferred by extraction, no window/pad thai/lasagne/flat white preference
    inferred = [s.text for s in allst if s.origin.value == "inferred"]
    ok("no inferred statements from extraction", not inferred, str(inferred))
    choice = [
        s.text
        for s in allst
        if any(
            w in s.text.lower()
            for w in (
                "window seat",
                "pad thai",
                "lasagne",
                "flat white",
                "vegetarian",
                "oat",
                "headline",
                "mother",
                "mum",
                "lidl",
            )
        )
    ]
    ok("no statement from single choices", not choice, str(choice))
    plans = [
        s.text for s in allst if any(w in s.text.lower() for w in ("cycle", "repaint", "kitchen"))
    ]
    ok("no statement from plans", not plans, str(plans))
    # A.3: each new non-user entity connected to the user
    ents = await graph.list_entities()
    for e in ents:
        if e.id == USER_ENTITY_ID or e.type in (
            "place",
            "organisation",
            "decision",
            "topic",
            "language",
        ):
            continue
        linked = [
            s
            for s in cur
            if (s.subject_id == USER_ENTITY_ID and s.object_id == e.id)
            or (s.subject_id == e.id and s.object_id == USER_ENTITY_ID)
            or (s.subject_id == e.id and "mira" in s.text.lower())
        ]
        ok(
            f"entity {e.name!r} ({e.type}) linked to the user",
            bool(linked),
            str([s.text for s in linked][:2]),
        )
    tort = [s.text for s in cur if "tortoise" in s.text.lower() and "biscuit" in s.text.lower()]
    ok("'what's my tortoise called' answerable", bool(tort), str(tort))
    # A.4 / C.8: Lisbon superseded by a 'lives in Porto' statement from 2026-10
    lis = [
        s
        for s in allst
        if "lisbon" in s.text.lower()
        and "lives" in s.text.lower()
        and "porto" not in s.text.lower()
    ]
    for s in lis:
        succ = by_id.get(s.superseded_by)
        ok(
            "lives in Lisbon superseded by lives in Porto",
            succ is not None
            and "porto" in succ.text.lower()
            and "live" in succ.text.lower()
            and "lisbon" not in succ.text.lower(),
            f"{s.text!r} ended {s.invalid_at} -> {succ.text if succ else None!r}",
        )
        ok(
            "Lisbon ended in 2026-10",
            s.invalid_at is not None and s.invalid_at.isoformat().startswith("2026-10"),
            str(s.invalid_at),
        )
    porto = [s.text for s in cur if "porto" in s.text.lower() and s.subject_id != USER_ENTITY_ID]
    ok("Tobias lives in Porto current", any("live" in t.lower() for t in porto), str(porto))
    # job change, no date: ended at the message date 2026-11-07
    hart = [s for s in allst if "hartley" in s.text.lower() and "work" in s.text.lower()]
    for s in hart:
        ok(
            "Hartley ended at message date 2026-11-07",
            s.invalid_at is not None and s.invalid_at.date().isoformat() == "2026-11-07",
            f"{s.text!r} {s.invalid_at} -> {s.superseded_by}",
        )
    # B.5/6 Italian goal closed at 2026-11-09 by a fact; habit ended too
    goal = [s for s in allst if s.kind.value == "goal" and "italian" in s.text.lower()]
    for g in goal:
        succ = by_id.get(g.superseded_by)
        ok(
            "Italian goal ended 2026-11-09",
            g.invalid_at is not None and g.invalid_at.date().isoformat() == "2026-11-09",
            str(g.invalid_at),
        )
        ok(
            "Italian goal closed by a fact",
            succ is not None and succ.kind.value == "fact",
            f"-> {succ.kind.value + ' ' + succ.text if succ else None}",
        )
    habit = [s for s in allst if s.kind.value == "habit" and "italian" in s.text.lower()]
    for h in habit:
        ok(
            "Italian habit ended",
            h.invalid_at is not None,
            f"{h.text!r} {h.invalid_at} -> {h.superseded_by}",
        )
    # D.9 no 'the user' left
    left = [s.text for s in allst if "the user" in s.text.lower()]
    ok("no 'the user' texts after naming", not left, str(left))
    print("\n[B] CHECKS")
    print("\n".join(out))


async def main(
    config: str, label: str, out_base: Path, which: str = "AB", limit: int | None = None
) -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings(config)
    registry.load_adapters(BUILTIN_ADAPTER_MODULES)
    gateway = LLMGateway.from_settings(settings.llm)
    model = settings.llm.roles["extraction"].options.get("model")
    if settings.embedder is None:
        raise SystemExit(f"{config} has no [embedder]")
    embedder = registry.create("embedder", settings.embedder.adapter, **settings.embedder.options)
    await llm_warm(gateway)
    sink = {}
    for name, turns in (("A", TURNS), ("B", TURNS_B)):
        if name not in which:
            continue
        turns = turns[:limit] if limit else turns
        db = fresh_db(out_base / f"{label}-{name}")
        memory = SQLiteMemoryStore(db)
        graph = SQLiteGraphStore(db, embedder, clock=lambda: TODAY)
        extractor = Extractor(graph, gateway, clock=lambda: TODAY)
        await run_scenario(name, turns, extractor, graph, memory, sink)
        if name == "B":
            await checks_b(graph)
        await graph.close()
        await memory.close()
    out = out_base / f"{label}.json"
    out.write_text(json.dumps(dict(model=model, **sink), indent=1, default=str))
    print(f"\nresults written to {out}")
    for llm_ in {id(gateway.for_role(r)): gateway.for_role(r) for r in gateway.roles()}.values():
        await llm_.aclose()
    await embedder.aclose()


async def llm_warm(gateway: LLMGateway) -> None:
    from pydantic import BaseModel

    from lucy.core.types import Message, Role

    class Ok(BaseModel):
        ok: bool

    await gateway.generate("extraction", [Message(role=Role.USER, content='Say {"ok": true}')], Ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", help="output directory (default: data/checks/extract_eval/)")
    ap.add_argument("--scenarios", default="AB", help="which scenarios to run: A, B or AB")
    ap.add_argument(
        "--turns", type=int, help="run only the first N turns of each scenario (smoke run)"
    )
    args = ap.parse_args()
    asyncio.run(
        main(args.config, args.label, out_dir("extract_eval", args.out), args.scenarios, args.turns)
    )
