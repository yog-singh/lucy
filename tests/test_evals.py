"""The memory eval (`lucy.evals`): matching, scenario files, the runner and reports.

The runner runs against fake models here (no network); the bundled scenarios
only ever run against real ones (`lucy eval`).
"""

from __future__ import annotations

import io
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from lucy.clients import cli
from lucy.core.config import Settings
from lucy.core.events import UserMessage
from lucy.evals import command, report
from lucy.evals.matching import Match, has_keyword
from lucy.evals.runner import (
    CheckResult,
    ScenarioResult,
    ScenarioRunner,
    find_everywhere,
    settings_for,
)
from lucy.evals.scenario import (
    CATEGORIES,
    START,
    Say,
    Scenario,
    ScenarioError,
    StatementCheck,
    load_scenario,
    load_scenarios,
    parse_step,
)
from tests.learning_support import Models, learning_settings, reply
from tests.memory_extraction_support import cand, extraction

# --- Matching -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "keyword", "found"),
    [
        ("Maya has two Cats", "cat", True),
        ("the cat's bowl", "cat", True),
        ("a category of things", "cat", False),
        ("She drinks dark   roast", "dark roast", True),
        ("Inês’s flat", "Inês's", True),
        ("boxes of tea", "box", True),
        ("Porto-based", "Porto", True),
        ("Lisbon", "Lisbo", False),
    ],
)
def test_has_keyword(text: str, keyword: str, found: bool) -> None:
    assert has_keyword(text, keyword) is found


def test_match_rules_and_explanations() -> None:
    match = Match(require=["Porto"], any_of=["lives", "moved"], forbid=["Lisbon"])
    assert match.matches("Inês lives in Porto")
    assert not match.matches("Inês lives in Porto, not Lisbon")
    assert not match.matches("Inês likes Porto")
    assert match.failures("Inês likes Lisbon") == [
        "missing 'Porto'",
        "none of 'lives', 'moved'",
        "contains 'Lisbon'",
    ]
    assert match.describe() == "with 'Porto' with one of 'lives', 'moved' without 'Lisbon'"
    assert Match().matches("anything") and Match().describe() == "(anything)"
    assert Match.model_validate({"any": ["x"]}).any_of == ["x"]  # TOML spelling
    with pytest.raises(ValueError):
        has_keyword("x", "  ")


# --- Scenario files -------------------------------------------------------------------


def test_bundled_scenarios_load_and_cover_every_category() -> None:
    scenarios = load_scenarios()
    by_category = {c: [s for s in scenarios if s.category == c] for c in CATEGORIES}
    for category in ("recall", "update", "inference", "restraint", "forgetting"):
        assert len(by_category[category]) >= 3, category
    assert by_category["correction"] and by_category["decay"]
    assert len({s.name for s in scenarios}) == len(scenarios)
    assert all(s.checks for s in scenarios)


def test_step_parsing_and_times() -> None:
    say = parse_step({"do": "say", "text": "hi", "day": 3, "time": "07:30"})
    assert isinstance(say, Say)
    assert say.at(START) == datetime(2025, 3, 6, 7, 30, tzinfo=UTC)
    assert START.weekday() == 0  # day 0 is a Monday
    later = START + timedelta(days=5, hours=2)
    assert parse_step({"do": "say", "text": "hi"}).at(later) == later + timedelta(minutes=1)
    assert parse_step({"do": "say", "text": "x", "time": "20:00"}).at(later) == datetime(
        2025, 3, 8, 20, 0, tzinfo=UTC
    )
    check = parse_step({"check": "statement", "require": ["x"], "kind": "habit", "any": ["y"]})
    assert isinstance(check, StatementCheck) and check.kind == ["habit"]
    assert check.match.any_of == ["y"]
    for bad in (
        {"text": "no kind"},
        {"do": "say", "check": "statement", "text": "both"},
        {"do": "dance"},
        {"check": "vibes"},
        {"do": "say", "text": "x", "colour": "red"},
    ):
        with pytest.raises((ValueError, ScenarioError)):
            parse_step(bad)


def test_scenario_file_errors(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text('category = "recall"\n[[steps]]\ndo = "say"\ntext = "hi"\n')
    with pytest.raises(ScenarioError, match="at least one check"):
        load_scenario(path)
    path.write_text('category = "gossip"\n[[steps]]\ncheck = "count"\nmax = 0\n')
    with pytest.raises(ScenarioError):
        load_scenario(path)
    path.write_text("category = \n")
    with pytest.raises(ScenarioError):
        load_scenario(path)
    path.write_text('category = "recall"\n[[steps]]\ncheck = "count"\nmax = 0\n')
    assert load_scenario(path).name == "broken"


# --- The runner, with fake models -------------------------------------------------------


def scenario(category: str, *steps: dict[str, Any], name: str = "test") -> Scenario:
    return Scenario(name=name, category=category, steps=[parse_step(s) for s in steps])


def summaries(text: str = "The user has a dog named Rex.") -> list:
    return [reply(text)] * 20


async def run(tmp_path: Path, scn: Scenario, models: Models) -> ScenarioResult:
    lines: list[str] = []
    runner = ScenarioRunner(
        scn,
        learning_settings(tmp_path / "unused.db"),
        tmp_path / f"{scn.name}.db",
        registry=models.registry(),
        progress=lines.append,
        turn_timeout=5,
        jobs_timeout=10,
    )
    result = await runner.run()
    assert any("step 1:" in line for line in lines)
    return result


async def test_runner_chats_extracts_and_grades(tmp_path: Path) -> None:
    dog = "The user has a dog named Rex"
    models = Models(
        chat=[reply("What a lovely name!"), reply("Your dog is called Rex."), reply("No idea.")],
        extraction=[extraction(cand(dog, predicate="has dog", value="Rex"))],
        summary=summaries(),
    )
    scn = scenario(
        "recall",
        {"do": "say", "text": "I have a dog called Rex.", "day": 2, "time": "10:00"},
        {"do": "jobs"},
        {"check": "statement", "require": ["Rex"], "origin": "stated", "kind": "fact"},
        {"check": "statement", "require": ["cat"]},
        {"check": "statement", "require": ["Rex"], "exists": False},
        {"check": "profile", "require": ["Rex"]},
        {"check": "count", "max": 0},
        {"check": "absent", "words": ["Rex"]},
        {"check": "entity", "name": "User"},
        {"check": "answer", "question": "What's my dog called?", "require": ["Rex"]},
        {"check": "answer", "question": "And my cat?", "require": ["Tom"]},
    )
    result = await run(tmp_path, scn, models)
    assert result.error is None
    passed = {c.step: c.passed for c in result.checks}
    assert passed == {
        3: True,
        4: False,
        5: False,
        6: True,
        7: False,
        8: False,
        9: True,
        10: True,
        11: False,
    }
    failed = {c.step: c for c in result.checks}
    assert "#1 [fact, stated, 0.90, 1 source(s)] The user has a dog named Rex" in failed[4].actual
    assert failed[5].expected == "no current statement with 'Rex'"
    assert failed[7].actual.startswith("1: #1")
    assert "episodes.payload" in failed[8].actual and "statements.text" in failed[8].actual
    assert not result.passed
    assert result.transcript[:2] == [
        "[main] user: I have a dog called Rex.",
        "lucy: What a lovely name!",
    ]
    assert "lucy: Your dog is called Rex." in result.transcript
    # The answer check was asked in a fresh session, and the memory given is reported.
    assert "[eval-ask-1] user: What's my dog called?" in result.transcript
    assert "memory given: The user has a dog named Rex" in failed[10].actual
    assert failed[11].actual.startswith("No idea. [missing 'Tom']")
    assert set(result.timings) >= {"say", "jobs", "checks", "answers"}
    # The message was logged at the scenario's time; the statement too.
    with sqlite3.connect(tmp_path / "test.db") as conn:
        ts = conn.execute("SELECT ts FROM episodes WHERE type = 'user_message' LIMIT 1").fetchone()
        recorded = conn.execute("SELECT recorded_at FROM statements").fetchone()
    assert datetime.fromisoformat(ts[0]) == datetime(2025, 3, 5, 10, 0, tzinfo=UTC)
    assert datetime.fromisoformat(recorded[0]) > datetime(2025, 3, 5, 10, 0, tzinfo=UTC)
    assert datetime.fromisoformat(recorded[0]) < datetime(2025, 3, 6, tzinfo=UTC)


async def test_runner_correct_forget_rebuild_and_history(tmp_path: Path) -> None:
    galway = "The user lives in Galway"
    halden = "The user's sister works at Halden Bank"
    models = Models(
        chat=[reply("Nice."), reply("Noted.")],
        extraction=[
            extraction(cand(galway, predicate="lives in", value="Galway")),
            extraction(cand(halden, predicate="sister works at", value="Halden Bank")),
            # After the rebuild: the Galway turn, then the logged correction.
            extraction(cand(galway, predicate="lives in", value="Galway")),
            extraction(cand("The user lives in Cork", predicate="lives in", value="Cork")),
        ],
        summary=summaries("The user lives in Cork."),
    )
    scn = scenario(
        "correction",
        {"do": "say", "text": "I live in Galway."},
        {"do": "say", "text": "My sister works at Halden Bank."},
        {"do": "jobs"},
        {"do": "correct", "find": {"require": ["Galway"]}, "text": "The user lives in Cork"},
        {"check": "statement", "require": ["Cork"], "origin": "stated"},
        {"check": "superseded", "require": ["Galway"], "replaced_by": {"require": ["Cork"]}},
        {"do": "forget", "find": {"require": ["Halden"]}},
        {"check": "absent", "words": ["Halden"]},
        {"do": "rebuild"},
        {"check": "absent", "words": ["Halden"]},
        {"check": "statement", "require": ["Cork"]},
        {
            "do": "history",
            "extract": False,
            "messages": [{"day": 30, "time": "08:00", "text": "Morning swim done."}],
        },
        {"do": "maintenance", "advance_days": 2},
        {"check": "count", "min": 1},
        {"do": "correct", "find": {"require": ["Atlantis"]}, "text": "x"},
    )
    result = await run(tmp_path, scn, models)
    assert result.error is None, result.error
    outcome = [(c.step, c.check, c.passed) for c in result.checks]
    assert outcome == [
        (5, "statement", True),
        (6, "superseded", True),
        (8, "absent", True),
        (10, "absent", True),
        (11, "statement", True),
        (14, "count", True),
        (15, "correct", False),  # nothing to correct is reported, not raised
    ]
    with sqlite3.connect(tmp_path / "test.db") as conn:
        rows = conn.execute(
            "SELECT payload, extracted_at FROM episodes WHERE type = 'user_message'"
        ).fetchall()
    swim = [r for r in rows if "Morning swim" in r[0]]
    [(payload, extracted_at)] = swim
    event = UserMessage.model_validate_json(payload)
    assert event.ts == datetime(2025, 4, 2, 8, 0, tzinfo=UTC)
    assert extracted_at is not None  # extract = false
    assert any('"source":"correction"' in r[0] for r in rows)


async def test_runner_records_errors(tmp_path: Path) -> None:
    models = Models(chat=[], summary=summaries())
    scn = scenario("recall", {"do": "say", "text": "Hi"}, {"check": "count", "max": 0})
    result = await run(tmp_path, scn, models)
    assert result.error is not None and "turn ended with error" in result.error
    assert not result.passed and result.checks == []


def test_eval_settings_turn_tools_off(tmp_path: Path) -> None:
    settings = Settings.model_validate(
        {
            "llm": {"roles": {"chat": {"adapter": "ollama", "model": "m"}}},
            "memory": {"adapter": "sqlite", "path": "data/lucy.db"},
            "tools": {
                "enabled": ["wikipedia", "web_search"],
                "web_search": {"searxng_url": "http://127.0.0.1:8890"},
            },
        }
    )
    copy = settings_for(settings, tmp_path / "eval.db")
    assert copy.tools.enabled == []
    assert copy.memory is not None and copy.memory.options == {"path": str(tmp_path / "eval.db")}
    assert settings.tools.enabled == ["wikipedia", "web_search"]  # the original is untouched


def test_find_everywhere_skips_index_internals(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER, body TEXT, data BLOB)")
        conn.execute("INSERT INTO notes VALUES (1, 'about Halden Bank', NULL)")
        conn.execute("INSERT INTO notes VALUES (2, 'nothing', ?)", (b"halden in a blob",))
        conn.execute("CREATE VIRTUAL TABLE idx USING fts5(body)")
        conn.execute("INSERT INTO idx VALUES ('Halden')")
    hits = find_everywhere(db, ["halden"])
    assert [place for place, _ in hits] == ["notes.body", "notes.data"]
    assert find_everywhere(db, ["Atlantis"]) == []


# --- Reports ------------------------------------------------------------------------------


def results(passed: bool = True) -> list[ScenarioResult]:
    good = ScenarioResult("recall_a", "recall", "d", duration_s=12.0, timings={"say": 3.0})
    good.checks = [CheckResult(3, "statement", "a statement", True, "a statement", "#1 x")]
    other = ScenarioResult("update_b", "update", "d", duration_s=40.0, timings={"jobs": 30.0})
    other.checks = [
        CheckResult(4, "statement", "with 'Porto'", passed, "with 'Porto'", "#2 Lisbon"),
        CheckResult(5, "answer", "answer", True, "answer", "Porto."),
    ]
    return [good, other]


def make_report(passed: bool = True, **settings: Any) -> dict[str, Any]:
    start = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
    return report.build_report(
        results(passed),
        settings=learning_settings(Path("x.db")),
        config="lucy.toml",
        started_at=start,
        finished_at=start + timedelta(seconds=52),
    )


def test_report_contents_and_formatting(tmp_path: Path) -> None:
    data = make_report(passed=False)
    assert data["passed"] == 1 and data["total"] == 2
    assert data["categories"]["update"] == {
        "passed": 0,
        "total": 1,
        "score": 0.0,
        "checks_passed": 1,
        "checks_total": 2,
    }
    assert data["models"] == {
        "chat": "chat:?",
        "embedder": "hash:?",
        "extraction": "extraction:?",
        "summary": "summary:?",
    }
    assert set(data["prompts"]) >= {"extraction", "persona", "reflection_patterns"}
    assert all(len(h) == 12 for h in data["prompts"].values())
    text = report.format_report(data)
    assert "PASS  recall_a (recall): 1/1 checks, 12s" in text
    assert "FAIL  update_b (update): 1/2 checks, 40s" in text
    assert "x step 4 statement: with 'Porto'" in text and "found: #2 Lisbon" in text
    assert "update      0/1 scenarios (0%), 1/2 checks" in text
    assert "Time by step: jobs 30s, say 3s" in text
    path = report.write_report(data, tmp_path / "evals")
    assert path.name == "eval-20261112-090000.json"
    assert json.loads(path.read_text())["scenarios"][1]["checks"][0]["actual"] == "#2 Lisbon"


def test_compare_reports() -> None:
    old, new = make_report(passed=False), make_report(passed=True)
    new["prompts"]["extraction"] = "000000000000"
    new["scenarios"][1]["duration_s"] = 80.0
    text = report.compare_reports(old, new)
    assert "prompt extraction:" in text and "-> 000000000000" in text
    assert "update      0/1 -> 1/1 (better)" in text
    assert "update_b: now passes" in text and "step 4 statement now passes" in text
    assert "update_b: 40s -> 80s" in text
    assert "Total: 1/2 -> 2/2" in text
    same = report.compare_reports(old, old)
    assert "Scenarios: no changes in outcome" in same


async def test_run_eval_command(tmp_path: Path) -> None:
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "small_talk.toml").write_text(
        'category = "restraint"\n[[steps]]\ndo = "say"\ntext = "Hi!"\n'
        '[[steps]]\ndo = "jobs"\n[[steps]]\ncheck = "count"\nmax = 0\n'
    )
    (scenarios / "other.toml").write_text(
        'category = "recall"\n[[steps]]\ncheck = "count"\nmin = 1\n'
    )
    models = Models(chat=[reply("Hello!")], extraction=[extraction()], summary=summaries())
    out = io.StringIO()
    code = await command.run_eval(
        learning_settings(tmp_path / "unused.db"),
        out,
        category="restraint",
        out_dir=tmp_path / "evals",
        scenario_dir=scenarios,
        registry=models.registry(),
        turn_timeout=5,
        jobs_timeout=10,
    )
    text = out.getvalue()
    assert code == 0, text
    assert "[1/1] small_talk (restraint)" in text and "Total: 1/1 scenarios passed" in text
    [path] = (tmp_path / "evals").glob("eval-*.json")
    assert json.loads(path.read_text())["scenarios"][0]["name"] == "small_talk"
    assert not (tmp_path / "unused.db").exists()  # the configured database is never used
    out = io.StringIO()
    code = await command.run_eval(
        learning_settings(tmp_path / "unused.db"),
        out,
        names=["other"],
        out_dir=tmp_path / "evals",
        scenario_dir=scenarios,
        registry=Models(summary=summaries()).registry(),
        compare=path,
    )
    assert code == 1 and "Comparing" in out.getvalue() and "other: added" in out.getvalue()
    with pytest.raises(ValueError, match="unknown scenario"):
        command.select(load_scenarios(scenarios), ["nope"])
    assert (
        await command.run_eval(
            learning_settings(tmp_path / "unused.db"),
            io.StringIO(),
            category="decay",
            scenario_dir=scenarios,
        )
        == 2
    )


def test_cli_list_and_compare_need_no_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "configure_logging", lambda verbose: None)
    monkeypatch.chdir(tmp_path)  # no lucy.toml here
    assert cli.main(["eval", "--list", "--category", "decay"]) == 0
    assert "decay_stopped_vs_continuing" in capsys.readouterr().out
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps(make_report(passed=False)))
    new.write_text(json.dumps(make_report(passed=True)))
    assert cli.main(["eval", "--compare", str(old), str(new)]) == 0
    assert "update_b: now passes" in capsys.readouterr().out
    assert cli.main(["eval"]) == 2  # running needs settings
    assert "no lucy.toml" in capsys.readouterr().err
