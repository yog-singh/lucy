"""Eval reports: the terminal summary, the JSON file, and comparing two runs.

The JSON file (one per run, under `data/evals/` by default) holds everything
needed to compare runs: when, which models and embedder, a hash of every prompt
file, and per scenario its checks (expected and actual), timings and errors.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from lucy.core.config import Settings
from lucy.evals.runner import ScenarioResult
from lucy.evals.scenario import CATEGORIES

#: Version of the JSON layout.
REPORT_VERSION: Final = 1
PROMPT_DIR: Final = Path(__file__).parent.parent / "prompts"


def prompt_hashes(directory: Path = PROMPT_DIR) -> dict[str, str]:
    """The first 12 hex digits of the SHA-256 of every prompt file, by name."""
    return {
        path.stem: hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        for path in sorted(directory.glob("*.md"))
    }


def models_used(settings: Settings) -> dict[str, str]:
    """The model behind every LLM role and the embedder ("role": "adapter:model")."""
    used = {
        role: f"{config.adapter}:{config.options.get('model', '?')}"
        for role, config in sorted(settings.llm.roles.items())
    }
    if settings.embedder is not None:
        options = settings.embedder.options
        used["embedder"] = f"{settings.embedder.adapter}:{options.get('model', '?')}"
    return used


def category_scores(results: Sequence[ScenarioResult]) -> dict[str, dict[str, Any]]:
    """Per category: scenarios passed, total and the share passed."""
    scores: dict[str, dict[str, Any]] = {}
    for category in [*CATEGORIES, *sorted({r.category for r in results} - set(CATEGORIES))]:
        mine = [r for r in results if r.category == category]
        if mine:
            passed = sum(r.passed for r in mine)
            scores[category] = {
                "passed": passed,
                "total": len(mine),
                "score": round(passed / len(mine), 3),
                "checks_passed": sum(c.passed for r in mine for c in r.checks),
                "checks_total": sum(len(r.checks) for r in mine),
            }
    return scores


def build_report(
    results: Sequence[ScenarioResult],
    *,
    settings: Settings,
    config: str | None,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    return {
        "version": REPORT_VERSION,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": round((finished_at - started_at).total_seconds(), 1),
        "config": config,
        "models": models_used(settings),
        "retrieval": settings.retrieval.model_dump(exclude_none=True),
        "prompts": prompt_hashes(),
        "passed": sum(r.passed for r in results),
        "total": len(results),
        "categories": category_scores(results),
        "scenarios": [
            {
                **{k: v for k, v in asdict(r).items() if k != "timings"},
                "passed": r.passed,
                "duration_s": round(r.duration_s, 1),
                "timings": {k: round(v, 1) for k, v in r.timings.items()},
            }
            for r in results
        ],
    }


def write_report(report: dict[str, Any], directory: Path) -> Path:
    """Write `report` as `eval-<start time>.json` in `directory` (created if needed)."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(report["started_at"]).strftime("%Y%m%d-%H%M%S")
    path = directory / f"eval-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _clip(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_report(report: dict[str, Any]) -> str:
    """The human-readable summary of a report."""
    lines = []
    for scenario in report["scenarios"]:
        status = "PASS" if scenario["passed"] else "FAIL"
        checks = scenario["checks"]
        ok = sum(c["passed"] for c in checks)
        lines.append(
            f"{status}  {scenario['name']} ({scenario['category']}):"
            f" {ok}/{len(checks)} checks, {scenario['duration_s']:.0f}s"
        )
        if scenario["error"]:
            lines.append(f"      error: {scenario['error']}")
        for check in checks:
            if not check["passed"]:
                lines.append(f"      x step {check['step']} {check['check']}: {check['expected']}")
                lines.append(f"        found: {_clip(check['actual'])}")
    lines.append("")
    lines.append("By category:")
    for category, score in report["categories"].items():
        lines.append(
            f"  {category:<11} {score['passed']}/{score['total']} scenarios"
            f" ({score['score']:.0%}), {score['checks_passed']}/{score['checks_total']} checks"
        )
    lines.append(
        f"Total: {report['passed']}/{report['total']} scenarios passed"
        f" in {report['duration_s']:.0f}s"
    )
    timings: dict[str, float] = {}
    for scenario in report["scenarios"]:
        for kind, seconds in scenario["timings"].items():
            timings[kind] = timings.get(kind, 0.0) + seconds
    if timings:
        lines.append(
            "Time by step: "
            + ", ".join(f"{k} {v:.0f}s" for k, v in sorted(timings.items(), key=lambda x: -x[1]))
        )
    lines.append("Models: " + ", ".join(f"{k}={v}" for k, v in report["models"].items()))
    lines.append("Prompts: " + ", ".join(f"{k}={v}" for k, v in report["prompts"].items()))
    return "\n".join(lines)


def compare_reports(old: dict[str, Any], new: dict[str, Any]) -> str:
    """What changed from `old` to `new`: setup, category scores, scenarios and checks."""
    lines = [f"Comparing {old['started_at']} (old) with {new['started_at']} (new)"]
    for label, key in (("model", "models"), ("prompt", "prompts"), ("retrieval", "retrieval")):
        before, after = old.get(key, {}), new.get(key, {})
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                lines.append(f"  {label} {name}: {before.get(name)} -> {after.get(name)}")
    lines.append("Categories:")
    for category in sorted(
        set(old["categories"]) | set(new["categories"]),
        key=lambda c: CATEGORIES.index(c) if c in CATEGORIES else len(CATEGORIES),
    ):
        a, b = old["categories"].get(category), new["categories"].get(category)
        fmt = lambda s: f"{s['passed']}/{s['total']}" if s else "-"  # noqa: E731
        change = ""
        if a and b and a["score"] != b["score"]:
            change = " (better)" if b["score"] > a["score"] else " (worse)"
        lines.append(f"  {category:<11} {fmt(a)} -> {fmt(b)}{change}")
    before = {s["name"]: s for s in old["scenarios"]}
    after = {s["name"]: s for s in new["scenarios"]}
    changes = []
    for name in sorted(set(before) | set(after)):
        a, b = before.get(name), after.get(name)
        if a is None or b is None:
            changes.append(f"  {name}: {'added' if a is None else 'removed'}")
            continue
        if a["passed"] != b["passed"]:
            changes.append(f"  {name}: {'now passes' if b['passed'] else 'now FAILS'}")
        checks_a = {(c["step"], c["expected"]): c for c in a["checks"]}
        for check in b["checks"]:
            earlier = checks_a.get((check["step"], check["expected"]))
            if earlier is not None and earlier["passed"] != check["passed"]:
                state = "passes" if check["passed"] else "fails"
                changes.append(f"    step {check['step']} {check['check']} now {state}")
                if not check["passed"]:
                    changes.append(f"      found: {_clip(check['actual'], 300)}")
        if abs(b["duration_s"] - a["duration_s"]) >= max(5.0, 0.2 * a["duration_s"]):
            changes.append(f"  {name}: {a['duration_s']:.0f}s -> {b['duration_s']:.0f}s")
    lines.append("Scenarios:" if changes else "Scenarios: no changes in outcome")
    lines += changes
    lines.append(
        f"Total: {old['passed']}/{old['total']} -> {new['passed']}/{new['total']};"
        f" {old['duration_s']:.0f}s -> {new['duration_s']:.0f}s"
    )
    return "\n".join(lines)
