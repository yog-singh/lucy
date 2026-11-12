"""`lucy eval`: select scenarios, run them, print and save the report."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TextIO

from lucy.core.config import Settings
from lucy.core.registry import Registry
from lucy.evals.report import build_report, compare_reports, format_report, write_report
from lucy.evals.runner import ScenarioResult, ScenarioRunner
from lucy.evals.scenario import SCENARIO_DIR, Scenario, load_scenarios

#: Where reports are written (relative to the working directory; `data/` is gitignored).
DEFAULT_OUT: Final = Path("data/evals")


def select(
    scenarios: Sequence[Scenario], names: Sequence[str] = (), category: str | None = None
) -> list[Scenario]:
    """The scenarios named (all if none), of `category` if given. Unknown names raise."""
    known = {s.name for s in scenarios}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown scenario(s): {', '.join(unknown)}")
    chosen = [s for s in scenarios if (not names or s.name in names)]
    if category is not None:
        chosen = [s for s in chosen if s.category == category]
    return chosen


def list_scenarios(scenarios: Sequence[Scenario], out: TextIO) -> None:
    for scenario in scenarios:
        checks = len(scenario.checks)
        out.write(f"{scenario.name:<34} {scenario.category:<11} {checks:>2} checks")
        out.write(f"  {scenario.description}\n" if scenario.description else "\n")


async def run_eval(
    settings: Settings,
    out: TextIO,
    *,
    names: Sequence[str] = (),
    category: str | None = None,
    config: str | None = None,
    out_dir: Path = DEFAULT_OUT,
    scenario_dir: Path = SCENARIO_DIR,
    registry: Registry | None = None,
    compare: Path | None = None,
    **runner_options: float,
) -> int:
    """Run the selected scenarios one by one, each on its own temporary database.

    Prints progress and the report to `out`, writes the JSON report to
    `out_dir`, and with `compare` also prints the changes from that earlier
    report. Returns 0 if every scenario passed, 1 if any failed, 2 if nothing
    was selected.
    """
    scenarios = select(load_scenarios(scenario_dir), names, category)
    if not scenarios:
        out.write("No scenario matches.\n")
        return 2
    previous = json.loads(compare.read_text(encoding="utf-8")) if compare else None
    started = datetime.now(UTC)
    results: list[ScenarioResult] = []

    def progress(line: str) -> None:
        out.write(line + "\n")
        out.flush()

    with tempfile.TemporaryDirectory(prefix="lucy-eval-") as tmp:
        for index, scenario in enumerate(scenarios, start=1):
            progress(f"[{index}/{len(scenarios)}] {scenario.name} ({scenario.category})")
            runner = ScenarioRunner(
                scenario,
                settings,
                Path(tmp) / f"{scenario.name}.db",
                registry=registry,
                progress=progress,
                **runner_options,
            )
            result = await runner.run()
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            progress(f"  {status} in {result.duration_s:.0f}s")
            if result.error:
                progress(f"  error: {result.error}")
    report = build_report(
        results,
        settings=settings,
        config=config,
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    path = write_report(report, out_dir)
    out.write("\n" + format_report(report) + "\n")
    out.write(f"Report: {path}\n")
    if previous is not None:
        out.write("\n" + compare_reports(previous, report) + "\n")
    return 0 if report["passed"] == report["total"] else 1
