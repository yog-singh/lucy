"""Scenario definitions: TOML files under `scenarios/`, one scenario per file.

```toml
name = "update_moved_city"            # defaults to the file name
category = "update"                   # recall, update, inference, restraint,
                                      # forgetting, correction or decay
description = "..."

[[steps]]
do = "say"                            # a full chat turn
text = "I live in Lisbon."
session = "main"                      # default "main"
day = 0                               # days after the scenario start (a Monday)
time = "09:00"                        # UTC; both optional (default: a minute later)

[[steps]]
check = "statement"
require = ["Lisbon"]
kind = "fact"
```

Steps (`do = ...`):

- `say`: `text`, `session`, `day`, `time`: one turn through the orchestrator.
- `history`: `session`, `extract` (default true), `messages = [{text, day, time}]`:
  user messages logged directly, each as a complete turn without a reply.
- `jobs`: run background jobs until none is due (extraction first).
- `maintenance`: `advance_days` (default 0): move the clock, then run every
  maintenance job (decay, merges, reflection, summaries, profile) and the
  jobs they cause.
- `advance`: `days`: move the clock only.
- `correct`: `find` (a match, see below), `text`: `lucy memory correct` on the
  best current statement matching `find`.
- `forget`: `find`: `lucy memory forget` on every current statement matching it.
- `rebuild`: `lucy memory rebuild`, then the jobs and maintenance it needs.

Checks (`check = ...`); keywords use `lucy.evals.matching` (`require`, `any`,
`forbid`):

- `statement`: a statement matching the keywords and, optionally, `kind` (one
  or a list), `origin`, `min_confidence`, `max_confidence` and `min_sources`
  exists (or, with `exists = false`, none does). Current statements only
  unless `current = false`.
- `superseded`: a statement matching the keywords is no longer current and has
  an end date; with `replaced_by` (a match), it was superseded by a statement
  matching that.
- `entity`: an entity called `name` (or with it as an alias) exists, or with
  `exists = false` none does.
- `count`: the number of current statements (optionally of `kind`) is between
  `min` and `max`.
- `absent`: none of `words` occurs anywhere in the database (every text column
  of every table: episodes, statements, entities, profiles, traces, jobs,
  metadata).
- `profile`: the current profile matches the keywords.
- `answer`: Lucy's reply to `question`, asked in a fresh session (or `session`),
  matches the keywords.
"""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from lucy.evals.matching import Match

#: Where the bundled scenarios live.
SCENARIO_DIR: Final = Path(__file__).parent / "scenarios"
#: Day 0 of every scenario: a Monday, far enough in the past that months of
#: simulated time stay before today (jobs are due by the real clock).
START: Final = datetime(2025, 3, 3, 9, 0, tzinfo=UTC)
#: The categories of docs/memory.md "Measuring improvement", plus two more.
CATEGORIES: Final = (
    "recall",
    "update",
    "inference",
    "restraint",
    "forgetting",
    "correction",
    "decay",
)


class ScenarioError(ValueError):
    """A scenario file is unreadable or invalid."""


def _listify(value: Any) -> Any:
    return [value] if isinstance(value, str) else value


Kinds = Annotated[list[str] | None, BeforeValidator(_listify)]


class _Step(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Timed(_Step):
    day: float | None = None
    time: str | None = None

    def at(self, now: datetime) -> datetime:
        """When this happens: `day` and `time` after `START`, or a minute after `now`."""
        if self.day is None and self.time is None:
            return now + timedelta(minutes=1)
        day = self.day if self.day is not None else (now - START).days
        clock = time.fromisoformat(self.time) if self.time else START.timetz()
        base = datetime.combine(START.date(), clock.replace(tzinfo=None), tzinfo=UTC)
        return base + timedelta(days=day)


class Say(_Timed):
    do: Literal["say"]
    text: str
    session: str = "main"


class Logged(_Timed):
    text: str


class History(_Step):
    do: Literal["history"]
    session: str = "main"
    extract: bool = True
    messages: list[Logged]


class Jobs(_Step):
    do: Literal["jobs"]


class Maintenance(_Step):
    do: Literal["maintenance"]
    advance_days: float = 0


class Advance(_Step):
    do: Literal["advance"]
    days: float


class Correct(_Step):
    do: Literal["correct"]
    find: Match
    text: str


class Forget(_Step):
    do: Literal["forget"]
    find: Match


class Rebuild(_Step):
    do: Literal["rebuild"]


class _Keywords(_Step):
    require: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list, alias="any")
    forbid: list[str] = Field(default_factory=list)

    @property
    def match(self) -> Match:
        return Match(require=self.require, any_of=self.any_of, forbid=self.forbid)


class StatementCheck(_Keywords):
    check: Literal["statement"]
    kind: Kinds = None
    origin: Literal["stated", "inferred"] | None = None
    current: bool = True
    exists: bool = True
    min_confidence: float | None = None
    max_confidence: float | None = None
    min_sources: int | None = None


class SupersededCheck(_Keywords):
    check: Literal["superseded"]
    replaced_by: Match | None = None


class EntityCheck(_Step):
    check: Literal["entity"]
    name: str
    exists: bool = True


class CountCheck(_Step):
    check: Literal["count"]
    kind: Kinds = None
    min: int = 0
    max: int | None = None


class AbsentCheck(_Step):
    check: Literal["absent"]
    words: list[str]


class ProfileCheck(_Keywords):
    check: Literal["profile"]


class AnswerCheck(_Keywords):
    check: Literal["answer"]
    question: str
    session: str | None = None


Action = Say | History | Jobs | Maintenance | Advance | Correct | Forget | Rebuild
Check = (
    StatementCheck
    | SupersededCheck
    | EntityCheck
    | CountCheck
    | AbsentCheck
    | ProfileCheck
    | AnswerCheck
)
Step = Action | Check

_ACTIONS: dict[str, type[BaseModel]] = {
    "say": Say,
    "history": History,
    "jobs": Jobs,
    "maintenance": Maintenance,
    "advance": Advance,
    "correct": Correct,
    "forget": Forget,
    "rebuild": Rebuild,
}
_CHECKS: dict[str, type[BaseModel]] = {
    "statement": StatementCheck,
    "superseded": SupersededCheck,
    "entity": EntityCheck,
    "count": CountCheck,
    "absent": AbsentCheck,
    "profile": ProfileCheck,
    "answer": AnswerCheck,
}


def parse_step(data: dict[str, Any]) -> Step:
    """One step from its TOML table (dispatching on `do` or `check`)."""
    if ("do" in data) == ("check" in data):
        raise ValueError("a step needs exactly one of `do` or `check`")
    if "do" in data:
        kind = _ACTIONS.get(data["do"])
        if kind is None:
            raise ValueError(f"unknown action {data['do']!r} (known: {', '.join(_ACTIONS)})")
    else:
        kind = _CHECKS.get(data["check"])
        if kind is None:
            raise ValueError(f"unknown check {data['check']!r} (known: {', '.join(_CHECKS)})")
    return kind.model_validate(data)  # type: ignore[return-value]


class Scenario(BaseModel):
    """A named sequence of steps in one category."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    category: Literal[
        "recall", "update", "inference", "restraint", "forgetting", "correction", "decay"
    ]
    description: str = ""
    steps: list[Any]

    @property
    def checks(self) -> list[Check]:
        return [s for s in self.steps if isinstance(s, tuple(_CHECKS.values()))]


def load_scenario(path: Path) -> Scenario:
    """Read and validate one scenario file. Raises `ScenarioError`."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ScenarioError(f"{path}: {exc}") from exc
    data.setdefault("name", path.stem)
    raw_steps = data.get("steps", [])
    try:
        steps = [parse_step(step) for step in raw_steps]
        scenario = Scenario.model_validate({**data, "steps": steps})
    except (ValidationError, ValueError) as exc:
        raise ScenarioError(f"{path}: {exc}") from exc
    if not scenario.checks:
        raise ScenarioError(f"{path}: a scenario needs at least one check")
    return scenario


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    """Every `*.toml` scenario in `directory`, by category order then name."""
    scenarios = [load_scenario(p) for p in sorted(directory.glob("*.toml"))]
    names = [s.name for s in scenarios]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ScenarioError(f"duplicate scenario names: {', '.join(duplicates)}")
    return sorted(scenarios, key=lambda s: (CATEGORIES.index(s.category), s.name))
