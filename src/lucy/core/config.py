"""Settings loaded from a TOML file (ADR 0007).

Each section names an adapter plus free-form options for its constructor. LLMs
are configured per role; `chat` is required.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigError(ValueError):
    """The settings file is missing, unreadable or invalid."""


class AdapterConfig(BaseModel):
    """Which adapter to use for a port, and the options passed to its constructor.

    In TOML every key other than `adapter` is collected into `options`, e.g.
    `{ adapter = "ollama", model = "qwen3:14b" }` gives
    `options == {"model": "qwen3:14b"}`. In Python it can also be built as
    `AdapterConfig(adapter="ollama", options={...})`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    adapter: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_options(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if set(data) <= {"adapter", "options"} and isinstance(data.get("options", {}), dict):
            return data  # already in canonical form
        options = {key: value for key, value in data.items() if key != "adapter"}
        return {"adapter": data.get("adapter"), "options": options}


class LLMSettings(BaseModel):
    """LLM configuration: one adapter config per role."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roles: dict[str, AdapterConfig]

    @field_validator("roles")
    @classmethod
    def _require_chat(cls, roles: dict[str, AdapterConfig]) -> dict[str, AdapterConfig]:
        if "chat" not in roles:
            raise ValueError("[llm.roles] must define a 'chat' role")
        return roles


class LearningSettings(BaseModel):
    """The learning loop (ADR 0012): background extraction and maintenance.

    Learning runs only when an embedder is configured and memory is the SQLite
    store; `enabled = false` turns it off even then (chat then replays the
    session's conversation only).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    maintenance_interval_hours: float = Field(default=24, gt=0)
    """How often decay, entity summaries, duplicate merging and the profile rewrite run."""
    extract_max_turns: int = Field(default=4, ge=1)
    """Turns one extraction job processes before handing over to a follow-up job."""
    decay_grace_days: float | None = Field(default=None, ge=0)
    """Days an inferred statement keeps its confidence after it was last observed.
    None: the memory module's default (`lucy.memory.decay.DecayPolicy`)."""
    decay_half_life_days: float | None = Field(default=None, gt=0)
    """After the grace period, confidence halves every this many days of not being
    observed. None: the memory module's default."""


class RetrievalSettings(BaseModel):
    """Per-turn memory retrieval thresholds (`lucy.memory.retrieval.Retriever`).

    Every field is optional; None (absent) keeps the retriever's default. The
    similarity values are cosine similarities and depend on the embedding model
    (the defaults were measured for `embeddinggemma-2` with the "query: " /
    "passage: " prefixes): re-measure them when the embedder changes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vector_min_similarity: float | None = Field(default=None, ge=-1, le=1)
    """Absolute floor: vector matches less similar than this are dropped."""
    vector_margin: float | None = Field(default=None, ge=0)
    """Relative floor: vector matches more than this below the best match are dropped
    (2 or more turns it off)."""
    keyword_min_similarity: float | None = Field(default=None, ge=-1, le=1)
    """Keyword matches whose vector similarity to the query is below this are dropped
    (-1 turns the check off)."""
    entity_min_similarity: float | None = Field(default=None, ge=-1, le=1)
    """Entities similar to the query (not named in it) are expanded only from this
    similarity up."""
    max_statements: int | None = Field(default=None, ge=1)
    """At most this many retrieved statements are given to a turn."""


class Settings(BaseModel):
    """All of Lucy's settings. Only `llm` is required; other sections are optional."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    llm: LLMSettings
    stt: AdapterConfig | None = None
    tts: AdapterConfig | None = None
    embedder: AdapterConfig | None = None
    memory: AdapterConfig | None = None
    learning: LearningSettings = Field(default_factory=LearningSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)


def load_settings(path: str | Path) -> Settings:
    """Read and validate settings from the TOML file at `path`.

    Raises `ConfigError` if the file does not exist, is not valid TOML, or does
    not match the expected shape.
    """
    path = Path(path)
    try:
        with path.open("rb") as file:
            raw = tomllib.load(file)
    except FileNotFoundError:
        raise ConfigError(f"settings file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read settings file {path}: {exc}") from exc
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid settings in {path}:\n{exc}") from exc
