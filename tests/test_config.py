from pathlib import Path

import pytest

from lucy.core.config import (
    AdapterConfig,
    ConfigError,
    LearningSettings,
    RetrievalSettings,
    load_settings,
)
from lucy.memory import retrieval
from lucy.memory.decay import DEFAULT_GRACE, DEFAULT_HALF_LIFE

EXAMPLE = Path(__file__).resolve().parent.parent / "lucy.example.toml"


def test_loads_example() -> None:
    settings = load_settings(EXAMPLE)
    chat = settings.llm.roles["chat"]
    assert chat.adapter == "ollama"
    no_thinking = {"reasoning_effort": "none"}
    assert chat.options == {
        "model": "gemma4:12b-mlx",
        "context_window": 32768,
        "extra": no_thinking,
    }
    assert set(settings.llm.roles) == {"chat", "extraction", "summary"}
    assert settings.llm.roles["extraction"].options == {
        "model": "gemma4:12b-mlx",
        "context_window": 32768,
        "temperature": 0,
        "extra": {"reasoning_effort": "none", "max_tokens": 2048},
    }
    assert settings.llm.roles["summary"].options == {
        "model": "gemma4:12b-mlx",
        "context_window": 32768,
        "extra": {"reasoning_effort": "none", "max_tokens": 1024},
    }
    assert settings.learning == LearningSettings(
        enabled=True,
        maintenance_interval_hours=24,
        extract_max_turns=4,
        decay_grace_days=30,
        decay_half_life_days=30,
    )
    # The example documents the defaults: keep them in step with the code.
    assert settings.learning.decay_grace_days == DEFAULT_GRACE.days
    assert settings.learning.decay_half_life_days == DEFAULT_HALF_LIFE.days
    assert settings.retrieval == RetrievalSettings(
        vector_min_similarity=retrieval.VECTOR_MIN_SIMILARITY,
        vector_margin=retrieval.VECTOR_MARGIN,
        keyword_min_similarity=retrieval.KEYWORD_MIN_SIMILARITY,
        entity_min_similarity=retrieval.ENTITY_MIN_SIMILARITY,
        max_statements=retrieval.MAX_STATEMENTS,
    )
    assert settings.stt == AdapterConfig(adapter="mlx_whisper", options={"model": "large-v3-turbo"})
    assert settings.tts is not None and settings.tts.options == {"voice": "af_heart"}
    assert settings.embedder == AdapterConfig(
        adapter="ollama",
        options={
            "model": "embeddinggemma-2",
            "dimensions": 768,
            "query_prefix": "query: ",
            "document_prefix": "passage: ",
        },
    )
    assert settings.memory is not None and settings.memory.options == {"path": "data/lucy.db"}


def test_missing_chat_role(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text('[llm.roles]\nsummary = { adapter = "ollama", model = "m" }\n')
    with pytest.raises(ConfigError, match="chat"):
        load_settings(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text("[llm.roles\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_settings(path)


def test_missing_adapter_key(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text('[llm.roles]\nchat = { model = "m" }\n')
    with pytest.raises(ConfigError, match="adapter"):
        load_settings(path)


def test_unknown_section_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text('[llm.roles]\nchat = { adapter = "a" }\n\n[sttt]\nadapter = "x"\n')
    with pytest.raises(ConfigError, match="sttt"):
        load_settings(path)


def test_minimal_settings(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text('[llm.roles]\nchat = { adapter = "a" }\n')
    settings = load_settings(path)
    assert settings.llm.roles["chat"].options == {}
    assert settings.stt is None and settings.memory is None
    assert settings.learning == LearningSettings()
    assert settings.learning.enabled
    assert settings.retrieval == RetrievalSettings()
    assert settings.retrieval.model_dump(exclude_none=True) == {}  # the code's defaults apply
    assert settings.learning.decay_grace_days is None


def test_learning_section(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(
        '[llm.roles]\nchat = { adapter = "a" }\n\n[learning]\nenabled = false\n'
        "maintenance_interval_hours = 0.5\n"
    )
    learning = load_settings(path).learning
    assert not learning.enabled
    assert learning.maintenance_interval_hours == 0.5
    assert learning.extract_max_turns == 4


@pytest.mark.parametrize(
    "line", ["maintenance_interval_hours = 0", "extract_max_turns = 0", "colour = 'blue'"]
)
def test_invalid_learning_section(tmp_path: Path, line: str) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(f'[llm.roles]\nchat = {{ adapter = "a" }}\n\n[learning]\n{line}\n')
    with pytest.raises(ConfigError):
        load_settings(path)


async def test_example_adapters_accept_their_options() -> None:
    """The example's LLM and embedder sections construct the registered adapters."""
    from lucy.core import registry

    registry.load_adapters(["lucy.adapters.llm_openai_compat", "lucy.adapters.embed_openai_compat"])
    settings = load_settings(EXAMPLE)
    for config in settings.llm.roles.values():
        llm = registry.create("llm", config.adapter, **config.options)
        assert llm.capabilities.json_schema is True
        await llm.aclose()
    assert settings.embedder is not None
    embedder = registry.create("embedder", settings.embedder.adapter, **settings.embedder.options)
    assert embedder.capabilities.dimensions == 768
    await embedder.aclose()


def test_retrieval_section(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(
        '[llm.roles]\nchat = { adapter = "a" }\n\n[retrieval]\nvector_min_similarity = 0.5\n'
        "max_statements = 12\n\n[learning]\ndecay_grace_days = 7\n"
    )
    settings = load_settings(path)
    assert settings.retrieval.model_dump(exclude_none=True) == {
        "vector_min_similarity": 0.5,
        "max_statements": 12,
    }
    assert settings.learning.decay_grace_days == 7
    assert settings.learning.decay_half_life_days is None


@pytest.mark.parametrize(
    "section",
    [
        "[retrieval]\nmax_statements = 0",
        "[retrieval]\nvector_margin = -0.1",
        "[retrieval]\nvector_min_similarity = 1.5",
        "[retrieval]\nrrf = 3",
        "[learning]\ndecay_half_life_days = 0",
        "[learning]\ndecay_grace_days = -1",
    ],
)
def test_invalid_memory_tuning(tmp_path: Path, section: str) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(f'[llm.roles]\nchat = {{ adapter = "a" }}\n\n{section}\n')
    with pytest.raises(ConfigError):
        load_settings(path)
