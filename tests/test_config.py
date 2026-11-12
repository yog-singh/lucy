from pathlib import Path

import pytest

from lucy.core.config import (
    AdapterConfig,
    ConfigError,
    LearningSettings,
    LLMRoleConfig,
    LLMSettings,
    ResidencySettings,
    RetrievalSettings,
    VoiceSettings,
    WakeSettings,
    load_settings,
)
from lucy.memory import retrieval
from lucy.memory.decay import DEFAULT_GRACE, DEFAULT_HALF_LIFE

EXAMPLE = Path(__file__).resolve().parent.parent / "lucy.example.toml"


def test_loads_example() -> None:
    settings = load_settings(EXAMPLE)
    chat = settings.llm.roles["chat"]
    assert chat.adapter == "ollama"
    assert chat.options == {
        "model": "gemma4:12b-mlx",
        "context_window": 32768,
        "hidden_loop_repeats": 4,
        "extra": {"reasoning_effort": "none", "max_tokens": 1024},
    }
    assert (chat.fallback, chat.first_output_timeout) == ("chat_deep", 60)
    deep = settings.llm.roles["chat_deep"]
    assert deep.options == {
        "model": "gemma4:12b-mlx",
        "context_window": 32768,
        "extra": {"reasoning_effort": "low", "max_tokens": 4096},
    }
    assert deep.fallback is None and deep.first_output_timeout is None
    assert set(settings.llm.roles) == {"chat", "chat_deep", "extraction", "summary"}
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
    assert settings.stt == AdapterConfig(
        adapter="parakeet_mlx",
        options={"model": "mlx-community/parakeet-tdt-0.6b-v2", "cache_limit_mb": 256},
    )
    assert settings.tts == AdapterConfig(
        adapter="ollama", options={"model": "phonon", "voice": "default"}
    )
    # The example's [voice] and [residency] sections document the defaults.
    assert settings.voice == VoiceSettings()
    assert settings.wake == WakeSettings()
    assert settings.residency == ResidencySettings()
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


def test_voice_section(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(
        '[llm.roles]\nchat = { adapter = "a" }\n\n[voice]\nendpoint_silence_ms = 450\n'
        'tail_ms = 0\ninput_device = "USB"\noutput_device = 3\nvad_threshold = 0.6\n'
    )
    voice = load_settings(path).voice
    assert voice.endpoint_silence_ms == 450 and voice.tail_ms == 0
    assert voice.input_device == "USB" and voice.output_device == 3
    assert voice.vad_threshold == 0.6
    assert voice.min_speech_ms == 160 and voice.preroll_ms == 300
    assert voice.max_utterance_s == 30


def test_voice_defaults_without_a_section(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text('[llm.roles]\nchat = { adapter = "a" }\n')
    assert load_settings(path).voice == VoiceSettings()


@pytest.mark.parametrize(
    "line",
    [
        "endpoint_silence_ms = 50",
        "vad_threshold = 1.0",
        "tail_ms = -1",
        "max_utterance_s = 0",
        "min_speech_ms = 10",
        "unknown = 1",
    ],
)
def test_invalid_voice_section(tmp_path: Path, line: str) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(f'[llm.roles]\nchat = {{ adapter = "a" }}\n\n[voice]\n{line}\n')
    with pytest.raises(ConfigError, match="voice"):
        load_settings(path)


async def test_example_speech_adapters_accept_their_options() -> None:
    """The example's [stt] and [tts] sections construct the registered adapters (nothing
    is loaded: Parakeet loads lazily)."""
    from lucy.core import registry

    registry.load_adapters(["lucy.adapters.stt_parakeet_mlx", "lucy.adapters.tts_openai_compat"])
    settings = load_settings(EXAMPLE)
    assert settings.stt is not None and settings.tts is not None
    stt = registry.create("stt", settings.stt.adapter, **settings.stt.options)
    assert not stt.loaded
    await stt.aclose()
    tts = registry.create("tts", settings.tts.adapter, **settings.tts.options)
    assert tts.voice == "default"
    await tts.aclose()


def test_role_fallback_and_first_output_timeout(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(
        "[llm.roles]\n"
        'chat = { adapter = "a", model = "m", fallback = "deep", first_output_timeout = 2.5, '
        'after_tool = "deep" }\n'
        'deep = { adapter = "a", model = "m" }\n'
    )
    roles = load_settings(path).llm.roles
    assert roles["chat"].options == {"model": "m"}  # not passed to the adapter
    assert (roles["chat"].fallback, roles["chat"].first_output_timeout) == ("deep", 2.5)
    assert roles["chat"].after_tool == "deep"
    assert (roles["deep"].fallback, roles["deep"].first_output_timeout) == (None, None)


@pytest.mark.parametrize(
    ("chat", "message"),
    [
        ('{ adapter = "a", fallback = "nope" }', "not a configured role"),
        ('{ adapter = "a", fallback = "chat" }', "its own fallback"),
        ('{ adapter = "a", first_output_timeout = 0 }', "greater than 0"),
        ('{ adapter = "a", after_tool = "nope" }', "after_tool 'nope', which is not"),
        ('{ adapter = "a", after_tool = "chat" }', "its own after_tool"),
    ],
)
def test_invalid_role_fallback_settings(tmp_path: Path, chat: str, message: str) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(f"[llm.roles]\nchat = {chat}\n")
    with pytest.raises(ConfigError, match=message):
        load_settings(path)


def test_plain_adapter_configs_are_accepted_as_roles() -> None:
    settings = LLMSettings(roles={"chat": AdapterConfig(adapter="a", options={"model": "m"})})
    chat = settings.roles["chat"]
    assert isinstance(chat, LLMRoleConfig) and chat.options == {"model": "m"}
    assert chat.fallback is None


def test_context_layout_defaults_to_stable_prefix(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text('[llm.roles]\nchat = { adapter = "a" }\n')
    assert load_settings(path).context.layout == "stable_prefix"


def test_context_layout_section(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text('[llm.roles]\nchat = { adapter = "a" }\n\n[context]\nlayout = "system"\n')
    assert load_settings(path).context.layout == "system"


@pytest.mark.parametrize("line", ['layout = "sideways"', "colour = 'blue'"])
def test_invalid_context_section(tmp_path: Path, line: str) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(f'[llm.roles]\nchat = {{ adapter = "a" }}\n\n[context]\n{line}\n')
    with pytest.raises(ConfigError):
        load_settings(path)


def test_residency_defaults(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text('[llm.roles]\nchat = { adapter = "a" }\n')
    residency = load_settings(path).residency
    assert residency.enabled and residency.idle_seconds == 1.0
    assert residency.min_size_gb is None


def test_residency_section(tmp_path: Path) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(
        '[llm.roles]\nchat = { adapter = "a" }\n\n[residency]\n'
        "enabled = false\nidle_seconds = 2\ntext_quiet_seconds = 30\nmin_size_gb = 12\n"
    )
    residency = load_settings(path).residency
    assert (residency.enabled, residency.idle_seconds) == (False, 2)
    assert (residency.text_quiet_seconds, residency.min_size_gb) == (30, 12)


@pytest.mark.parametrize("line", ["idle_seconds = 0", "min_size_gb = -1", "colour = 'blue'"])
def test_invalid_residency_section(tmp_path: Path, line: str) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text(f'[llm.roles]\nchat = {{ adapter = "a" }}\n\n[residency]\n{line}\n')
    with pytest.raises(ConfigError):
        load_settings(path)
