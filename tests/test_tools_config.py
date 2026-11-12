"""The `[tools]` settings and the tool catalogue."""

from pathlib import Path

import pytest

from lucy.core.config import (
    ConfigError,
    LLMSettings,
    Settings,
    ToolsSettings,
    load_settings,
)
from lucy.core.ports import Tool
from lucy.core.types import Risk
from lucy.tools import CATALOGUE, build_tools

LLM = '[llm.roles]\nchat = { adapter = "fake" }\n'


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "lucy.toml"
    path.write_text(LLM + text)
    return path


def test_no_tools_by_default(tmp_path: Path) -> None:
    settings = load_settings(write(tmp_path, ""))
    assert settings.tools.enabled == []
    assert build_tools(settings) == []
    assert settings.tools.wikipedia.language == "en"
    assert settings.tools.web_search.searxng_url is None
    assert settings.tools.fetch_page.allow_private_hosts == []
    assert settings.tools.fetch_page.max_chars == 6000
    assert settings.tools.fetch_page.timeout == 10


def test_tools_section_is_parsed(tmp_path: Path) -> None:
    settings = load_settings(
        write(
            tmp_path,
            """
[tools]
enabled = ["wikipedia", "web_search", "fetch_page"]

[tools.wikipedia]
language = "de"

[tools.web_search]
searxng_url = "http://127.0.0.1:8890"
max_results = 3

[tools.fetch_page]
allow_private_hosts = ["homeassistant.lan", "192.168.1.0/24"]
max_chars = 3000
""",
        )
    )
    tools = build_tools(settings)
    assert [t.name for t in tools] == ["wikipedia", "web_search", "fetch_page"]
    assert all(isinstance(t, Tool) and t.risk is Risk.READ for t in tools)
    assert settings.tools.wikipedia.language == "de"
    assert settings.tools.web_search.max_results == 3
    assert settings.tools.fetch_page.max_chars == 3000


def test_unknown_tool_name_fails_at_load(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"unknown tool\(s\) 'lights'.*known tools: wikipedia"):
        load_settings(write(tmp_path, '[tools]\nenabled = ["wikipedia", "lights"]\n'))


def test_web_search_without_searxng_url_fails_at_load(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="web_search is enabled but .* no searxng_url"):
        load_settings(write(tmp_path, '[tools]\nenabled = ["web_search"]\n'))


def test_searxng_url_must_be_http(tmp_path: Path) -> None:
    text = '[tools]\nenabled = ["web_search"]\n[tools.web_search]\nsearxng_url = "searx.local"\n'
    with pytest.raises(ConfigError, match="searxng_url"):
        load_settings(write(tmp_path, text))


def test_unknown_option_and_duplicates_fail(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="langauge"):
        load_settings(write(tmp_path, '[tools.wikipedia]\nlangauge = "en"\n'))
    with pytest.raises(ConfigError, match="twice"):
        load_settings(write(tmp_path, '[tools]\nenabled = ["wikipedia", "wikipedia"]\n'))
    with pytest.raises(ConfigError, match="tools.lights"):
        load_settings(write(tmp_path, "[tools.lights]\nroom = 'x'\n"))
    with pytest.raises(ConfigError, match="units"):
        load_settings(write(tmp_path, "[tools.weather]\nunits = 'kelvin'\n"))


def test_catalogue_matches_the_settings_tables() -> None:
    assert sorted(CATALOGUE) == sorted(ToolsSettings.known_tools())


def test_build_tools_order_follows_enabled() -> None:
    tools = ToolsSettings(enabled=["fetch_page", "wikipedia"])
    settings = Settings(llm=LLMSettings(roles={"chat": {"adapter": "fake"}}), tools=tools)
    assert [t.name for t in build_tools(settings)] == ["fetch_page", "wikipedia"]


def test_example_config_documents_tools_but_enables_none() -> None:
    example = Path(__file__).resolve().parent.parent / "lucy.example.toml"
    settings = load_settings(example)
    assert settings.tools.enabled == []
    assert settings.tools.web_search.searxng_url == "http://127.0.0.1:8890"
    text = example.read_text()
    assert '# enabled = ["wikipedia", "web_search", "fetch_page", "time", "weather"]' in text
    assert settings.tools.weather.units == "metric"
