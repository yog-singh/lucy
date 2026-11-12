import pytest

from lucy.prompts import load_prompt


def test_persona_prompt_loads() -> None:
    persona = load_prompt("persona")
    assert persona.startswith("You are Lucy, a personal assistant")
    assert "no markdown, no lists, no emoji" in persona


def test_unknown_prompt_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("no_such_prompt")
