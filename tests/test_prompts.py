import pytest

from lucy.prompts import load_prompt


def test_persona_prompt_loads() -> None:
    persona = load_prompt("persona")
    assert persona.startswith("You are Lucy, a personal assistant")
    assert "no markdown, no lists, no emoji" in persona
    # The chat model does not store memories, so it must not claim to (live probe:
    # "I have noted that" in 16 of 50 replies before this rule, 0 of 50 after).
    assert "never say that you have noted, saved, added or updated something" in persona


def test_profile_and_summary_prompts_forbid_guessing_gender() -> None:
    for name in ("profile", "entity_summary"):
        assert "never guess anyone's gender" in load_prompt(name).lower(), name
    # The profile's own example must not model a guessed pronoun.
    example = load_prompt("profile").split("Example output:")[1]
    assert not {"he", "his", "him", "she", "her"} & set(example.lower().replace(".", " ").split())


def test_unknown_prompt_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("no_such_prompt")
