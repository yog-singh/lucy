import sys
from pathlib import Path

import pytest

from lucy.core import registry
from lucy.core.registry import Registry, RegistryError


class Dummy:
    def __init__(self, model: str = "default") -> None:
        self.model = model


def test_register_and_create() -> None:
    reg = Registry()
    reg.register("llm", "dummy")(Dummy)
    instance = reg.create("llm", "dummy", model="m1")
    assert isinstance(instance, Dummy) and instance.model == "m1"
    assert reg.available("llm") == ["dummy"]
    assert reg.available("tts") == []


def test_decorator_returns_class_unchanged() -> None:
    reg = Registry()

    @reg.register("tts", "dummy")
    class Tts:
        pass

    assert reg.get("tts", "dummy") is Tts


def test_registering_same_class_again_is_a_no_op() -> None:
    reg = Registry()
    assert reg.register("stt", "dummy")(Dummy) is Dummy
    assert reg.register("stt", "dummy")(Dummy) is Dummy
    assert reg.get("stt", "dummy") is Dummy
    assert reg.available("stt") == ["dummy"]


def test_different_class_under_existing_name_raises() -> None:
    reg = Registry()
    reg.register("stt", "dummy")(Dummy)

    class Lookalike(Dummy):
        pass

    with pytest.raises(RegistryError, match="already registered"):
        reg.register("stt", "dummy")(Lookalike)
    assert reg.get("stt", "dummy") is Dummy


def test_same_name_on_different_ports_is_fine() -> None:
    reg = Registry()
    reg.register("llm", "ollama")(Dummy)
    reg.register("embedder", "ollama")(Dummy)
    assert reg.available("embedder") == ["ollama"]


def test_unknown_port_raises() -> None:
    reg = Registry()
    with pytest.raises(RegistryError, match="unknown port"):
        reg.register("nope", "x")
    with pytest.raises(RegistryError, match="unknown port"):
        reg.available("nope")
    with pytest.raises(RegistryError, match="unknown port"):
        reg.create("nope", "x")


def test_unknown_name_raises() -> None:
    reg = Registry()
    reg.register("memory", "a")(Dummy)
    with pytest.raises(RegistryError, match="no memory adapter named 'b'; registered: a"):
        reg.create("memory", "b")


def test_load_adapters_runs_decorators(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = tmp_path / "lucy_test_adapter_mod.py"
    module.write_text(
        "from lucy.core.registry import register\n"
        "@register('tool', 'lucy_test_tool')\n"
        "class T:\n"
        "    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        registry.load_adapters(["lucy_test_adapter_mod"])
        assert "lucy_test_tool" in registry.available("tool")
        assert registry.create("tool", "lucy_test_tool").__class__.__name__ == "T"
    finally:
        registry._default._classes["tool"].pop("lucy_test_tool", None)
        sys.modules.pop("lucy_test_adapter_mod", None)


def test_load_adapters_missing_module_raises() -> None:
    with pytest.raises(RegistryError, match="cannot import"):
        registry.load_adapters(["lucy_no_such_module_xyz"])
