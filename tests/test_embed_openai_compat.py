"""Tests for the OpenAI-compatible embedder adapter.

Response bodies mirror what Ollama 0.40.16 sends from `/v1/embeddings`.
"""

import json
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from lucy.adapters.embed_openai_compat import (
    OLLAMA_BASE_URL,
    EmbedderError,
    OllamaEmbedder,
    OpenAICompatEmbedder,
)
from lucy.core import registry
from lucy.core.ports import Embedder

Handler = Callable[[httpx.Request], httpx.Response]


def vector_for(text: str, dims: int) -> list[float]:
    """A distinct vector per text, so tests can check which text produced which vector."""
    seed = sum(map(ord, text)) + len(text)
    return [float(seed + i) for i in range(dims)]


def ollama_reply(request: httpx.Request, dims: int = 4, reverse: bool = False) -> httpx.Response:
    body = json.loads(request.content)
    data = [
        {"object": "embedding", "embedding": vector_for(text, dims), "index": i}
        for i, text in enumerate(body["input"])
    ]
    if reverse:
        data.reverse()
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": data,
            "model": body["model"],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        },
    )


def make_embedder(
    handler: Handler | None = None,
    cls: type[OpenAICompatEmbedder] = OllamaEmbedder,
    **options: Any,
) -> tuple[OpenAICompatEmbedder, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return (handler or ollama_reply)(request)

    options.setdefault("model", "embeddinggemma")
    options.setdefault("dimensions", 4)
    return cls(transport=httpx.MockTransport(record), **options), requests


def inputs(requests: list[httpx.Request]) -> list[list[str]]:
    return [json.loads(r.content)["input"] for r in requests]


# --- Registry and protocol ----------------------------------------------------


def test_classes_satisfy_protocol_and_are_registered() -> None:
    assert isinstance(OllamaEmbedder(model="m", dimensions=8), Embedder)
    assert isinstance(
        OpenAICompatEmbedder(model="m", base_url="http://x/v1", dimensions=8), Embedder
    )
    assert registry.get("embedder", "ollama") is OllamaEmbedder
    assert registry.get("embedder", "openai_compat") is OpenAICompatEmbedder


def test_builtin_modules_include_the_embedder() -> None:
    from lucy.app import BUILTIN_ADAPTER_MODULES

    assert "lucy.adapters.embed_openai_compat" in BUILTIN_ADAPTER_MODULES
    registry.load_adapters(["lucy.adapters.embed_openai_compat"] * 2)  # re-import is harmless
    assert registry.get("embedder", "ollama") is OllamaEmbedder


def test_constructor_options() -> None:
    embedder = registry.create("embedder", "ollama", model="embeddinggemma", dimensions=768)
    assert embedder.base_url == OLLAMA_BASE_URL
    assert embedder.capabilities.dimensions == 768
    assert embedder.capabilities.model == "embeddinggemma"
    with pytest.raises(TypeError):
        registry.create("embedder", "openai_compat", model="m", dimensions=8)  # no base_url
    with pytest.raises(TypeError):
        registry.create("embedder", "ollama", model="m")  # dimensions is required
    with pytest.raises(TypeError):
        registry.create("embedder", "ollama", model="m", dimensions=8, bogus=1)
    with pytest.raises(ValueError, match="batch_size"):
        OllamaEmbedder(model="m", dimensions=8, batch_size=0)


# --- Requests and results -----------------------------------------------------


async def test_request_shape() -> None:
    embedder, requests = make_embedder(
        cls=OpenAICompatEmbedder, base_url="http://studio:1234/v1/", api_key="secret"
    )
    vectors = await embedder.embed(["hello", "world"])
    assert vectors == [vector_for("hello", 4), vector_for("world", 4)]
    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == "http://studio:1234/v1/embeddings"
    assert request.headers["authorization"] == "Bearer secret"
    assert json.loads(request.content) == {"model": "embeddinggemma", "input": ["hello", "world"]}


async def test_no_authorization_header_without_api_key() -> None:
    embedder, requests = make_embedder()
    await embedder.embed(["x"])
    assert "authorization" not in requests[0].headers


async def test_empty_input_makes_no_request() -> None:
    embedder, requests = make_embedder()
    assert await embedder.embed([]) == []
    assert await embedder.embed([], kind="query") == []
    assert requests == []


async def test_batching_preserves_order() -> None:
    texts = [f"text {i}" for i in range(7)]
    embedder, requests = make_embedder(batch_size=3)
    vectors = await embedder.embed(texts)
    assert inputs(requests) == [texts[0:3], texts[3:6], texts[6:7]]
    assert vectors == [vector_for(t, 4) for t in texts]


async def test_results_are_ordered_by_index() -> None:
    texts = ["a", "bb", "ccc"]
    embedder, _ = make_embedder(lambda r: ollama_reply(r, reverse=True), batch_size=2)
    assert await embedder.embed(texts) == [vector_for(t, 4) for t in texts]


async def test_prefixes_by_kind() -> None:
    embedder, requests = make_embedder(query_prefix="query: ", document_prefix="text: ")
    documents = await embedder.embed(["Priya lives in Pune."])
    queries = await embedder.embed(["Where does Priya live?"], kind="query")
    assert inputs(requests) == [["text: Priya lives in Pune."], ["query: Where does Priya live?"]]
    assert documents == [vector_for("text: Priya lives in Pune.", 4)]
    assert queries == [vector_for("query: Where does Priya live?", 4)]


async def test_no_prefixes_by_default() -> None:
    embedder, requests = make_embedder()
    await embedder.embed(["a"], kind="query")
    await embedder.embed(["a"], kind="document")
    assert inputs(requests) == [["a"], ["a"]]


async def test_invalid_kind_is_rejected() -> None:
    embedder, requests = make_embedder()
    with pytest.raises(ValueError, match="kind"):
        await embedder.embed(["a"], kind="passage")  # type: ignore[arg-type]
    assert requests == []


# --- Errors -------------------------------------------------------------------


async def test_dimension_mismatch_is_actionable() -> None:
    embedder, _ = make_embedder(lambda r: ollama_reply(r, dims=768), dimensions=512)
    with pytest.raises(EmbedderError, match=r"768-dimensional .* dimensions = 512"):
        await embedder.embed(["x"])


async def test_http_error_includes_server_message() -> None:
    # Captured from Ollama for an unknown model.
    body = {
        "error": {
            "message": "model 'nope' not found",
            "type": "not_found_error",
            "param": None,
            "code": None,
        }
    }
    embedder, _ = make_embedder(lambda r: httpx.Response(404, json=body), model="nope")
    with pytest.raises(EmbedderError, match=r"Ollama .* HTTP 404 .*model 'nope' not found"):
        await embedder.embed(["x"])


async def test_http_error_with_plain_text_body() -> None:
    embedder, _ = make_embedder(
        lambda r: httpx.Response(500, text="boom"), OpenAICompatEmbedder, base_url="http://h/v1"
    )
    with pytest.raises(EmbedderError, match="HTTP 500.*boom"):
        await embedder.embed(["x"])


async def test_connection_error_is_actionable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    embedder, _ = make_embedder(refuse)
    with pytest.raises(EmbedderError, match=r"cannot reach Ollama at http://localhost:11434/v1"):
        await embedder.embed(["x"])


async def test_real_connection_refused() -> None:
    embedder = OllamaEmbedder(model="m", dimensions=4, base_url="http://127.0.0.1:9/v1")
    with pytest.raises(EmbedderError, match="cannot reach Ollama"):
        await embedder.embed(["x"])
    await embedder.aclose()


async def test_timeout_is_reported() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    embedder, _ = make_embedder(slow, timeout=5)
    with pytest.raises(EmbedderError, match="did not respond within 5s"):
        await embedder.embed(["x"])


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(200, text="not json"), "not JSON"),
        (httpx.Response(200, json={"data": []}), "returned 0 embeddings for 1 inputs"),
        (httpx.Response(200, json={"object": "list"}), "returned no embeddings"),
        (httpx.Response(200, json={"data": [{"index": 0}]}), "malformed embedding"),
        (httpx.Response(200, json={"data": [{"index": 5, "embedding": [1.0] * 4}]}), "index"),
        (httpx.Response(200, json={"data": [{"index": 0, "embedding": ["a"] * 4}]}), "non-numeric"),
    ],
)
async def test_malformed_responses_raise(response: httpx.Response, match: str) -> None:
    embedder, _ = make_embedder(lambda r: response)
    with pytest.raises(EmbedderError, match=match):
        await embedder.embed(["x"])


async def test_duplicate_index_raises() -> None:
    item = {"index": 0, "embedding": [1.0] * 4}
    embedder, _ = make_embedder(lambda r: httpx.Response(200, json={"data": [item, item]}))
    with pytest.raises(EmbedderError, match="index"):
        await embedder.embed(["a", "b"])


async def test_aclose() -> None:
    embedder, _ = make_embedder()
    await embedder.aclose()
    with pytest.raises(EmbedderError, match="closed"):
        await embedder.embed(["x"])


# --- Live ---------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


@pytest.mark.live
async def test_live_ollama_embeds() -> None:
    """Embeds with the `[embedder]` section of lucy.example.toml."""
    from pathlib import Path

    from lucy.core.config import load_settings

    example = Path(__file__).resolve().parent.parent / "lucy.example.toml"
    config = load_settings(example).embedder
    assert config is not None and config.adapter == "ollama"
    options = {"timeout": 300, **config.options}
    embedder = OllamaEmbedder(**options)
    dims = embedder.capabilities.dimensions
    try:
        documents = await embedder.embed(
            ["The user's sister Priya lives in Pune.", "The user is allergic to peanuts."]
        )
        (query,) = await embedder.embed(["Where does my sister live?"], kind="query")
    finally:
        await embedder.aclose()
    assert [len(v) for v in documents] == [dims, dims]
    assert cosine(query, documents[0]) > cosine(query, documents[1])
