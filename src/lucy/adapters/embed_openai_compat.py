"""Embedder adapter for OpenAI-compatible `/embeddings` servers (Ollama, LM Studio, ...).

Request: `POST {base_url}/embeddings` with `{"model": ..., "input": [texts]}`.
Response (as sent by Ollama 0.40.16): `{"object": "list", "model": ...,
"data": [{"object": "embedding", "index": i, "embedding": [floats]}, ...],
"usage": {...}}`. Results are ordered by `index`, so order never depends on
the server.

Retrieval models often expect different prefixes for queries and documents
(e.g. EmbeddingGemma's "task: search result | query: "). Ollama applies none
itself (the model template is just the prompt), so the adapter prepends
`query_prefix` or `document_prefix` according to `kind` (ADR 0009).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final, Literal

import httpx

from lucy.core import registry
from lucy.core.types import EmbedderCaps

OLLAMA_BASE_URL: Final = "http://localhost:11434/v1"
"""Default endpoint of Ollama's OpenAI-compatible API."""


class EmbedderError(RuntimeError):
    """The embedding server could not be reached, refused the request or sent bad vectors."""


def _error_detail(response: httpx.Response) -> str:
    """The server's error message from a non-200 response body."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:500] or response.reason_phrase
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str):
        return error
    return json.dumps(body)[:500]


@registry.register("embedder", "openai_compat")
class OpenAICompatEmbedder:
    """Embeddings from an OpenAI-compatible `/embeddings` endpoint.

    `dimensions` is the vector length the model produces; every returned
    vector is checked against it. It is not sent to the server. Texts are sent
    in batches of `batch_size`. `transport` is for tests.
    """

    server_name: str = "the embedding server"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        dimensions: int,
        api_key: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        timeout: float = 60,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.capabilities = EmbedderCaps(dimensions=dimensions, model=model)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self._timeout = timeout
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _where(self) -> str:
        return f"{self.server_name} at {self.base_url}"

    async def embed(
        self, texts: Sequence[str], *, kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        """Return one vector per text, in input order; see the `Embedder` port."""
        if kind not in ("document", "query"):
            raise ValueError(f"kind must be 'document' or 'query', not {kind!r}")
        if not texts:
            return []
        if self._client.is_closed:
            raise EmbedderError(f"the embedder for {self.model!r} has been closed")
        prefix = self.query_prefix if kind == "query" else self.document_prefix
        inputs = [prefix + text for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            vectors += await self._embed_batch(inputs[start : start + self.batch_size])
        return vectors

    async def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/embeddings", json={"model": self.model, "input": inputs}
            )
        except httpx.ConnectError as exc:
            raise EmbedderError(f"cannot reach {self._where()}; is it running? ({exc})") from exc
        except httpx.TimeoutException as exc:
            raise EmbedderError(
                f"{self._where()} did not respond within {self._timeout:g}s "
                f"(model {self.model!r}); it may still be loading the model"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbedderError(f"connection to {self._where()} failed: {exc!r}") from exc
        if response.status_code != 200:
            raise EmbedderError(
                f"{self._where()} returned HTTP {response.status_code} for model "
                f"{self.model!r}: {_error_detail(response)}"
            )
        return self._parse(response, len(inputs))

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        try:
            body: Any = response.json()
        except ValueError as exc:
            raise EmbedderError(
                f"{self._where()} sent a response that is not JSON: {response.text[:200]!r}"
            ) from exc
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != expected:
            got = len(data) if isinstance(data, list) else "no"
            raise EmbedderError(
                f"{self._where()} returned {got} embeddings for {expected} inputs "
                f"(model {self.model!r})"
            )
        by_index: dict[int, list[float]] = {}
        for position, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise EmbedderError(
                    f"{self._where()} sent a malformed embedding item: {str(item)[:200]}"
                )
            index = item.get("index", position)
            if not isinstance(index, int) or not 0 <= index < expected or index in by_index:
                raise EmbedderError(f"{self._where()} sent an invalid embedding index: {index!r}")
            vector = item["embedding"]
            if len(vector) != self.capabilities.dimensions:
                raise EmbedderError(
                    f"model {self.model!r} returned {len(vector)}-dimensional vectors but the "
                    f"embedder is configured with dimensions = {self.capabilities.dimensions}; "
                    "set `dimensions` in [embedder] to the model's embedding length"
                )
            try:
                by_index[index] = [float(x) for x in vector]
            except (TypeError, ValueError) as exc:
                raise EmbedderError(f"{self._where()} sent a non-numeric embedding value") from exc
        return [by_index[i] for i in range(expected)]


@registry.register("embedder", "ollama")
class OllamaEmbedder(OpenAICompatEmbedder):
    """`OpenAICompatEmbedder` defaulting to a local Ollama server."""

    server_name = "Ollama"

    def __init__(self, *, base_url: str = OLLAMA_BASE_URL, **options: Any) -> None:
        super().__init__(base_url=base_url, **options)
