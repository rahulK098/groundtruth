"""Two ways to ask the same question (ADR-0008).

``InProcessClient`` calls the ``Retriever`` directly -- the gate's path: no
server, no serialization, no infrastructure. ``HttpClient`` asks a running
service over HTTP -- the black-box path. A parity test asserts both return
identical rankings, which is what licenses the gate measuring the in-process
path while claiming to measure the service.

Only this module and the rest of ``groundtruth.service`` may import anything
HTTP-shaped; ``httpx`` here is a core dependency, not FastAPI.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import httpx

from groundtruth.retrieval.models import RetrievalResult, RetrievedPassage
from groundtruth.retrieval.pipeline import Retriever


class SearchClient(Protocol):
    def search(self, query: str, config_name: str, top_k: int | None = None) -> RetrievalResult: ...


class InProcessClient:
    """The gate's client: the Retriever, called directly."""

    def __init__(self, retrievers: Mapping[str, Retriever]) -> None:
        self._retrievers = retrievers

    def search(self, query: str, config_name: str, top_k: int | None = None) -> RetrievalResult:
        return self._retrievers[config_name].retrieve(query, top_k=top_k)


class HttpClient:
    """The black-box client: a running service, over HTTP."""

    def __init__(
        self, base_url: str = "http://127.0.0.1:8000", *, client: httpx.Client | None = None
    ) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=300.0)

    def search(self, query: str, config_name: str, top_k: int | None = None) -> RetrievalResult:
        payload: dict[str, object] = {"query": query, "config_name": config_name}
        if top_k is not None:
            payload["top_k"] = top_k
        response = self._client.post("/search", json=payload)
        response.raise_for_status()
        body = response.json()
        return RetrievalResult(
            query=query,
            config_name=body["config_name"],
            config_hash=body["config_hash"],
            passages=tuple(RetrievedPassage(**p) for p in body["passages"]),
            latency_ms=body["latency_ms"],
        )
