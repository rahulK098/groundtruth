"""The HTTP surface, against a state built over the mini corpus.

The real Retriever, real chunker, real indexes; only the encoder is the
deterministic hashing stand-in. No model, no database -- so every contract in
api.md is checked on the fast path.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

# The gate image installs only --extra dev (no FastAPI), and `pytest -m gate`
# still COLLECTS every module before filtering by marker. Skipping here keeps
# the gate runnable where the service's dependencies deliberately are not.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from groundtruth.retrieval.build import build_retriever
from groundtruth.service.app import REQUEST_ID_HEADER, create_app
from groundtruth.service.clients import HttpClient, InProcessClient
from groundtruth.service.state import ServiceState, served_names, version_info
from tests.fixtures.mini_corpus import HashingEmbedder, mini_config, mini_documents

QUERY = "what is the standard for granting summary judgment"


def make_state(*, serve: tuple[str, ...] = ("mini_dense", "mini_hybrid")) -> ServiceState:
    configs = {
        "mini_dense": mini_config(
            name="mini_dense", retrieval_mode="dense", top_k=5, dense_top_n=20
        ),
        "mini_hybrid": mini_config(
            name="mini_hybrid", retrieval_mode="hybrid", top_k=5, dense_top_n=20
        ),
        "mini_unserved": mini_config(
            name="mini_unserved", retrieval_mode="dense", top_k=3, dense_top_n=10
        ),
    }
    documents = mini_documents()
    retrievers = {
        name: build_retriever(configs[name], documents, HashingEmbedder()) for name in serve
    }
    return ServiceState(
        configs=configs,
        retrievers=retrievers,
        version=version_info(configs, corpus_manifest_hash="sha256:c", golden_set_hash="sha256:g"),
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(make_state())) as test_client:
        yield test_client


class TestHealth:
    def test_healthz_is_always_ok(self, client: TestClient):
        assert client.get("/healthz").json() == {"status": "ok"}

    def test_readyz_lists_the_served_configs(self, client: TestClient):
        body = client.get("/readyz").json()
        assert body == {"status": "ready", "served": ["mini_dense", "mini_hybrid"]}

    def test_readyz_is_503_problem_json_when_the_state_failed_to_build(self):
        def broken() -> ServiceState:
            raise RuntimeError("postgres is down")

        with TestClient(create_app(state_factory=broken)) as test_client:
            response = test_client.get("/readyz")
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/problem+json")
        assert "postgres is down" in response.json()["detail"]

    def test_data_endpoints_are_503_not_4xx_while_not_ready(self):
        # A model still loading is not the client's fault (api.md).
        with TestClient(create_app(make_state(serve=()))) as test_client:
            assert (
                test_client.post(
                    "/search", json={"query": QUERY, "config_name": "mini_dense"}
                ).status_code
                == 503
            )


class TestSearch:
    def test_returns_ranked_passages_with_per_stage_scores(self, client: TestClient):
        body = client.post("/search", json={"query": QUERY, "config_name": "mini_hybrid"}).json()
        assert body["config_name"] == "mini_hybrid"
        assert [p["rank"] for p in body["passages"]] == list(range(1, len(body["passages"]) + 1))
        first = body["passages"][0]
        assert first["doc_id"] == "mini-001"
        # Per stage, never blended (api.md).
        assert set(first["scores"]) == {"dense", "lexical", "fused", "rerank"}
        assert set(body["latency_ms"]) == {"embed", "dense", "lexical", "fuse", "rerank", "total"}

    def test_top_k_truncates(self, client: TestClient):
        body = client.post(
            "/search", json={"query": QUERY, "config_name": "mini_dense", "top_k": 2}
        ).json()
        assert len(body["passages"]) == 2

    def test_top_k_beyond_the_candidate_pool_is_422(self, client: TestClient):
        response = client.post(
            "/search", json={"query": QUERY, "config_name": "mini_dense", "top_k": 999}
        )
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/problem+json")

    def test_an_unknown_config_is_404(self, client: TestClient):
        assert (
            client.post("/search", json={"query": QUERY, "config_name": "nope"}).status_code == 404
        )

    def test_a_shipped_but_unserved_config_is_409_not_404(self, client: TestClient):
        # It exists; it just isn't loaded here. The distinction tells the
        # caller the fix (GT_SERVE_CONFIGS) rather than implying a typo.
        response = client.post("/search", json={"query": QUERY, "config_name": "mini_unserved"})
        assert response.status_code == 409
        assert "GT_SERVE_CONFIGS" in response.json()["detail"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": "", "config_name": "mini_dense"},
            {"query": "   ", "config_name": "mini_dense"},
            {"query": QUERY},
            {"query": QUERY, "config_name": "mini_dense", "top_k": 0},
            {"query": QUERY, "config_name": "mini_dense", "unexpected": 1},
        ],
    )
    def test_invalid_input_is_422_problem_json(self, client: TestClient, payload: dict):
        response = client.post("/search", json=payload)
        assert response.status_code == 422
        assert response.json()["status"] == 422

    def test_http_and_in_process_return_identical_rankings(self, client: TestClient):
        # ADR-0008's parity claim, on the fast path: serialization over HTTP
        # must not move a single rank.
        state = make_state()
        over_http = HttpClient(client=client).search(QUERY, "mini_hybrid")
        in_process = InProcessClient(state.retrievers).search(QUERY, "mini_hybrid")
        assert [p.chunk_id for p in over_http.passages] == [p.chunk_id for p in in_process.passages]
        assert [p.scores for p in over_http.passages] == [p.scores for p in in_process.passages]


class TestMetadata:
    def test_configs_marks_what_is_served(self, client: TestClient):
        rows = {row["name"]: row for row in client.get("/configs").json()}
        assert rows["mini_dense"]["served"] is True
        assert rows["mini_unserved"]["served"] is False

    def test_one_config_includes_its_hash(self, client: TestClient):
        body = client.get("/configs/mini_dense").json()
        assert body["name"] == "mini_dense"
        assert body["config_hash"]

    def test_an_unknown_config_detail_is_404(self, client: TestClient):
        assert client.get("/configs/nope").status_code == 404

    def test_version_reports_the_run_identity_components(self, client: TestClient):
        body = client.get("/version").json()
        assert body["corpus_manifest_hash"] == "sha256:c"
        assert body["golden_set_hash"] == "sha256:g"
        assert body["embedding_model"]["id"]
        assert body["code_version"]


class TestRequestId:
    def test_is_echoed_when_supplied(self, client: TestClient):
        response = client.get("/healthz", headers={REQUEST_ID_HEADER: "abc123"})
        assert response.headers[REQUEST_ID_HEADER] == "abc123"

    def test_is_generated_otherwise_and_appears_in_problems(self, client: TestClient):
        response = client.get("/configs/nope")
        assert response.headers[REQUEST_ID_HEADER]
        assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


class TestServedNames:
    def test_empty_means_the_default_set(self):
        assert served_names("", {"dense_512": 1, "dense_256": 1, "hybrid_512": 1}) == (
            "dense_512",
            "dense_256",
            "hybrid_512",
        )

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(ValueError, match="nope"):
            served_names("dense_512,nope", {"dense_512": 1})
