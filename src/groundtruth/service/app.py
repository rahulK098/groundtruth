"""``create_app()`` -- the FastAPI adapter (ADR-0008).

Routes validate, dispatch the shared ``Retriever`` off the event loop, and
serialize. Nothing here ranks, scores or fuses anything.

Run:  uv run uvicorn groundtruth.service.app:create_app --factory
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Final

import anyio
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from groundtruth.retrieval.pipeline import Retriever
from groundtruth.service.schemas import (
    ConfigSummary,
    SearchRequest,
    SearchResponse,
    VersionInfo,
)
from groundtruth.service.state import ServiceState

logger = logging.getLogger("groundtruth.service")

#: Retrieval is CPU-bound (query embedding) and holds a DB connection per
#: index. Two concurrent searches is plenty for a demo surface, and bounding
#: it keeps a burst from queueing unbounded work behind the model.
SEARCH_CONCURRENCY: Final[int] = 2

REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

router = APIRouter()


class ServiceError(Exception):
    """An error the service renders as RFC 9457 problem+json."""

    def __init__(self, status: int, title: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.title = title
        self.detail = detail


# --- helpers ------------------------------------------------------------------


def _problem(request: Request, status: int, title: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": "about:blank",
            "title": title,
            "status": status,
            "detail": detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


def _state(request: Request) -> ServiceState:
    state: ServiceState | None = request.app.state.gt_state
    if state is None or not state.ready:
        # A model still loading, or a database not yet loaded, is not the
        # client's fault: 503, never 4xx (api.md).
        raise ServiceError(503, "Service not ready", request.app.state.gt_not_ready_reason)
    return state


def _served_retriever(served: ServiceState, body: SearchRequest) -> Retriever:
    """The retriever for a request, or the precise reason there is none."""
    retriever = served.retrievers.get(body.config_name)
    if retriever is None:
        if body.config_name in served.configs:
            # It exists; it just is not loaded here. 409 names the fix rather
            # than implying a typo.
            raise ServiceError(
                409,
                "Config not served",
                f"{body.config_name!r} is a shipped config but is not served here; "
                f"set GT_SERVE_CONFIGS to include it",
            )
        raise ServiceError(404, "Unknown config", f"no configuration named {body.config_name!r}")

    if not body.query.strip():
        raise ServiceError(422, "Invalid request", "query is empty")
    pool = retriever.config.dense_top_n
    if body.top_k is not None and body.top_k > pool:
        raise ServiceError(
            422,
            "Invalid request",
            f"top_k {body.top_k} exceeds {body.config_name}'s candidate pool of {pool}",
        )
    return retriever


# --- routes -------------------------------------------------------------------


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up. Says nothing about models or data."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, object]:
    """Readiness: retrievers built, which means models loaded and DB reachable."""
    return {"status": "ready", "served": sorted(_state(request).retrievers)}


@router.get("/version", response_model=VersionInfo)
async def version(request: Request) -> VersionInfo:
    return _state(request).version


@router.get("/configs", response_model=list[ConfigSummary])
async def configs(request: Request) -> list[ConfigSummary]:
    served = _state(request)
    return [
        ConfigSummary(
            name=name,
            config_hash=config.config_hash,
            retrieval_mode=config.retrieval_mode,
            reranker=config.reranker.enabled,
            served=name in served.retrievers,
        )
        for name, config in sorted(served.configs.items())
    ]


@router.get("/configs/{name}")
async def config(name: str, request: Request) -> dict[str, object]:
    served = _state(request)
    if name not in served.configs:
        raise ServiceError(404, "Unknown config", f"no configuration named {name!r}")
    found = served.configs[name]
    return {"config_hash": found.config_hash, **found.model_dump(mode="json")}


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    retriever = _served_retriever(_state(request), body)
    # CPU-bound embedding plus blocking DB reads: never on the event loop.
    result = await anyio.to_thread.run_sync(
        lambda: retriever.retrieve(body.query, top_k=body.top_k),
        limiter=request.app.state.limiter,
    )
    return SearchResponse.from_result(result)


# --- wiring -------------------------------------------------------------------


async def _request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request.state.request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request.state.request_id
    return response


async def _service_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ServiceError)
    return _problem(request, exc.status, exc.title, exc.detail)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return _problem(request, 422, "Invalid request", str(exc.errors()))


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    return _problem(request, exc.status_code, "HTTP error", str(exc.detail))


def _lifespan(
    state: ServiceState | None, state_factory: Callable[[], ServiceState] | None
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.gt_state = state
        app.state.gt_not_ready_reason = "the service state has not been built"
        app.state.limiter = anyio.CapacityLimiter(SEARCH_CONCURRENCY)
        if state is None:
            factory = state_factory
            if factory is None:
                from groundtruth.service.state import build_state

                factory = build_state
            try:
                # Loading models and opening Postgres block; off the loop.
                app.state.gt_state = await anyio.to_thread.run_sync(factory)
            except Exception as exc:
                logger.exception("service state failed to build")
                app.state.gt_not_ready_reason = f"{type(exc).__name__}: {exc}"
        yield

    return lifespan


def create_app(
    state: ServiceState | None = None,
    *,
    state_factory: Callable[[], ServiceState] | None = None,
) -> FastAPI:
    """Build the app. Tests pass ``state``; production builds it at startup."""
    app = FastAPI(
        title="Groundtruth",
        summary="Retrieval over a legal corpus -- the same Retriever the regression gate measures.",
        lifespan=_lifespan(state, state_factory),
    )
    app.middleware("http")(_request_id)
    app.add_exception_handler(ServiceError, _service_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.include_router(router)
    return app
