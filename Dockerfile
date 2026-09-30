# Two images from one file, and the split is the point (ADR-0003, ADR-0006):
#
#   gate  --extra dev only. No torch, no model weights, no database driver,
#         no network needed at run time. `docker compose run --rm gate`.
#   app   --extra api --extra models. The FastAPI service over Postgres.
#
# The gate image CANNOT download a model: the libraries that would do it are
# not installed. That makes "the gate needs no model" structural, not a promise.

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./


FROM base AS gate
# Dependencies first, in their own layer, so a source edit does not reinstall them.
RUN uv sync --frozen --extra dev --no-install-project
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY tests ./tests
COPY scripts ./scripts
RUN uv sync --frozen --extra dev
# results/ is bind-mounted (see docker-compose.yml): the baseline lives in the
# working tree, and a gate run must read the committed one, not a copy baked
# in at build time.
CMD ["sh", "-c", "uv run gt corpus verify && uv run gt cache verify && uv run pytest -m gate -q"]


FROM base AS app
RUN uv sync --frozen --extra api --extra models --no-install-project
COPY src ./src
COPY configs ./configs
COPY data ./data
RUN uv sync --frozen --extra api --extra models
EXPOSE 8000
# `gt db load` is idempotent: it converges the database to the committed
# vectors on every start, so there is no separate migration step to forget.
CMD ["sh", "-c", "uv run gt db load && uv run uvicorn groundtruth.service.app:create_app --factory --host 0.0.0.0 --port 8000"]
