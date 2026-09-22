"""The one HTTP call every provider makes.

Shared so that a transport failure, a non-2xx status and an unparseable body
produce the same three error messages whichever vendor is in use -- each one
naming the provider, because "401" on its own does not say which key to fix.
"""

from __future__ import annotations

from typing import Any

import httpx

from groundtruth.llm.models import DEFAULT_TIMEOUT_SECONDS, ProviderRequestError


def post_json(
    client: httpx.Client | None,
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    params: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """POST JSON and return the decoded object, or raise ProviderRequestError.

    A ``client`` is passed in by tests (and reused across a run); when it is
    ``None`` a short-lived one is created for the single call.
    """
    owned = client is None
    active = client or httpx.Client(timeout=timeout)
    try:
        response = active.post(url, headers=headers, json=payload, params=params)
    except httpx.HTTPError as exc:
        raise ProviderRequestError(f"{provider}: request failed: {exc}") from exc
    finally:
        if owned:
            active.close()

    if response.status_code >= 400:
        # The body is where vendors put the actionable part -- a quota name, a
        # deployment that does not exist, a model that was retired.
        raise ProviderRequestError(
            f"{provider}: HTTP {response.status_code}: {response.text[:400]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderRequestError(
            f"{provider}: response was not JSON: {response.text[:200]!r}"
        ) from exc

    if not isinstance(body, dict):
        raise ProviderRequestError(f"{provider}: expected a JSON object, got {type(body).__name__}")
    return body
