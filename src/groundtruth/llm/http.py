"""The one HTTP call every provider makes.

Shared so that a transport failure, a non-2xx status and an unparseable body
produce the same three error messages whichever vendor is in use -- each one
naming the provider, because "401" on its own does not say which key to fix.

Also shared: retry with backoff on a **transient** failure (429, or a 5xx that
means the vendor is temporarily overloaded, not that the request is wrong).
Groq's free tier rate-limits per organization, and a 120-passage generation
run can trip it mid-run; a 429 body typically names how long to wait, and that
is honoured via ``Retry-After`` before falling back to exponential backoff.

This is retry, not fallback. It tries the SAME provider and the SAME model
again -- unlike the provider chain the registry deliberately refuses to have
(see registry.py), which would let a run be answered by a different vendor
with nothing in the record marking that as unintended. A quota that never
recovers still surfaces as a failure once retries are exhausted.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Final

import httpx

from groundtruth.llm.models import DEFAULT_TIMEOUT_SECONDS, ProviderRequestError

#: 429 (rate limit) and the 5xx codes that mean "temporarily overloaded, ask
#: again" rather than "the request itself is wrong". A 4xx other than 429 is
#: never retried -- retrying a bad deployment name or an expired key wastes
#: the whole backoff window finding out what one call already established.
RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

DEFAULT_MAX_RETRIES: Final[int] = 5

#: Exponential backoff when the vendor does not say how long to wait:
#: 1s, 2s, 4s, 8s, 16s, capped at 30s. Used only as a fallback -- a vendor's
#: own Retry-After is authoritative when present (also capped, so a vendor
#: response does not stall a run for an unbounded time).
_BACKOFF_BASE_SECONDS: Final[float] = 1.0
_BACKOFF_CAP_SECONDS: Final[float] = 30.0


def _retry_delay(response: httpx.Response, retry_index: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            # The delta-seconds form (RFC 9110 10.2.3). The HTTP-date form
            # also appears in the wild; it fails this parse and falls
            # through to backoff rather than being decoded, since a wait
            # computed from the wrong clock is worse than a reasonable guess.
            return min(max(float(retry_after), 0.0), _BACKOFF_CAP_SECONDS)
        except ValueError:
            pass
    return min(_BACKOFF_BASE_SECONDS * (2.0**retry_index), _BACKOFF_CAP_SECONDS)


def post_json(
    client: httpx.Client | None,
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    params: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """POST JSON and return the decoded object, or raise ProviderRequestError.

    A ``client`` is passed in by tests (and reused across a run); when it is
    ``None`` a short-lived one is created and closed after every attempt.

    A transport-level failure (DNS, connection refused, timeout) is never
    retried here -- distinguishing "the network is down" from "the vendor
    is asking me to slow down" needs a response to look at, and only the
    latter is what a batch generation run can recover from by waiting.

    ``sleep`` defaults to ``None`` rather than to ``time.sleep`` directly, and
    is resolved inside the function body instead. A default argument is bound
    once, at import time -- a test patching ``time.sleep`` afterwards would
    silently miss an already-captured reference and sleep for real.
    """
    owned = client is None
    active = client or httpx.Client(timeout=timeout)
    try:
        return _post_with_retries(
            active,
            provider=provider,
            url=url,
            headers=headers,
            payload=payload,
            params=params,
            max_retries=max_retries,
            sleep=sleep or time.sleep,
        )
    finally:
        if owned:
            active.close()


def _post_with_retries(
    client: httpx.Client,
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    params: dict[str, str] | None,
    max_retries: int,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.post(url, headers=headers, json=payload, params=params)
        except httpx.HTTPError as exc:
            raise ProviderRequestError(f"{provider}: request failed: {exc}") from exc

        if response.status_code in RETRYABLE_STATUSES and attempt <= max_retries:
            sleep(_retry_delay(response, retry_index=attempt - 1))
            continue

        if response.status_code >= 400:
            retried = f" after {attempt} attempts" if attempt > 1 else ""
            # The body is where vendors put the actionable part -- a quota
            # name, a deployment that does not exist, a model that was
            # retired.
            raise ProviderRequestError(
                f"{provider}: HTTP {response.status_code}{retried}: {response.text[:400]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderRequestError(
                f"{provider}: response was not JSON: {response.text[:200]!r}"
            ) from exc

        if not isinstance(body, dict):
            raise ProviderRequestError(
                f"{provider}: expected a JSON object, got {type(body).__name__}"
            )
        return body
