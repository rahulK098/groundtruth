"""CourtListener REST client.

HTTP machinery only. Everything decidable without a network -- which opinions
to ask for, how to map a record to a Document -- lives in ``selection.py`` and
is fully tested. This module is excluded from coverage because exercising it
means talking to a live API, and a mock-only test here would assert that the
mock behaves like the mock.

Run rarely and never in the gate: its output is the committed snapshot, which
is what makes reproduction offline and free.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Final

import httpx

from groundtruth.corpus.models import Document
from groundtruth.corpus.selection import (
    CorpusSelection,
    cluster_query,
    extract_body,
    is_eligible,
    opinion_query,
    to_document,
)

API_ROOT: Final[str] = "https://www.courtlistener.com/api/rest/v4"
USER_AGENT: Final[str] = "groundtruth-retrieval-eval/0.1 (research harness)"

_MAX_RETRIES: Final[int] = 4
_BACKOFF_SECONDS: Final[float] = 2.0

#: Beyond this, fail with guidance rather than sleep. A silent 35-minute sleep
#: is indistinguishable from a hung process.
_MAX_RETRY_AFTER_SECONDS: Final[float] = 120.0
_RETRYABLE: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

#: Called as ``(completed, total)`` so a long fetch is not a silent stall.
ProgressCallback = Callable[[int, int], None]


class CourtListenerError(Exception):
    """The CourtListener API could not be queried."""


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Token {token}", "User-Agent": USER_AGENT}


def _get(client: httpx.Client, url: str, params: Mapping[str, str] | None = None) -> dict[str, Any]:
    """GET with backoff on rate limiting and transient server errors."""
    last: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.get(url, params=params)
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
            continue

        if response.status_code == 200:
            payload: dict[str, Any] = response.json()
            return payload

        if response.status_code == 401:
            raise CourtListenerError(
                "CourtListener rejected the token (401). Check COURTLISTENER_API_TOKEN in .env."
            )

        if response.status_code in _RETRYABLE:
            # Honour Retry-After when supplied -- guessing shorter than the
            # server asked is how a client earns a longer ban -- but refuse to
            # sleep for an unbounded period. CourtListener's free tier allows
            # 50 requests/hour and answers an exhausted quota with a
            # Retry-After of up to an hour. Sleeping on that looks exactly like
            # a hang: no output, no error, for 35 minutes.
            delay = float(response.headers.get("Retry-After", _BACKOFF_SECONDS * (attempt + 1)))
            if delay > _MAX_RETRY_AFTER_SECONDS:
                raise CourtListenerError(
                    f"CourtListener rate limit exhausted. It will not accept "
                    f"another request for {delay:.0f}s (~{delay / 60:.0f} min).\n\n"
                    f"The free tier allows 50 requests/hour. A full fetch needs "
                    f"only about 5, so this usually means an earlier run burned "
                    f"the quota. Wait for the window to reset, or request a "
                    f"higher limit from CourtListener.\n\n"
                    f"Nothing was written; re-run when the quota resets."
                )
            time.sleep(delay)
            continue

        raise CourtListenerError(
            f"GET {url} failed: HTTP {response.status_code} {response.text[:200]}"
        )

    raise CourtListenerError(f"GET {url} failed after {_MAX_RETRIES} attempts: {last}")


def _paginate(
    client: httpx.Client, url: str, params: Mapping[str, str] | None
) -> Iterator[Mapping[str, Any]]:
    page = _get(client, url, params)
    while True:
        yield from page.get("results", [])
        next_url = page.get("next")
        if not next_url:
            return
        page = _get(client, next_url)


def _fetch_cluster_metadata(
    client: httpx.Client,
    cluster_ids: set[int],
    selection: CorpusSelection,
) -> dict[int, Mapping[str, Any]]:
    """Collect case names and dates from the paginated cluster LIST endpoint.

    Deliberately not one request per cluster. The API rejects ``id__in``, so
    per-cluster fetching would cost one request per document -- 150 requests
    against a 50/hour free-tier limit, which takes hours and looks like a hang.
    Filtering the list by the same court and date window costs ~2 requests.

    Best effort by design: a metadata failure yields an empty title rather than
    discarding opinion text already in hand.
    """
    if not cluster_ids:
        return {}

    wanted = set(cluster_ids)
    highest = max(wanted)
    found: dict[int, Mapping[str, Any]] = {}

    try:
        for record in _paginate(client, f"{API_ROOT}/clusters/", cluster_query(selection)):
            cluster_id = int(record.get("id") or 0)
            if cluster_id in wanted:
                found[cluster_id] = record
                wanted.discard(cluster_id)
                if not wanted:
                    break
            # Results are ordered by id, so once past the highest id we need,
            # nothing further can match and paging on would burn quota.
            if cluster_id > highest:
                break
    except CourtListenerError:
        return found

    return found


def fetch_documents(
    selection: CorpusSelection,
    token: str,
    *,
    with_metadata: bool = True,
    on_progress: ProgressCallback | None = None,
) -> tuple[Document, ...]:
    """Fetch a deterministic slice of opinions as normalized Documents."""
    court = selection.courts[0] if len(selection.courts) == 1 else ",".join(selection.courts)
    kept: list[Mapping[str, Any]] = []

    with httpx.Client(headers=_headers(token), timeout=60.0, follow_redirects=True) as client:
        for record in _paginate(client, f"{API_ROOT}/opinions/", opinion_query(selection)):
            body = extract_body(record)
            if not is_eligible(record, selection, body):
                continue
            kept.append(record)
            if on_progress is not None and len(kept) % 25 == 0:
                on_progress(len(kept), selection.limit)
            if len(kept) >= selection.limit:
                break

        clusters: Mapping[int, Mapping[str, Any]] = {}
        if with_metadata and kept:
            ids = {int(r["cluster_id"]) for r in kept if r.get("cluster_id")}
            clusters = _fetch_cluster_metadata(client, ids, selection)

    documents = (to_document(record, court=court, clusters=clusters) for record in kept)
    return tuple(doc for doc in documents if doc is not None)
