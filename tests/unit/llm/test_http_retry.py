"""Retry-with-backoff for transient provider failures (429, 5xx).

Groq's free tier rate-limits per organization, and a 120-passage generation
run can trip it mid-run -- exactly what happened: a 429 whose body said
"try again in 9.4s". The fix lives in the ONE shared HTTP call every
provider makes (see http.py), so every vendor gets it for free rather than
each provider reimplementing its own retry loop.

This is scoped retry, not the provider fallback chain the registry
deliberately refuses to have: the same provider, the same model, the same
request, tried again after a transient condition has had time to clear. A
quota that never recovers still surfaces as a failure once retries are
exhausted -- see the "gives up" test below.

Real sleeping would make this suite slow and flaky under load, so every test
injects a recording stand-in for ``time.sleep`` and asserts the delays chosen
rather than waiting through them.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from groundtruth.llm.http import post_json
from groundtruth.llm.models import ProviderRequestError


def recording_sleep() -> tuple[list[float], Callable[[float], None]]:
    calls: list[float] = []

    def sleep(seconds: float) -> None:
        calls.append(seconds)

    return calls, sleep


def sequenced_transport(responses: list[httpx.Response]) -> tuple[httpx.Client, dict[str, int]]:
    """A MockTransport that returns one response per call, holding the last."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[index]

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def call(client: httpx.Client, sleep: Callable[[float], None], **kwargs: object) -> dict:
    return post_json(
        client,
        provider="groq",
        url="https://example/x",
        headers={},
        payload={},
        sleep=sleep,
        **kwargs,
    )


class TestRetryOn429:
    def test_succeeds_after_one_retry_and_waits_for_retry_after(self):
        client, calls = sequenced_transport(
            [
                httpx.Response(429, headers={"retry-after": "9"}, text="rate limited"),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        delays, sleep = recording_sleep()

        body = call(client, sleep)

        assert body == {"ok": True}
        assert calls["n"] == 2
        assert delays == [9.0]

    def test_falls_back_to_exponential_backoff_without_retry_after(self):
        client, calls = sequenced_transport(
            [
                httpx.Response(429, text="rate limited"),
                httpx.Response(429, text="rate limited"),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        delays, sleep = recording_sleep()

        call(client, sleep)

        assert calls["n"] == 3
        assert delays == [1.0, 2.0]

    def test_a_non_numeric_retry_after_falls_back_to_backoff(self):
        # Some vendors send an HTTP-date instead of delta-seconds.
        client, _ = sequenced_transport(
            [
                httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        delays, sleep = recording_sleep()

        call(client, sleep)

        assert delays == [1.0]

    def test_retry_after_is_capped_rather_than_waited_out_in_full(self):
        client, _ = sequenced_transport(
            [
                httpx.Response(429, headers={"retry-after": "999999"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        delays, sleep = recording_sleep()

        call(client, sleep)

        assert delays == [30.0]

    def test_gives_up_after_max_retries_naming_the_attempt_count(self):
        client, calls = sequenced_transport([httpx.Response(429, text="still limited")])
        delays, sleep = recording_sleep()

        with pytest.raises(ProviderRequestError, match="3 attempts"):
            call(client, sleep, max_retries=2)

        # 1 initial attempt + 2 retries, and a sleep before each retry.
        assert calls["n"] == 3
        assert len(delays) == 2

    def test_zero_max_retries_means_the_first_failure_is_final(self):
        client, calls = sequenced_transport([httpx.Response(429, text="limited")])
        delays, sleep = recording_sleep()

        with pytest.raises(ProviderRequestError):
            call(client, sleep, max_retries=0)

        assert calls["n"] == 1
        assert delays == []


class TestRetryOnTransientServerErrors:
    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_retries_and_succeeds(self, status: int):
        client, calls = sequenced_transport(
            [httpx.Response(status, text="overloaded"), httpx.Response(200, json={"ok": True})]
        )
        delays, sleep = recording_sleep()

        body = call(client, sleep)

        assert body == {"ok": True}
        assert calls["n"] == 2
        assert delays == [1.0]


class TestNoRetryOnClientErrors:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_is_not_retried(self, status: int):
        client, calls = sequenced_transport([httpx.Response(status, text="bad request")])
        delays, sleep = recording_sleep()

        with pytest.raises(ProviderRequestError, match=str(status)):
            call(client, sleep)

        assert calls["n"] == 1
        assert delays == []

    def test_a_transport_failure_is_not_retried(self):
        def explode(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = httpx.Client(transport=httpx.MockTransport(explode))
        delays, sleep = recording_sleep()

        with pytest.raises(ProviderRequestError):
            call(client, sleep)

        assert delays == []
