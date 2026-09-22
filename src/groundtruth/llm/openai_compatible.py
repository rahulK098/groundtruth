"""The OpenAI chat-completions family: OpenAI, Azure, Groq, OpenRouter, DeepSeek.

One wire format, two authentication styles. Azure puts the model in the URL as
a *deployment* and authenticates with ``api-key``; everyone else names the
model in the body and sends a bearer token. Both funnel through
:func:`_complete` so the tool-call parsing has exactly one implementation.

Azure also supports a **secondary key**, which this honours: an auth failure
on the primary retries once on the secondary. That is key rotation -- same
endpoint, same deployment, same model -- and so it leaves the candidate set
reproducible, unlike a fallback to a different vendor.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

import httpx

from groundtruth.llm.http import post_json
from groundtruth.llm.models import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    Completion,
    ProviderRequestError,
    ToolSpec,
)

#: Statuses worth retrying on the secondary Azure key. A 500 is the service
#: failing, not the key, so burning the second key on it teaches nothing.
_AUTH_FAILURES: Final[frozenset[int]] = frozenset({401, 403})

DEFAULT_AZURE_API_VERSION: Final[str] = "2024-10-21"
DEFAULT_AZURE_DEPLOYMENT: Final[str] = "chat"


def _payload(
    *,
    model: str | None,
    system: str,
    user: str,
    tool: ToolSpec,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": tool.name}},
    }
    if model is not None:
        # Azure omits this: the deployment in the URL selects the model.
        payload["model"] = model
    return payload


def _tool_input(body: dict[str, Any], provider: str) -> dict[str, Any] | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None

    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict):
        return None

    # Unlike every other provider here, the arguments arrive as a JSON *string*.
    raw = function.get("arguments")
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderRequestError(
            f"{provider}: tool arguments were not valid JSON: {raw[:200]!r}"
        ) from exc
    return dict(parsed) if isinstance(parsed, dict) else None


def _complete(
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    params: dict[str, str] | None,
    client: httpx.Client | None,
    model: str | None,
    fallback_model: str,
    system: str,
    user: str,
    tool: ToolSpec,
    max_tokens: int,
    temperature: float,
) -> Completion:
    body = post_json(
        client,
        provider=provider,
        url=url,
        headers=headers,
        params=params,
        payload=_payload(
            model=model,
            system=system,
            user=user,
            tool=tool,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return Completion(
        model=str(body.get("model") or fallback_model),
        tool_input=_tool_input(body, provider),
    )


class OpenAICompatibleProvider:
    """Any vendor speaking OpenAI chat-completions with a bearer token."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._client = client
        self._extra_headers = extra_headers or {}

    def complete(
        self,
        *,
        system: str,
        user: str,
        tool: ToolSpec,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Completion:
        return _complete(
            provider=self.name,
            url=self._url,
            headers={"Authorization": f"Bearer {self._api_key}", **self._extra_headers},
            params=None,
            client=self._client,
            model=self.model,
            fallback_model=self.model,
            system=system,
            user=user,
            tool=tool,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class AzureOpenAIProvider:
    """Azure OpenAI, with deployment-in-URL addressing and key rotation."""

    def __init__(
        self,
        *,
        api_keys: Sequence[str],
        endpoint: str,
        deployment: str = DEFAULT_AZURE_DEPLOYMENT,
        api_version: str = DEFAULT_AZURE_API_VERSION,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "azure"
        #: The deployment *is* the model selector on Azure. The underlying
        #: model id comes back on the response and is what gets recorded.
        self.model = deployment
        self._keys = tuple(key for key in api_keys if key)
        self._url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
        self._api_version = api_version
        self._client = client

    def complete(
        self,
        *,
        system: str,
        user: str,
        tool: ToolSpec,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Completion:
        last: ProviderRequestError | None = None
        for index, key in enumerate(self._keys):
            try:
                return _complete(
                    provider=self.name,
                    url=self._url,
                    headers={"api-key": key},
                    params={"api-version": self._api_version},
                    client=self._client,
                    model=None,
                    fallback_model=self.model,
                    system=system,
                    user=user,
                    tool=tool,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except ProviderRequestError as exc:
                last = exc
                is_last_key = index == len(self._keys) - 1
                if is_last_key or not _is_auth_failure(exc):
                    raise
        raise last or ProviderRequestError("azure: no API key configured")


def _is_auth_failure(exc: ProviderRequestError) -> bool:
    return any(f"HTTP {status}" in str(exc) for status in _AUTH_FAILURES)
