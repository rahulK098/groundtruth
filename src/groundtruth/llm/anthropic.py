"""Anthropic Messages API.

Over HTTP rather than through the SDK, so candidate generation needs no
optional extra and every provider fails in the same way. The request is small
and the response shape is stable; the SDK's value is in streaming and retries,
neither of which this path uses.
"""

from __future__ import annotations

from typing import Any, Final

import httpx

from groundtruth.llm.http import post_json
from groundtruth.llm.models import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    Completion,
    ToolSpec,
)

API_URL: Final[str] = "https://api.anthropic.com/v1/messages"
API_VERSION: Final[str] = "2023-06-01"
DEFAULT_MODEL: Final[str] = "claude-sonnet-5"


class AnthropicProvider:
    """Forced tool use against the Messages API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "anthropic"
        self.model = model
        self._api_key = api_key
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
        body = post_json(
            self._client,
            provider=self.name,
            url=API_URL,
            headers={"x-api-key": self._api_key, "anthropic-version": API_VERSION},
            payload={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                # Anthropic takes the system prompt as a top-level field, not
                # as a message with role "system".
                "system": system,
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.schema,
                    }
                ],
                "tool_choice": {"type": "tool", "name": tool.name},
                "messages": [{"role": "user", "content": user}],
            },
        )

        return Completion(
            model=str(body.get("model") or self.model),
            tool_input=_tool_input(body),
        )


def _tool_input(body: dict[str, Any]) -> dict[str, Any] | None:
    content = body.get("content")
    if not isinstance(content, list):
        return None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            arguments = block.get("input")
            if isinstance(arguments, dict):
                return dict(arguments)
    return None
