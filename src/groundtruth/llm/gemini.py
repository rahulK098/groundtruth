"""Google Gemini generateContent.

The odd one out in three ways, all of them handled here rather than leaking
into the caller: the system prompt is ``systemInstruction``, forcing a call
means ``functionCallingConfig.mode = "ANY"`` rather than naming the tool in a
``tool_choice``, and the resolved model comes back as ``modelVersion``.
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

BASE_URL: Final[str] = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL: Final[str] = "gemini-2.5-flash"


class GeminiProvider:
    """Forced function calling against generateContent."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "gemini"
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
            url=f"{BASE_URL}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self._api_key},
            payload={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.schema,
                            }
                        ]
                    }
                ],
                # "ANY" with a single declared function is Gemini's equivalent
                # of naming the tool in tool_choice.
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [tool.name],
                    }
                },
            },
        )

        return Completion(
            model=str(body.get("modelVersion") or self.model),
            tool_input=_tool_input(body),
        )


def _tool_input(body: dict[str, Any]) -> dict[str, Any] | None:
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    if not isinstance(content, dict):
        return None

    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    for part in parts:
        if not isinstance(part, dict):
            continue
        call = part.get("functionCall")
        if isinstance(call, dict) and isinstance(call.get("args"), dict):
            return dict(call["args"])
    return None
