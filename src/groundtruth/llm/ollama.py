"""Local Ollama.

Useful for rehearsing the review loop without spending anything: run
`gt golden generate` against a small local model, throw the candidates away,
and confirm the pipeline works before paying for the real run.

Not recommended for the committed candidate set. A 3B local model proposes
noticeably weaker questions, and the rejection rate stops measuring the
generator's judgment and starts measuring its size.
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

DEFAULT_BASE_URL: Final[str] = "http://localhost:11434"
DEFAULT_MODEL: Final[str] = "qwen2.5:3b"


class OllamaProvider:
    """Ollama's OpenAI-shaped /api/chat, with no credential."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "ollama"
        self.model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"
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
            url=self._url,
            headers={},
            payload={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": temperature, "num_predict": max_tokens},
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
            },
        )

        return Completion(
            model=str(body.get("model") or self.model),
            tool_input=_tool_input(body),
        )


def _tool_input(body: dict[str, Any]) -> dict[str, Any] | None:
    message = body.get("message")
    if not isinstance(message, dict):
        return None
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict):
        return None
    # Ollama returns arguments already decoded, unlike the OpenAI wire format.
    arguments = function.get("arguments")
    return dict(arguments) if isinstance(arguments, dict) else None
