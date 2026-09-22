"""Provider value objects and errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

#: Every provider is called with a schema-checked tool rather than asked for
#: JSON in prose. A model asked for prose JSON eventually wraps it in an
#: explanation and the parser has to guess.
DEFAULT_MAX_TOKENS: int = 1200

#: Reduces variation; does not eliminate it. No vendor here offers a seed, so
#: reproducibility comes from committing the output (ADR-0011), not from this.
DEFAULT_TEMPERATURE: float = 0.0

#: Long enough for a slow local model on CPU, short enough that a hung
#: endpoint does not stall a 120-passage run indefinitely.
DEFAULT_TIMEOUT_SECONDS: float = 120.0


class LLMError(Exception):
    """Base for every provider failure."""


class ProviderNotConfiguredError(LLMError):
    """The provider is unknown, or its credentials are missing."""


class ProviderRequestError(LLMError):
    """The provider was reached but did not return a usable answer."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A forced tool call, in vendor-neutral terms.

    ``schema`` is a plain JSON Schema object. Each provider wraps it in
    whatever shape its API documents -- ``input_schema`` for Anthropic,
    ``function.parameters`` for the OpenAI-compatible family, and so on.
    """

    name: str
    description: str
    schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Completion:
    """One forced tool call, and the model that actually produced it.

    ``tool_input`` is ``None`` when the model answered without calling the
    tool -- a refusal or a decline. That is reported rather than treated as an
    error, because a passage the model declines is a real outcome the
    rejection counts should show.
    """

    model: str
    tool_input: dict[str, Any] | None


@runtime_checkable
class ChatProvider(Protocol):
    """What candidate generation needs from a vendor, and nothing more."""

    #: Registry name, as it appears in `--provider` and in the run record.
    name: str
    #: Requested model or deployment. The *resolved* id is on the Completion.
    model: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        tool: ToolSpec,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Completion: ...
