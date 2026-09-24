"""Building a provider from settings, by name.

Deliberately **no fallback chain**. A provider whose credentials are missing
raises naming the variable to set; it never becomes a different provider.

The reason is specific to this project rather than stylistic: every candidate
records the model that produced it, and the report breaks results down by
origin. A chain would let one run be answered by Azure, the next by whatever
key happened to still work, and nothing in the record would mark that as
unintended. Key rotation *within* one provider is a different matter and is
supported -- see :class:`~groundtruth.llm.openai_compatible.AzureOpenAIProvider`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import httpx
from pydantic import SecretStr

from groundtruth.llm import anthropic as anthropic_provider
from groundtruth.llm import gemini as gemini_provider
from groundtruth.llm import ollama as ollama_provider
from groundtruth.llm.models import ChatProvider, ProviderNotConfiguredError
from groundtruth.llm.openai_compatible import (
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_AZURE_DEPLOYMENT,
    AzureOpenAIProvider,
    OpenAICompatibleProvider,
)
from groundtruth.settings import Settings

#: Bearer-token vendors that speak OpenAI chat-completions unchanged.
#: name -> (settings prefix, base URL, default model, env var to name in errors)
_OPENAI_FAMILY: Final[dict[str, tuple[str, str, str]]] = {
    "openai": ("openai", "https://api.openai.com/v1", "gpt-4o"),
    "groq": ("groq", "https://api.groq.com/openai/v1", "qwen/qwen3.8-27b"),
    "openrouter": (
        "openrouter",
        "https://openrouter.ai/api/v1",
        # The :free variants answer on a zero balance, rate-limited. A paid
        # model returns 402 without credits, which is a confusing first
        # experience for someone just trying the harness out.
        "meta-llama/llama-3.3-70b-instruct:free",
    ),
    "deepseek": ("deepseek", "https://api.deepseek.com/v1", "deepseek-chat"),
}

PROVIDER_NAMES: Final[tuple[str, ...]] = (
    "anthropic",
    "azure",
    "deepseek",
    "gemini",
    "groq",
    "ollama",
    "openai",
    "openrouter",
)

#: Sent by OpenRouter convention so requests are attributable.
_OPENROUTER_HEADERS: Final[dict[str, str]] = {
    "HTTP-Referer": "https://github.com/rahulK098/groundtruth",
    "X-Title": "Groundtruth",
}


def _env_name(prefix: str) -> str:
    return f"{prefix.upper()}_API_KEY"


def _require(secret: SecretStr | None, variable: str, provider: str) -> str:
    if secret is None:
        raise ProviderNotConfiguredError(
            f"{variable} is not set, which provider {provider!r} requires. Add it "
            f"to a .env file in the project root, or choose another provider with "
            f"--provider."
        )
    return secret.get_secret_value()


def _build_azure(
    settings: Settings, model: str | None, client: httpx.Client | None
) -> ChatProvider:
    keys = tuple(
        secret.get_secret_value()
        for secret in (settings.azure_openai_api_key, settings.azure_openai_api_key2)
        if secret is not None
    )
    if not keys:
        raise ProviderNotConfiguredError(
            "AZURE_OPENAI_API_KEY is not set, which provider 'azure' requires. "
            "Add it (and AZURE_OPENAI_ENDPOINT) to a .env file in the project root."
        )
    if not settings.azure_openai_endpoint:
        raise ProviderNotConfiguredError(
            "AZURE_OPENAI_ENDPOINT is not set, which provider 'azure' requires. "
            "It looks like https://<resource>.openai.azure.com"
        )

    return AzureOpenAIProvider(
        api_keys=keys,
        endpoint=settings.azure_openai_endpoint,
        # On Azure the deployment name selects the model, so --model sets it.
        deployment=model or settings.azure_openai_deployment or DEFAULT_AZURE_DEPLOYMENT,
        api_version=settings.azure_openai_api_version or DEFAULT_AZURE_API_VERSION,
        client=client,
    )


def _build_openai_family(
    name: str, settings: Settings, model: str | None, client: httpx.Client | None
) -> ChatProvider:
    prefix, base_url, default_model = _OPENAI_FAMILY[name]
    api_key = _require(getattr(settings, f"{prefix}_api_key"), _env_name(prefix), name)
    return OpenAICompatibleProvider(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model or getattr(settings, f"{prefix}_model") or default_model,
        client=client,
        extra_headers=_OPENROUTER_HEADERS if name == "openrouter" else None,
    )


def _build_anthropic(
    settings: Settings, model: str | None, client: httpx.Client | None
) -> ChatProvider:
    return anthropic_provider.AnthropicProvider(
        api_key=_require(settings.anthropic_api_key, "ANTHROPIC_API_KEY", "anthropic"),
        model=model or settings.anthropic_model or anthropic_provider.DEFAULT_MODEL,
        client=client,
    )


def _build_gemini(
    settings: Settings, model: str | None, client: httpx.Client | None
) -> ChatProvider:
    return gemini_provider.GeminiProvider(
        api_key=_require(settings.gemini_api_key, "GEMINI_API_KEY", "gemini"),
        model=model or settings.gemini_model or gemini_provider.DEFAULT_MODEL,
        client=client,
    )


def _build_ollama(
    settings: Settings, model: str | None, client: httpx.Client | None
) -> ChatProvider:
    return ollama_provider.OllamaProvider(
        base_url=settings.ollama_base_url or ollama_provider.DEFAULT_BASE_URL,
        model=model or settings.ollama_model or ollama_provider.DEFAULT_MODEL,
        client=client,
    )


_BUILDERS: Final[dict[str, Callable[[Settings, str | None, httpx.Client | None], ChatProvider]]] = {
    "anthropic": _build_anthropic,
    "azure": _build_azure,
    "gemini": _build_gemini,
    "ollama": _build_ollama,
    **{
        name: (lambda n: lambda s, m, c: _build_openai_family(n, s, m, c))(name)
        for name in _OPENAI_FAMILY
    },
}


def build_provider(
    name: str,
    settings: Settings,
    *,
    model: str | None = None,
    client: httpx.Client | None = None,
) -> ChatProvider:
    """Build the named provider, or raise naming what is missing."""
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ProviderNotConfiguredError(
            f"unknown provider {name!r}; expected one of {', '.join(PROVIDER_NAMES)}"
        )
    return builder(settings, model, client)
