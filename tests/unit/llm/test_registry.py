"""Choosing a provider from settings.

The property this file defends: **selection is explicit and failure is loud.**
A provider whose credentials are missing raises with the name of the variable
to set; it never quietly becomes a different provider. A candidate set built
from whichever vendor happened to answer is not reproducible, and its
per-candidate `generator_model` would make the mixture look deliberate.
"""

from __future__ import annotations

import pytest

from groundtruth.llm.models import ProviderNotConfiguredError
from groundtruth.llm.registry import PROVIDER_NAMES, build_provider
from groundtruth.settings import Settings

#: Every variable the registry reads. Cleared before each test: a developer
#: with OPENAI_API_KEY exported would otherwise see "refuses without a key"
#: pass for the wrong reason.
PROVIDER_ENV = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_API_KEY2",
    "AZURE_OPENAI_KEY2",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in PROVIDER_ENV:
        monkeypatch.delenv(variable, raising=False)


def settings(**values: str) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


class TestNames:
    def test_every_documented_provider_is_registered(self):
        assert set(PROVIDER_NAMES) == {
            "anthropic",
            "azure",
            "openrouter",
            "groq",
            "gemini",
            "deepseek",
            "openai",
            "ollama",
        }

    def test_an_unknown_provider_lists_the_known_ones(self):
        with pytest.raises(ProviderNotConfiguredError) as excinfo:
            build_provider("nope", settings())
        assert "azure" in str(excinfo.value)


class TestAzure:
    def test_builds_from_endpoint_and_key(self):
        provider = build_provider(
            "azure",
            settings(
                azure_openai_api_key="k",
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_deployment="gpt-4o",
            ),
        )
        assert provider.name == "azure"
        assert provider.model == "gpt-4o"

    def test_accepts_the_secondary_key_alone(self):
        provider = build_provider(
            "azure",
            settings(
                azure_openai_api_key2="only-secondary",
                azure_openai_endpoint="https://example.openai.azure.com",
            ),
        )
        assert provider.name == "azure"

    def test_missing_key_names_the_variable(self):
        with pytest.raises(ProviderNotConfiguredError, match="AZURE_OPENAI_API_KEY"):
            build_provider(
                "azure", settings(azure_openai_endpoint="https://example.openai.azure.com")
            )

    def test_missing_endpoint_names_the_variable(self):
        with pytest.raises(ProviderNotConfiguredError, match="AZURE_OPENAI_ENDPOINT"):
            build_provider("azure", settings(azure_openai_api_key="k"))


class TestKeyedProviders:
    @pytest.mark.parametrize(
        "name, field, variable",
        [
            ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY"),
            ("groq", "groq_api_key", "GROQ_API_KEY"),
            ("openrouter", "openrouter_api_key", "OPENROUTER_API_KEY"),
            ("deepseek", "deepseek_api_key", "DEEPSEEK_API_KEY"),
            ("gemini", "gemini_api_key", "GEMINI_API_KEY"),
            ("openai", "openai_api_key", "OPENAI_API_KEY"),
        ],
    )
    def test_builds_with_its_key_and_refuses_without_it(self, name: str, field: str, variable: str):
        provider = build_provider(name, settings(**{field: "k"}))
        assert provider.name == name
        assert provider.model

        with pytest.raises(ProviderNotConfiguredError, match=variable):
            build_provider(name, settings())


class TestOllama:
    def test_needs_no_credential(self):
        # The point of the local provider: it runs with nothing configured.
        provider = build_provider("ollama", settings())
        assert provider.name == "ollama"


class TestModelOverride:
    def test_an_explicit_model_wins_over_the_default(self):
        provider = build_provider("groq", settings(groq_api_key="k"), model="llama-3.1-8b-instant")
        assert provider.model == "llama-3.1-8b-instant"

    def test_settings_model_wins_over_the_built_in_default(self):
        provider = build_provider("groq", settings(groq_api_key="k", groq_model="from-settings"))
        assert provider.model == "from-settings"
