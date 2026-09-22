"""Runtime credentials and settings.

Secrets are read from the environment or a gitignored ``.env``, never from
code or configuration files. They are held as ``SecretStr`` so an accidental
log line, repr, or API response prints ``**********`` instead of the value.

Deliberately unprefixed: ``ANTHROPIC_API_KEY`` and ``COURTLISTENER_API_TOKEN``
are the names those vendors document, and inventing a ``GT_`` prefix would
mean a key that already works everywhere else silently fails here.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingCredentialError(Exception):
    """A required credential is not configured."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    #: Free token from courtlistener.com. Needed only to fetch the corpus,
    #: never to run the gate -- the corpus snapshot is committed.
    courtlistener_api_token: SecretStr | None = None

    #: Needed only to regenerate generation-layer judgments. The gate runs
    #: against the committed judgment cache and never calls Anthropic.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = ""

    # --- other providers ----------------------------------------------------
    #
    # Candidate generation and the judge run through one provider protocol
    # (see groundtruth.llm), so any of these can drive them. Which one is
    # always chosen explicitly -- there is no fallback chain, because a rerun
    # that quietly used a different vendor would silently change what the
    # candidate set is.

    #: Azure addresses a *deployment*, not a model, so it needs an endpoint
    #: and a deployment name alongside the key. AZURE_OPENAI_KEY is accepted
    #: as well, since that is the spelling the portal's sample code uses.
    azure_openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY")
    )
    #: Azure issues two keys so one can be rotated while the other serves. An
    #: auth failure on the primary retries once here -- same deployment, same
    #: model, so the candidate set stays reproducible.
    azure_openai_api_key2: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_OPENAI_API_KEY2", "AZURE_OPENAI_KEY2")
    )
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = ""

    openai_api_key: SecretStr | None = None
    openai_model: str = ""

    groq_api_key: SecretStr | None = None
    groq_model: str = ""

    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = ""

    deepseek_api_key: SecretStr | None = None
    deepseek_model: str = ""

    gemini_api_key: SecretStr | None = None
    gemini_model: str = ""

    #: Local, so no credential. Present to point at a non-default host.
    ollama_base_url: str = ""
    ollama_model: str = ""

    def require_courtlistener_token(self) -> str:
        if self.courtlistener_api_token is None:
            raise MissingCredentialError(
                "COURTLISTENER_API_TOKEN is not set. Opinion text requires "
                "authentication (the API returns 401 without it). Sign up free "
                "at https://www.courtlistener.com/ and add the token to a .env "
                "file in the project root:\n\n"
                "    COURTLISTENER_API_TOKEN=your_token_here\n\n"
                "This is only needed to re-fetch the corpus. Reproducing the "
                "comparison table needs no credentials at all."
            )
        return self.courtlistener_api_token.get_secret_value()

    def require_anthropic_key(self) -> str:
        if self.anthropic_api_key is None:
            raise MissingCredentialError(
                "ANTHROPIC_API_KEY is not set. It is required only to "
                "regenerate generation-layer judgments; the regression gate "
                "runs against the committed judgment cache and never calls "
                "the API."
            )
        return self.anthropic_api_key.get_secret_value()
