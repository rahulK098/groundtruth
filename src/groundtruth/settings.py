"""Runtime credentials and settings.

Secrets are read from the environment or a gitignored ``.env``, never from
code or configuration files. They are held as ``SecretStr`` so an accidental
log line, repr, or API response prints ``**********`` instead of the value.

Deliberately unprefixed: ``ANTHROPIC_API_KEY`` and ``COURTLISTENER_API_TOKEN``
are the names those vendors document, and inventing a ``GT_`` prefix would
mean a key that already works everywhere else silently fails here.
"""

from __future__ import annotations

from pydantic import SecretStr
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
