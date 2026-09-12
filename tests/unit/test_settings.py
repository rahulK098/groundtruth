"""Credential handling.

Two things matter here: a missing credential must fail with a message that says
what to do about it, and a present credential must never be printable by
accident.
"""

from __future__ import annotations

import pytest

from groundtruth.settings import MissingCredentialError, Settings


def isolated(**values) -> Settings:
    # _env_file=None so these tests never pick up a developer's real .env.
    # Without it, the "missing credential" cases would pass or fail depending
    # on whose machine ran them.
    return Settings(_env_file=None, **values)


class TestMissingCredentials:
    def test_missing_courtlistener_token_raises(self):
        with pytest.raises(MissingCredentialError, match="COURTLISTENER_API_TOKEN"):
            isolated().require_courtlistener_token()

    def test_courtlistener_error_says_where_to_get_one(self):
        # An error that only says "missing" sends someone hunting. This one
        # names the signup URL, the variable, and the file.
        with pytest.raises(MissingCredentialError) as exc:
            isolated().require_courtlistener_token()
        message = str(exc.value)
        assert "courtlistener.com" in message
        assert ".env" in message

    def test_courtlistener_error_says_the_gate_does_not_need_it(self):
        # Otherwise a reader assumes the project cannot be reproduced without
        # signing up, which is the opposite of true.
        with pytest.raises(MissingCredentialError) as exc:
            isolated().require_courtlistener_token()
        assert "no credentials" in str(exc.value)

    def test_missing_anthropic_key_raises(self):
        with pytest.raises(MissingCredentialError, match="ANTHROPIC_API_KEY"):
            isolated().require_anthropic_key()

    def test_anthropic_error_says_the_gate_never_calls_the_api(self):
        with pytest.raises(MissingCredentialError) as exc:
            isolated().require_anthropic_key()
        assert "never calls" in str(exc.value)


class TestPresentCredentials:
    def test_returns_the_courtlistener_token(self):
        assert (
            isolated(courtlistener_api_token="tok-123").require_courtlistener_token() == "tok-123"
        )

    def test_returns_the_anthropic_key(self):
        assert isolated(anthropic_api_key="sk-abc").require_anthropic_key() == "sk-abc"


class TestSecrecy:
    """A leaked key in a log line is a real incident, not a style problem."""

    def test_token_is_not_in_the_repr(self):
        assert "tok-123" not in repr(isolated(courtlistener_api_token="tok-123"))

    def test_token_is_not_in_the_string_form(self):
        assert "tok-123" not in str(isolated(courtlistener_api_token="tok-123"))

    def test_anthropic_key_is_not_in_the_repr(self):
        assert "sk-abc" not in repr(isolated(anthropic_api_key="sk-abc"))

    def test_serialization_masks_the_secret(self):
        # model_dump is what an accidental "log the settings" call would hit.
        dumped = str(isolated(courtlistener_api_token="tok-123").model_dump())
        assert "tok-123" not in dumped

    def test_value_is_still_reachable_deliberately(self):
        # Masked by default, available on purpose -- via the explicit accessor.
        settings = isolated(courtlistener_api_token="tok-123")
        assert settings.courtlistener_api_token is not None
        assert settings.courtlistener_api_token.get_secret_value() == "tok-123"
