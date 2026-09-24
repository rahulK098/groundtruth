"""Every provider, against a mock transport.

One shared contract is under test five times: a forced tool call goes out in
the shape that vendor documents, the arguments come back as a dict, and the
model id recorded is **the one the response reports** rather than the one that
was requested. That last property is what keeps a floating alias out of a
candidate's provenance.

No network and no key: httpx's MockTransport answers every request, so these
run in the gate environment like any other unit test.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from groundtruth.llm.anthropic import AnthropicProvider
from groundtruth.llm.gemini import GeminiProvider
from groundtruth.llm.models import (
    Completion,
    ProviderRequestError,
    ToolSpec,
)
from groundtruth.llm.ollama import OllamaProvider
from groundtruth.llm.openai_compatible import AzureOpenAIProvider, OpenAICompatibleProvider

TOOL = ToolSpec(
    name="propose_candidate",
    description="Propose one candidate question.",
    schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

ARGUMENTS = {"query": "what is the summary judgment standard", "category": "factual-lookup"}


def transport(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def capture() -> tuple[list[httpx.Request], Any]:
    """A handler that records requests; the body it returns is set per test."""
    seen: list[httpx.Request] = []
    return seen, seen.append


# --- per-provider response shapes -------------------------------------------


def anthropic_body(model: str = "claude-sonnet-5-20260101") -> dict[str, Any]:
    return {
        "model": model,
        "content": [{"type": "tool_use", "name": TOOL.name, "input": ARGUMENTS}],
    }


def openai_body(model: str = "gpt-4o-2024-11-20") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": TOOL.name,
                                "arguments": json.dumps(ARGUMENTS),
                            },
                        }
                    ]
                }
            }
        ],
    }


def gemini_body(model: str = "gemini-2.5-flash-002") -> dict[str, Any]:
    return {
        "modelVersion": model,
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": TOOL.name, "args": ARGUMENTS}}]}}
        ],
    }


def ollama_body(model: str = "qwen2.5:3b") -> dict[str, Any]:
    return {
        "model": model,
        "message": {"tool_calls": [{"function": {"name": TOOL.name, "arguments": ARGUMENTS}}]},
    }


def build(name: str, client: httpx.Client) -> Any:
    builders = {
        "anthropic": lambda: AnthropicProvider(api_key="k", model="claude-sonnet-5", client=client),
        "azure": lambda: AzureOpenAIProvider(
            api_keys=("k",),
            endpoint="https://example.openai.azure.com",
            deployment="chat",
            api_version="2024-10-21",
            client=client,
        ),
        "groq": lambda: OpenAICompatibleProvider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key="k",
            model="llama-3.3-70b-versatile",
            client=client,
        ),
        "gemini": lambda: GeminiProvider(api_key="k", model="gemini-2.5-flash", client=client),
        "ollama": lambda: OllamaProvider(
            base_url="http://localhost:11434", model="qwen2.5:3b", client=client
        ),
    }
    return builders[name]()


BODIES = {
    "anthropic": anthropic_body,
    "azure": openai_body,
    "groq": openai_body,
    "gemini": gemini_body,
    "ollama": ollama_body,
}

EVERY_PROVIDER = pytest.mark.parametrize("provider_name", sorted(BODIES))


def complete(provider: Any) -> Completion:
    return provider.complete(system="sys", user="usr", tool=TOOL, max_tokens=512, temperature=0.0)


class TestTheSharedContract:
    @EVERY_PROVIDER
    def test_returns_the_tool_arguments_as_a_dict(self, provider_name: str):
        client = transport(lambda request: httpx.Response(200, json=BODIES[provider_name]()))
        result = complete(build(provider_name, client))

        assert result.tool_input == ARGUMENTS

    @EVERY_PROVIDER
    def test_records_the_model_the_response_reports_not_the_one_requested(self, provider_name: str):
        # The whole point: an alias must never reach a candidate's provenance.
        body = BODIES[provider_name]("resolved-snapshot-id")
        client = transport(lambda request: httpx.Response(200, json=body))

        result = complete(build(provider_name, client))
        assert result.model == "resolved-snapshot-id"

    @EVERY_PROVIDER
    def test_no_tool_call_is_reported_rather_than_invented(self, provider_name: str):
        empty: dict[str, Any] = {
            "anthropic": {"model": "m", "content": [{"type": "text", "text": "I decline"}]},
            "azure": {"model": "m", "choices": [{"message": {"content": "I decline"}}]},
            "groq": {"model": "m", "choices": [{"message": {"content": "I decline"}}]},
            "gemini": {"modelVersion": "m", "candidates": [{"content": {"parts": []}}]},
            "ollama": {"model": "m", "message": {"content": "I decline"}},
        }[provider_name]
        client = transport(lambda request: httpx.Response(200, json=empty))

        result = complete(build(provider_name, client))
        assert result.tool_input is None
        assert result.model == "m"

    @EVERY_PROVIDER
    def test_an_http_error_names_the_provider_and_the_status(self, provider_name: str):
        client = transport(lambda request: httpx.Response(401, text="nope"))

        with pytest.raises(ProviderRequestError) as excinfo:
            complete(build(provider_name, client))

        message = str(excinfo.value)
        assert provider_name in message
        assert "401" in message

    @EVERY_PROVIDER
    def test_a_transport_failure_is_wrapped_not_leaked(self, provider_name: str):
        def explode(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(ProviderRequestError, match=provider_name):
            complete(build(provider_name, transport(explode)))

    @EVERY_PROVIDER
    def test_a_non_json_body_is_wrapped_not_leaked(self, provider_name: str):
        client = transport(lambda request: httpx.Response(200, text="<html>gateway</html>"))

        with pytest.raises(ProviderRequestError, match=provider_name):
            complete(build(provider_name, client))


class TestAnthropicWireFormat:
    def test_forces_the_tool_and_sends_the_system_prompt_top_level(self):
        seen, handler = capture()
        client = transport(
            lambda request: (handler(request), httpx.Response(200, json=anthropic_body()))[1]
        )
        complete(build("anthropic", client))

        (request,) = seen
        body = json.loads(request.content)
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "k"
        assert request.headers["anthropic-version"]
        # System is a top-level field for Anthropic, not a message.
        assert body["system"] == "sys"
        assert body["tool_choice"] == {"type": "tool", "name": TOOL.name}
        assert body["tools"][0]["input_schema"] == TOOL.schema


class TestOpenAIWireFormat:
    def test_sends_a_function_tool_with_the_system_as_a_message(self):
        seen, handler = capture()
        client = transport(
            lambda request: (handler(request), httpx.Response(200, json=openai_body()))[1]
        )
        complete(build("groq", client))

        (request,) = seen
        body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer k"
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert body["tools"][0]["function"]["parameters"] == TOOL.schema
        assert body["tool_choice"] == {"type": "function", "function": {"name": TOOL.name}}

    def test_tool_arguments_that_are_not_json_are_an_error_not_a_guess(self):
        body = {
            "model": "m",
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"function": {"name": TOOL.name, "arguments": "{not json"}}]
                    }
                }
            ],
        }
        client = transport(lambda request: httpx.Response(200, json=body))

        with pytest.raises(ProviderRequestError, match="arguments"):
            complete(build("groq", client))


class TestAzure:
    def test_url_carries_the_deployment_and_api_version(self):
        seen, handler = capture()
        client = transport(
            lambda request: (handler(request), httpx.Response(200, json=openai_body()))[1]
        )
        complete(build("azure", client))

        (request,) = seen
        assert request.url.path == "/openai/deployments/chat/chat/completions"
        assert request.url.params["api-version"] == "2024-10-21"
        # Azure authenticates with api-key, not a bearer token.
        assert request.headers["api-key"] == "k"
        assert "authorization" not in request.headers

    def test_falls_back_to_the_secondary_key_on_an_auth_failure(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.headers["api-key"] == "primary":
                return httpx.Response(401, text="expired")
            return httpx.Response(200, json=openai_body())

        provider = AzureOpenAIProvider(
            api_keys=("primary", "secondary"),
            endpoint="https://example.openai.azure.com",
            deployment="chat",
            api_version="2024-10-21",
            client=transport(handler),
        )
        result = complete(provider)

        # Key rotation, not provider fallback: same endpoint, same deployment,
        # same model -- so the candidate set stays reproducible.
        assert result.tool_input == ARGUMENTS
        assert [r.headers["api-key"] for r in seen] == ["primary", "secondary"]

    def test_gives_up_when_every_key_fails(self):
        provider = AzureOpenAIProvider(
            api_keys=("primary", "secondary"),
            endpoint="https://example.openai.azure.com",
            deployment="chat",
            api_version="2024-10-21",
            client=transport(lambda request: httpx.Response(401, text="expired")),
        )
        with pytest.raises(ProviderRequestError, match="azure"):
            complete(provider)

    def test_a_non_auth_failure_does_not_burn_the_second_key(self, monkeypatch: pytest.MonkeyPatch):
        # A 500 is transient-retried (same key, see http.py) before Azure's
        # own key-rotation logic ever sees it -- a server that is overloaded
        # is overloaded regardless of which key asks, so there is nothing for
        # a second key to fix. Real sleeping is patched out; the count of
        # attempts on "primary" is http.py's default and not this test's
        # concern, only that "secondary" is never touched.
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(500, text="server error")

        provider = AzureOpenAIProvider(
            api_keys=("primary", "secondary"),
            endpoint="https://example.openai.azure.com",
            deployment="chat",
            api_version="2024-10-21",
            client=transport(handler),
        )
        with pytest.raises(ProviderRequestError):
            complete(provider)
        assert seen
        assert {r.headers["api-key"] for r in seen} == {"primary"}

    def test_endpoint_trailing_slash_does_not_double_up(self):
        seen, handler = capture()
        provider = AzureOpenAIProvider(
            api_keys=("k",),
            endpoint="https://example.openai.azure.com/",
            deployment="chat",
            api_version="2024-10-21",
            client=transport(
                lambda request: (handler(request), httpx.Response(200, json=openai_body()))[1]
            ),
        )
        complete(provider)
        assert "//openai" not in str(seen[0].url)


class TestGeminiWireFormat:
    def test_forces_a_function_call_and_sends_the_key_as_a_header(self):
        seen, handler = capture()
        client = transport(
            lambda request: (handler(request), httpx.Response(200, json=gemini_body()))[1]
        )
        complete(build("gemini", client))

        (request,) = seen
        body = json.loads(request.content)
        assert request.headers["x-goog-api-key"] == "k"
        assert body["systemInstruction"]["parts"][0]["text"] == "sys"
        assert body["tools"][0]["functionDeclarations"][0]["name"] == TOOL.name
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"


class TestOllamaWireFormat:
    def test_posts_a_non_streaming_chat_request(self):
        seen, handler = capture()
        client = transport(
            lambda request: (handler(request), httpx.Response(200, json=ollama_body()))[1]
        )
        complete(build("ollama", client))

        (request,) = seen
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["stream"] is False
        assert body["messages"][0]["role"] == "system"
        assert body["tools"][0]["function"]["name"] == TOOL.name
