"""Tests for SDK client surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from dottxt.client import (
    AsyncDotTxt,
    DotTxt,
    InvalidOutputError,
    _completion_finish_reason,
)

MODEL_ID = "openai/gpt-oss-20b"
CONTACT_JSON = '{"name": "John Smith"}'
CONTACT_SCHEMA_JSON = '{"type": "object", "properties": {"name": {"type": "string"}}}'


class Contact(BaseModel):
    """Example structured output model."""

    name: str


@dataclass
class FakeMessage:
    """Minimal completion message stub."""

    content: str | None


@dataclass
class FakeChoice:
    """Minimal completion choice stub."""

    message: FakeMessage
    finish_reason: str | None = None


@dataclass
class FakeCompletion:
    """Minimal chat completion stub."""

    model: str
    choices: list[FakeChoice]


@dataclass
class FakeModel:
    """Minimal model entry stub."""

    id: str
    object: str = "model"


@dataclass
class FakeModelPage:
    """Minimal model page stub."""

    data: list[FakeModel]
    object: str = "list"


class FakeSyncCompletions:
    """Sync completions namespace stub."""

    def __init__(self) -> None:
        """Initialize the stub state."""
        self.calls: list[dict[str, Any]] = []
        self.result = FakeCompletion(
            model=MODEL_ID,
            choices=[FakeChoice(message=FakeMessage(CONTACT_JSON))],
        )

    def create(self, **kwargs: Any) -> FakeCompletion:
        """Record the call and return a fake completion."""
        self.calls.append(kwargs)
        return self.result


class FakeAsyncCompletions:
    """Async completions namespace stub."""

    def __init__(self) -> None:
        """Initialize the stub state."""
        self.calls: list[dict[str, Any]] = []
        self.result = FakeCompletion(
            model=MODEL_ID,
            choices=[FakeChoice(message=FakeMessage(CONTACT_JSON))],
        )

    async def create(self, **kwargs: Any) -> FakeCompletion:
        """Record the call and return a fake completion."""
        self.calls.append(kwargs)
        return self.result


class FakeSyncResponses:
    """Sync responses namespace stub."""

    def __init__(self) -> None:
        """Initialize the stub state."""
        self.calls: list[dict[str, Any]] = []
        self.result: dict[str, str] = {"id": "resp_123"}

    def create(self, **kwargs: Any) -> dict[str, str]:
        """Record the call and return a fake response."""
        self.calls.append(kwargs)
        return self.result


class FakeAsyncResponses:
    """Async responses namespace stub."""

    def __init__(self) -> None:
        """Initialize the stub state."""
        self.calls: list[dict[str, Any]] = []
        self.result: dict[str, str] = {"id": "resp_123"}

    async def create(self, **kwargs: Any) -> dict[str, str]:
        """Record the call and return a fake response."""
        self.calls.append(kwargs)
        return self.result


class FakeSyncModels:
    """Sync models namespace stub."""

    def __init__(self) -> None:
        """Initialize the stub state."""
        self.calls = 0
        self.result = FakeModelPage(data=[FakeModel(id=MODEL_ID)])

    def list(self) -> FakeModelPage:
        """Record the call and return a fake model page."""
        self.calls += 1
        return self.result


class FakeAsyncModels:
    """Async models namespace stub."""

    def __init__(self) -> None:
        """Initialize the stub state."""
        self.calls = 0
        self.result = FakeModelPage(data=[FakeModel(id=MODEL_ID)])

    async def list(self) -> FakeModelPage:
        """Record the call and return a fake model page."""
        self.calls += 1
        return self.result


class FakeOpenAIClient:
    """Fake sync OpenAI SDK client."""

    instances: list[FakeOpenAIClient] = []

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the fake client."""
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(
            completions=FakeSyncCompletions(), responses=object()
        )
        self.responses = FakeSyncResponses()
        self.models = FakeSyncModels()
        self.closed = False
        self.__class__.instances.append(self)

    def close(self) -> None:
        """Record that the client was closed."""
        self.closed = True


class FakeAsyncOpenAIClient:
    """Fake async OpenAI SDK client."""

    instances: list[FakeAsyncOpenAIClient] = []

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the fake client."""
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(
            completions=FakeAsyncCompletions(),
            responses=object(),
        )
        self.responses = FakeAsyncResponses()
        self.models = FakeAsyncModels()
        self.closed = False
        self.__class__.instances.append(self)

    async def close(self) -> None:
        """Record that the client was closed."""
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fake_clients() -> None:
    """Reset class-level fake client state between tests."""
    FakeOpenAIClient.instances.clear()
    FakeAsyncOpenAIClient.instances.clear()


@pytest.fixture
def patch_sync_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the sync OpenAI SDK constructor."""
    monkeypatch.setattr("dottxt.client.OpenAISDK", FakeOpenAIClient)


@pytest.fixture
def patch_async_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the async OpenAI SDK constructor."""
    monkeypatch.setattr("dottxt.client.AsyncOpenAISDK", FakeAsyncOpenAIClient)


def _sync_sdk_client() -> FakeOpenAIClient:
    """Return the current fake sync SDK client."""
    return FakeOpenAIClient.instances[0]


def _async_sdk_client() -> FakeAsyncOpenAIClient:
    """Return the current fake async SDK client."""
    return FakeAsyncOpenAIClient.instances[0]


def _assert_generated_messages(call: dict[str, Any], *, expected_input: str) -> None:
    """Assert the generated payload keeps the canonical message structure."""
    assert call["messages"] == [{"role": "user", "content": expected_input}]


def _assert_structured_response_format(call: dict[str, Any]) -> None:
    """Assert the request uses OpenAI-style json_schema response format."""
    assert call["response_format"]["type"] == "json_schema"


def test_dottxt_initializes_sdk_with_resolved_config(
    patch_sync_sdk: None,
) -> None:
    """The native client should initialize the OpenAI SDK with config values."""
    client = DotTxt(
        api_key="test-key",
        base_url="https://example.test/v1",
        timeout=12.5,
    )

    sdk_client = _sync_sdk_client()
    assert sdk_client.init_kwargs == {
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "timeout": 12.5,
    }
    assert client.chat.completions is sdk_client.chat.completions
    assert client.models is sdk_client.models
    missing_attr = "responses"
    with pytest.raises(AttributeError):
        getattr(client, missing_attr)


def test_dottxt_initializes_from_environment(
    patch_sync_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native client should resolve settings from environment variables."""
    monkeypatch.setenv("DOTTXT_API_KEY", "env-key")
    monkeypatch.setenv("DOTTXT_BASE_URL", "https://env.example/v1/")

    DotTxt()

    sdk_client = _sync_sdk_client()
    assert sdk_client.init_kwargs == {
        "api_key": "env-key",
        "base_url": "https://env.example/v1",
    }


def test_dottxt_requires_api_key(
    patch_sync_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native client should reject missing API keys."""
    monkeypatch.delenv("DOTTXT_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        DotTxt()


def test_dottxt_forwards_timeout_when_provided(
    patch_sync_sdk: None,
) -> None:
    """The native client should pass timeout through to the SDK."""
    DotTxt(api_key="test-key", timeout=0)

    sdk_client = _sync_sdk_client()
    assert sdk_client.init_kwargs["timeout"] == 0


def test_dottxt_models_namespace_delegates_to_sdk(patch_sync_sdk: None) -> None:
    """The native client should expose the SDK models namespace."""
    client = DotTxt(api_key="test-key")

    response = client.models.list()

    sdk_client = _sync_sdk_client()
    assert sdk_client.models.calls == 1
    assert response.data[0].id == MODEL_ID


def test_dottxt_chat_namespace_exposes_only_completions(
    patch_sync_sdk: None,
) -> None:
    """The native chat namespace should only allow completions."""
    client = DotTxt(api_key="test-key")

    missing_attr = "responses"
    with pytest.raises(AttributeError):
        getattr(client.chat, missing_attr)


def test_dottxt_generate_builds_structured_request(patch_sync_sdk: None) -> None:
    """The native client should wrap schemas before calling the SDK."""
    client = DotTxt(api_key="test-key")

    result = client.generate(
        model="openai/gpt-oss-20b",
        # Keep the public examples aligned with the fake transport.
        input="Extract contact",
        response_format=Contact,
    )

    sdk_client = _sync_sdk_client()
    call = sdk_client.chat.completions.calls[0]
    assert result.name == "John Smith"
    _assert_generated_messages(call, expected_input="Extract contact")
    _assert_structured_response_format(call)


@pytest.mark.parametrize(
    ("generate_kwargs", "expected_payload_fields"),
    [
        (
            {"temperature": 0.2, "max_tokens": 64, "seed": 42},
            {"temperature": 0.2, "max_tokens": 64, "seed": 42},
        ),
        ({"top_p": 0.75}, {"top_p": 0.75}),
        (
            {"messages": [{"role": "user", "content": "override"}]},
            {"messages": [{"role": "user", "content": "Extract contact"}]},
        ),
    ],
)
def test_dottxt_generate_forwards_kwargs_and_preserves_core_fields(
    patch_sync_sdk: None,
    generate_kwargs: dict[str, Any],
    expected_payload_fields: dict[str, Any],
) -> None:
    """The native client should forward extras without allowing core-field override."""
    client = DotTxt(api_key="test-key")

    result = client.generate(
        model="openai/gpt-oss-20b",
        input="Extract contact",
        response_format=CONTACT_SCHEMA_JSON,
        **generate_kwargs,
    )

    sdk_client = _sync_sdk_client()
    call = sdk_client.chat.completions.calls[0]
    assert result["name"] == "John Smith"
    assert call["model"] == "openai/gpt-oss-20b"
    _assert_structured_response_format(call)
    _assert_generated_messages(call, expected_input="Extract contact")
    for key, expected_value in expected_payload_fields.items():
        assert call[key] == expected_value


def test_dottxt_generate_accepts_message_list(patch_sync_sdk: None) -> None:
    """The native client should forward a list of messages as-is."""
    client = DotTxt(api_key="test-key")
    messages = [
        {"role": "system", "content": "You extract contacts."},
        {"role": "user", "content": "Extract contact"},
    ]

    client.generate(
        model="openai/gpt-oss-20b",
        input=messages,
        response_format=Contact,
    )

    sdk_client = _sync_sdk_client()
    call = sdk_client.chat.completions.calls[0]
    assert call["messages"] == messages
    _assert_structured_response_format(call)


def test_dottxt_generate_rejects_non_string_completion_content(
    patch_sync_sdk: None,
) -> None:
    """The native client should fail clearly on non-string content."""
    client = DotTxt(api_key="test-key")
    sdk_client = _sync_sdk_client()
    sdk_client.chat.completions.result = FakeCompletion(
        model=MODEL_ID,
        choices=[FakeChoice(message=FakeMessage(None))],
    )

    with pytest.raises(TypeError, match="must be a string"):
        client.generate(
            model="openai/gpt-oss-20b",
            input="Extract contact",
            response_format=Contact,
        )


def test_dottxt_generate_rejects_missing_choices(patch_sync_sdk: None) -> None:
    """The native client should fail clearly when choices are missing."""
    client = DotTxt(api_key="test-key")
    sdk_client = _sync_sdk_client()
    sdk_client.chat.completions.result = FakeCompletion(model=MODEL_ID, choices=[])

    with pytest.raises(ValueError, match="did not include any choices"):
        client.generate(
            model="openai/gpt-oss-20b",
            input="Extract contact",
            response_format=Contact,
        )


def test_dottxt_generate_rejects_missing_message(patch_sync_sdk: None) -> None:
    """The native client should fail clearly when the message is missing."""
    client = DotTxt(api_key="test-key")
    sdk_client = _sync_sdk_client()
    sdk_client.chat.completions.result = SimpleNamespace(
        model=MODEL_ID,
        choices=[SimpleNamespace(message=None)],
    )

    with pytest.raises(ValueError, match="did not include a message"):
        client.generate(
            model="openai/gpt-oss-20b",
            input="Extract contact",
            response_format=Contact,
        )


def test_dottxt_generate_raises_structured_output_error_on_invalid_json(
    patch_sync_sdk: None,
) -> None:
    """The native client should expose partial raw output on parse failure."""
    client = DotTxt(api_key="test-key")
    sdk_client = _sync_sdk_client()
    sdk_client.chat.completions.result = FakeCompletion(
        model=MODEL_ID,
        choices=[
            FakeChoice(
                message=FakeMessage('{"name":"John'),
                finish_reason="length",
            )
        ],
    )

    with pytest.raises(InvalidOutputError) as error_info:
        client.generate(
            model="openai/gpt-oss-20b",
            input="Extract contact",
            response_format=CONTACT_SCHEMA_JSON,
        )

    error = error_info.value
    assert error.raw_output == '{"name":"John'
    assert error.finish_reason == "length"
    assert error.model == "openai/gpt-oss-20b"


def test_invalid_structured_output_error_omits_truncation_hint_without_length() -> None:
    """Error message should omit truncation hint when finish reason is not length."""
    error = InvalidOutputError(
        model="openai/gpt-oss-20b",
        raw_output="{}",
        finish_reason="stop",
        original_error=ValueError("bad parse"),
    )

    assert "finish_reason=length" not in str(error)


def test_completion_finish_reason_returns_none_when_choices_missing() -> None:
    """Finish reason helper should return None when choices are absent."""
    completion = SimpleNamespace(choices=[])

    assert _completion_finish_reason(completion) is None


def test_completion_finish_reason_returns_none_for_non_string_value() -> None:
    """Finish reason helper should return None for non-string values."""
    completion = SimpleNamespace(choices=[SimpleNamespace(finish_reason=1)])

    assert _completion_finish_reason(completion) is None


def test_dottxt_close_delegates_to_sdk(patch_sync_sdk: None) -> None:
    """Closing the native client should close the SDK client."""
    client = DotTxt(api_key="test-key")

    client.close()

    assert _sync_sdk_client().closed is True


@pytest.mark.asyncio
async def test_async_dottxt_models_namespace_delegates(
    patch_async_sdk: None,
) -> None:
    """The async client should expose the SDK models namespace."""
    client = AsyncDotTxt(api_key="test-key")

    response = await client.models.list()

    sdk_client = _async_sdk_client()
    assert sdk_client.models.calls == 1
    assert response.data[0].id == MODEL_ID


def test_async_dottxt_requires_api_key(
    patch_async_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async client should reject missing API keys."""
    monkeypatch.delenv("DOTTXT_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        AsyncDotTxt()


def test_async_dottxt_forwards_timeout_when_provided(
    patch_async_sdk: None,
) -> None:
    """The async client should pass timeout through to the SDK."""
    AsyncDotTxt(api_key="test-key", timeout=0)

    sdk_client = _async_sdk_client()
    assert sdk_client.init_kwargs["timeout"] == 0


@pytest.mark.asyncio
async def test_async_dottxt_chat_namespace_exposes_only_completions(
    patch_async_sdk: None,
) -> None:
    """The async chat namespace should only allow completions."""
    client = AsyncDotTxt(api_key="test-key")

    missing_attr = "responses"
    with pytest.raises(AttributeError):
        getattr(client.chat, missing_attr)


@pytest.mark.asyncio
async def test_async_dottxt_client_does_not_expose_responses(
    patch_async_sdk: None,
) -> None:
    """The async client should not expose the top-level responses namespace."""
    client = AsyncDotTxt(api_key="test-key")

    missing_attr = "responses"
    with pytest.raises(AttributeError):
        getattr(client, missing_attr)


@pytest.mark.asyncio
async def test_async_dottxt_generate(patch_async_sdk: None) -> None:
    """The async client should support structured generation."""
    client = AsyncDotTxt(api_key="test-key")

    result = await client.generate(
        model="openai/gpt-oss-20b",
        input="Extract contact",
        response_format=Contact,
    )

    sdk_client = _async_sdk_client()
    call = sdk_client.chat.completions.calls[0]
    assert result.name == "John Smith"
    _assert_structured_response_format(call)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generate_kwargs", "expected_payload_fields"),
    [
        (
            {"temperature": 0.2, "max_tokens": 64, "seed": 42},
            {"temperature": 0.2, "max_tokens": 64, "seed": 42},
        ),
        ({"top_p": 0.75}, {"top_p": 0.75}),
        (
            {"messages": [{"role": "user", "content": "override"}]},
            {"messages": [{"role": "user", "content": "Extract contact"}]},
        ),
    ],
)
async def test_async_dottxt_generate_forwards_kwargs_and_preserves_core_fields(
    patch_async_sdk: None,
    generate_kwargs: dict[str, Any],
    expected_payload_fields: dict[str, Any],
) -> None:
    """The async client should forward extras without allowing core-field override."""
    client = AsyncDotTxt(api_key="test-key")

    result = await client.generate(
        model="openai/gpt-oss-20b",
        input="Extract contact",
        response_format=CONTACT_SCHEMA_JSON,
        **generate_kwargs,
    )

    sdk_client = _async_sdk_client()
    call = sdk_client.chat.completions.calls[0]
    assert result["name"] == "John Smith"
    assert call["model"] == "openai/gpt-oss-20b"
    _assert_structured_response_format(call)
    _assert_generated_messages(call, expected_input="Extract contact")
    for key, expected_value in expected_payload_fields.items():
        assert call[key] == expected_value


@pytest.mark.asyncio
async def test_async_dottxt_generate_accepts_message_list(
    patch_async_sdk: None,
) -> None:
    """The async client should forward a list of messages as-is."""
    client = AsyncDotTxt(api_key="test-key")
    messages = [
        {"role": "system", "content": "You extract contacts."},
        {"role": "user", "content": "Extract contact"},
    ]

    await client.generate(
        model="openai/gpt-oss-20b",
        input=messages,
        response_format=Contact,
    )

    sdk_client = _async_sdk_client()
    call = sdk_client.chat.completions.calls[0]
    assert call["messages"] == messages
    _assert_structured_response_format(call)


@pytest.mark.asyncio
async def test_async_dottxt_generate_rejects_non_string_completion_content(
    patch_async_sdk: None,
) -> None:
    """The async client should fail clearly on non-string content."""
    client = AsyncDotTxt(api_key="test-key")
    sdk_client = _async_sdk_client()
    sdk_client.chat.completions.result = FakeCompletion(
        model=MODEL_ID,
        choices=[FakeChoice(message=FakeMessage(None))],
    )

    with pytest.raises(TypeError, match="must be a string"):
        await client.generate(
            model="openai/gpt-oss-20b",
            input="Extract contact",
            response_format=Contact,
        )


@pytest.mark.asyncio
async def test_async_dottxt_generate_raises_structured_output_error_on_invalid_json(
    patch_async_sdk: None,
) -> None:
    """The async client should expose partial raw output on parse failure."""
    client = AsyncDotTxt(api_key="test-key")
    sdk_client = _async_sdk_client()
    sdk_client.chat.completions.result = FakeCompletion(
        model=MODEL_ID,
        choices=[
            FakeChoice(
                message=FakeMessage('{"name":"John'),
                finish_reason="length",
            )
        ],
    )

    with pytest.raises(InvalidOutputError) as error_info:
        await client.generate(
            model="openai/gpt-oss-20b",
            input="Extract contact",
            response_format=CONTACT_SCHEMA_JSON,
        )

    error = error_info.value
    assert error.raw_output == '{"name":"John'
    assert error.finish_reason == "length"
    assert error.model == "openai/gpt-oss-20b"


@pytest.mark.asyncio
async def test_async_dottxt_close_delegates_to_sdk(patch_async_sdk: None) -> None:
    """Closing the async client should close the SDK client."""
    client = AsyncDotTxt(api_key="test-key")

    await client.close()

    assert _async_sdk_client().closed is True
