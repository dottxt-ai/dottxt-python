from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI as AsyncOpenAISDK
from openai import OpenAI as OpenAISDK
from pydantic import BaseModel, ValidationError

from dottxt.schemas import SchemaInput, build_response_format

DEFAULT_BASE_URL = "https://api.dottxt.ai/v1"


class InvalidOutputError(ValueError):
    """Raised when a completion cannot be parsed into the requested structure."""

    def __init__(
        self,
        *,
        model: str,
        raw_output: str,
        finish_reason: str | None,
        original_error: Exception,
    ) -> None:
        """Create an invalid-output error with parse context.

        Args:
            model: Model identifier used for generation.
            raw_output: Raw completion text returned by the model.
            finish_reason: Completion finish reason when available.
            original_error: Original parse/validation exception.
        """
        message = f"Failed to parse structured output from model '{model}'."
        if finish_reason == "length":
            message += " Output may be truncated (finish_reason=length)."
        super().__init__(message)
        self.model = model
        self.raw_output = raw_output
        self.finish_reason = finish_reason
        self.original_error = original_error


class _CompletionsNamespace:
    """Restricted chat namespace exposing only completions."""

    def __init__(self, completions: Any) -> None:
        """Create a namespace that only exposes completions.

        Args:
            completions: The SDK completions namespace.
        """
        self.completions: Any = completions


class DotTxt:
    """SDK client for structured generation on the dottxt API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Create a dottxt client.

        Args:
            api_key: API key override.
            base_url: Base URL override.
            **client_kwargs: Additional options forwarded to ``openai.OpenAI``.
                See OpenAI base-client parameters:
                https://github.com/openai/openai-python/blob/main/src/openai/_base_client.py
        """
        resolved_api_key = api_key or os.getenv("DOTTXT_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "An API key is required. Set DOTTXT_API_KEY or pass api_key."
            )
        resolved_base_url = base_url or os.getenv("DOTTXT_BASE_URL") or DEFAULT_BASE_URL
        self._client = OpenAISDK(
            api_key=resolved_api_key,
            base_url=resolved_base_url.rstrip("/"),
            **client_kwargs,
        )
        self.chat = _CompletionsNamespace(self._client.chat.completions)
        self.models: Any = self._client.models

    def generate(
        self,
        *,
        model: str,
        response_format: SchemaInput,
        input: str | list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        **extra: Any,
    ) -> Any:
        """Generate structured output and return parsed structured data.

        Args:
            model: Model identifier.
            response_format: Schema input accepted by ``build_response_format``.
                This includes JSON Schema strings/objects, objects exposing
                ``to_json() -> str``, and typed inputs supported by Pydantic
                ``TypeAdapter``.
            input: A prompt string (sent as a single user message) or a list
                of chat messages (dicts with `role` and `content` keys).
            temperature: Optional temperature value.
            max_tokens: Optional max output tokens.
            seed: Optional deterministic seed.
            **extra: Additional chat-completions parameters.
                See:
                https://github.com/openai/openai-python/blob/main/src/openai/resources/chat/completions/completions.py#L247

        Returns:
            A parsed Pydantic model or decoded JSON object.
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        payload: dict[str, Any] = {
            **extra,
            "model": model,
            "messages": input,
            "response_format": build_response_format(response_format),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        completion = self.chat.completions.create(**payload)
        completion_text = _completion_text(completion)
        try:
            return _parse_output(completion_text, response_format)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise InvalidOutputError(
                model=model,
                raw_output=completion_text,
                finish_reason=_completion_finish_reason(completion),
                original_error=exc,
            ) from exc

    def close(self) -> None:
        """Close the underlying SDK client."""
        self._client.close()


class AsyncDotTxt:
    """Async SDK client for structured generation on the dottxt API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Create an async dottxt client.

        Args:
            api_key: API key override.
            base_url: Base URL override.
            **client_kwargs: Additional options forwarded to ``openai.AsyncOpenAI``.
                See OpenAI base-client parameters:
                https://github.com/openai/openai-python/blob/main/src/openai/_base_client.py
        """
        resolved_api_key = api_key or os.getenv("DOTTXT_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "An API key is required. Set DOTTXT_API_KEY or pass api_key."
            )
        resolved_base_url = base_url or os.getenv("DOTTXT_BASE_URL") or DEFAULT_BASE_URL
        self._client = AsyncOpenAISDK(
            api_key=resolved_api_key,
            base_url=resolved_base_url.rstrip("/"),
            **client_kwargs,
        )
        self.chat = _CompletionsNamespace(self._client.chat.completions)
        self.models: Any = self._client.models

    async def generate(
        self,
        *,
        model: str,
        response_format: SchemaInput,
        input: str | list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        **extra: Any,
    ) -> Any:
        """Generate structured output and return parsed structured data.

        Args:
            model: Model identifier.
            response_format: Schema input accepted by ``build_response_format``.
                This includes JSON Schema strings/objects, objects exposing
                ``to_json() -> str``, and typed inputs supported by Pydantic
                ``TypeAdapter``.
            input: A prompt string (sent as a single user message) or a list
                of chat messages (dicts with `role` and `content` keys).
            temperature: Optional temperature value.
            max_tokens: Optional max output tokens.
            seed: Optional deterministic seed.
            **extra: Additional chat-completions parameters.
                See:
                https://github.com/openai/openai-python/blob/main/src/openai/resources/chat/completions/completions.py#L247

        Returns:
            A parsed Pydantic model or decoded JSON object.
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        payload: dict[str, Any] = {
            **extra,
            "model": model,
            "messages": input,
            "response_format": build_response_format(response_format),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        completion = await self.chat.completions.create(**payload)
        completion_text = _completion_text(completion)
        try:
            return _parse_output(completion_text, response_format)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise InvalidOutputError(
                model=model,
                raw_output=completion_text,
                finish_reason=_completion_finish_reason(completion),
                original_error=exc,
            ) from exc

    async def close(self) -> None:
        """Close the underlying SDK client."""
        await self._client.close()


def _completion_text(completion: Any) -> str:
    """Extract the first completion message text.

    Args:
        completion: OpenAI SDK chat completion response.

    Returns:
        The first assistant message content.

    Raises:
        ValueError: If the completion response does not contain a message.
        TypeError: If the message content is not a string.
    """
    choices = getattr(completion, "choices", [])
    if not choices:
        raise ValueError("chat completion did not include any choices")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("chat completion did not include a message")

    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise TypeError(
            "chat completion message content must be a string for generate()"
        )
    return content


def _completion_finish_reason(completion: Any) -> str | None:
    """Extract the first completion finish reason when available."""
    choices = getattr(completion, "choices", [])
    if not choices:
        return None
    finish_reason = getattr(choices[0], "finish_reason", None)
    if isinstance(finish_reason, str):
        return finish_reason
    return None


def _parse_output(payload_text: str, response_format: SchemaInput) -> Any:
    """Parse completion text using either Pydantic or raw JSON.

    Args:
        payload_text: Raw completion message text.
        response_format: Original schema input used for generation.

    Returns:
        A validated Pydantic model or decoded JSON object.
    """
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return response_format.model_validate_json(payload_text)
    return json.loads(payload_text)
