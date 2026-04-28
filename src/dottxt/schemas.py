"""Schema normalization helpers."""

from __future__ import annotations

import inspect
import json
from typing import Any, Protocol, runtime_checkable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import TypeAdapter

SchemaInput = Any
NormalizedSchema = Any


class InvalidSchemaError(Exception):
    """Raised when schema input cannot be normalized."""


@runtime_checkable
class _JsonSerializableSchema(Protocol):
    """Protocol for third-party schema builders exposing to_json()."""

    def to_json(self) -> str:
        """Return schema JSON as a string."""
        ...


def _validate_json_schema_object(schema: Any, *, source: str) -> dict[str, Any]:
    """Validate a JSON Schema object, with structural-tag passthrough support."""
    if not isinstance(schema, dict):
        raise InvalidSchemaError(f"{source} must contain a JSON object schema")
    if schema.get("type") == "structural-tag":
        return schema
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise InvalidSchemaError(
            f"{source} is not a valid JSON Schema: {exc.message}"
        ) from exc
    return schema


def normalize_schema(schema: SchemaInput) -> NormalizedSchema:
    """Normalize supported schema inputs to a JSON Schema object.

    Args:
        schema: One of:
            - JSON text that decodes to a JSON Schema object
            - a JSON Schema object (``dict``)
            - an object exposing ``to_json() -> str`` that returns a JSON
              Schema object
            - root ``{"type": "structural-tag", ...}`` objects are accepted
              without JSON Schema metaschema validation
            - any typed schema accepted by ``pydantic.TypeAdapter``
              (for example Pydantic models, TypedDict, dataclasses, Enum,
              Literal, Union/Optional, and typed containers like ``list[T]``)

    Returns:
        A JSON Schema object (`dict`) suitable for
        ``response_format["json_schema"]["schema"]``.

    Raises:
        InvalidSchemaError: If input JSON is invalid, not a JSON Schema object,
            or the type is unsupported.
    """
    if isinstance(schema, str):
        try:
            parsed = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise InvalidSchemaError("schema string must contain valid JSON") from exc
        return _validate_json_schema_object(parsed, source="schema string")
    if isinstance(schema, dict):
        return _validate_json_schema_object(schema, source="schema object")
    if isinstance(schema, list):
        raise InvalidSchemaError(
            "raw list schema inputs are not supported; "
            "pass a JSON schema object or typed list (list[T])"
        )
    if isinstance(schema, _JsonSerializableSchema):
        try:
            parsed = json.loads(schema.to_json())
        except json.JSONDecodeError as exc:
            raise InvalidSchemaError("schema.to_json() must return valid JSON") from exc
        return _validate_json_schema_object(parsed, source="schema.to_json()")
    if inspect.isroutine(schema):
        raise InvalidSchemaError(
            "callable schema inputs are not supported; pass a type or JSON schema"
        )
    try:
        return TypeAdapter(schema).json_schema()
    except Exception as exc:
        raise InvalidSchemaError(
            "schema must be valid JSON or a type supported by pydantic TypeAdapter"
        ) from exc


def build_response_format(
    schema: SchemaInput,
    *,
    name: str = "response",
) -> dict[str, Any]:
    """Create the OpenAI-compatible response_format payload.

    The dottxt endpoint does not currently use ``name`` for behavior. It is
    included only for compatibility with the OpenAI-compatible request shape.

    OpenAI treats ``strict`` as optional for json_schema response formats.
    On dottxt, strict-mode does not currently change behavior. See
    https://docs.dottxt.ai/json-schema/overview for endpoint-specific details.

    Args:
        schema: Schema input to normalize.
        name: Schema name sent to the API.

    Returns:
        A response_format payload for chat completions.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": normalize_schema(schema),
        },
    }
