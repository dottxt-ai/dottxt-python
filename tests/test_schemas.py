"""Tests for schema normalization helpers."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel

from dottxt.schemas import InvalidSchemaError, build_response_format, normalize_schema


class Contact(BaseModel):
    """Example Pydantic schema."""

    name: str


class Severity(Enum):
    """Example enum schema."""

    LOW = "low"
    HIGH = "high"


def test_normalize_schema_accepts_pydantic_model() -> None:
    """Pydantic model classes should be converted to JSON Schema."""
    schema = normalize_schema(Contact)
    assert schema["type"] == "object"
    assert "name" in schema["properties"]


def test_normalize_schema_accepts_json_string() -> None:
    """JSON schema strings should be parsed into dictionaries."""
    schema = normalize_schema('{"type": "object", "properties": {}}')
    assert schema["type"] == "object"


def test_normalize_schema_accepts_json_object() -> None:
    """Dictionary schemas should pass through unchanged."""
    source = {"type": "object", "properties": {"name": {"type": "string"}}}
    schema = normalize_schema(source)
    assert schema == source


def test_normalize_schema_rejects_json_array_instance() -> None:
    """Raw list instances should not be accepted as schema inputs."""
    with pytest.raises(InvalidSchemaError, match="raw list schema inputs"):
        normalize_schema([{"type": "string"}, {"type": "number"}])  # type: ignore[arg-type]


def test_normalize_schema_accepts_enum_type() -> None:
    """Enum classes should normalize to enum JSON Schema."""
    schema = normalize_schema(Severity)
    assert schema["type"] == "string"
    assert schema["enum"] == ["low", "high"]


def test_normalize_schema_accepts_literal_type() -> None:
    """Literal types should normalize to enum JSON Schema."""
    literal_schema = Literal["low", "high"]
    schema = normalize_schema(literal_schema)  # type: ignore[arg-type]
    assert schema["type"] == "string"
    assert schema["enum"] == ["low", "high"]


def test_normalize_schema_accepts_union_type() -> None:
    """Union types should normalize via anyOf schemas."""
    schema = normalize_schema(int | str)  # type: ignore[arg-type]
    assert schema["anyOf"] == [{"type": "integer"}, {"type": "string"}]


def test_normalize_schema_accepts_optional_type() -> None:
    """Optional types should include the null branch."""
    schema = normalize_schema(int | None)  # type: ignore[arg-type]
    assert schema["anyOf"] == [{"type": "integer"}, {"type": "null"}]


def test_normalize_schema_accepts_typed_list_type() -> None:
    """Typed list aliases should normalize to array schemas."""
    schema = normalize_schema(list[int])  # type: ignore[arg-type]
    assert schema["type"] == "array"
    assert schema["items"] == {"type": "integer"}


def test_normalize_schema_accepts_typed_dict_type() -> None:
    """Typed dict aliases should normalize additionalProperties schema."""
    schema = normalize_schema(dict[str, int])  # type: ignore[arg-type]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] == {"type": "integer"}


def test_normalize_schema_accepts_tuple_type() -> None:
    """Typed tuples should normalize to prefixItems schema."""
    schema = normalize_schema(tuple[int, str])  # type: ignore[arg-type]
    assert schema["type"] == "array"
    assert schema["prefixItems"] == [{"type": "integer"}, {"type": "string"}]
    assert schema["minItems"] == 2
    assert schema["maxItems"] == 2


def test_normalize_schema_accepts_to_json_schema_builder() -> None:
    """Objects exposing to_json() should be supported."""

    class _SchemaBuilder:
        def to_json(self) -> str:
            return '{"type":"object","properties":{"ok":{"type":"boolean"}}}'

    schema = normalize_schema(_SchemaBuilder())  # type: ignore[arg-type]
    assert schema["type"] == "object"
    assert schema["properties"]["ok"]["type"] == "boolean"


def test_normalize_schema_rejects_invalid_to_json_payload() -> None:
    """to_json() values must contain valid JSON."""

    class _BrokenSchemaBuilder:
        def to_json(self) -> str:
            return "{"

    with pytest.raises(InvalidSchemaError, match="to_json\\(\\)"):
        normalize_schema(_BrokenSchemaBuilder())  # type: ignore[arg-type]


def test_normalize_schema_rejects_invalid_json() -> None:
    """Invalid JSON strings should raise a schema error."""
    with pytest.raises(InvalidSchemaError):
        normalize_schema("{")


def test_normalize_schema_rejects_json_string_non_object_schema() -> None:
    """JSON schema strings must decode to JSON object schemas."""
    with pytest.raises(InvalidSchemaError, match="JSON object schema"):
        normalize_schema("[]")


def test_normalize_schema_accepts_json_object_with_custom_keywords() -> None:
    """Dictionary schemas may include custom keywords and pass through."""
    source = {"a": 1}
    schema = normalize_schema(source)
    assert schema == source


def test_normalize_schema_rejects_invalid_json_schema_object_shape() -> None:
    """Dictionary schema inputs should fail metaschema validation when malformed."""
    with pytest.raises(InvalidSchemaError, match="valid JSON Schema"):
        normalize_schema({"type": 1})  # type: ignore[arg-type]


def test_normalize_schema_rejects_invalid_json_schema_string_shape() -> None:
    """String schema inputs should fail metaschema validation when malformed."""
    with pytest.raises(InvalidSchemaError, match="valid JSON Schema"):
        normalize_schema('{"type": 1}')


def test_normalize_schema_accepts_structural_tag_schema_object() -> None:
    """Structural-tag root schemas should bypass metaschema validation."""
    source = {
        "type": "structural-tag",
        "tag": "incident",
        "variants": [{"name": "ticket"}],
    }
    schema = normalize_schema(source)
    assert schema == source


def test_normalize_schema_accepts_structural_tag_schema_string() -> None:
    """Structural-tag root schema strings should bypass metaschema validation."""
    schema = normalize_schema(
        '{"type":"structural-tag","tag":"incident","variants":[{"name":"ticket"}]}'
    )
    assert schema["type"] == "structural-tag"
    assert schema["tag"] == "incident"


def test_normalize_schema_rejects_unsupported_type() -> None:
    """Unsupported schema inputs should raise a schema error."""
    with pytest.raises(InvalidSchemaError, match="TypeAdapter"):
        normalize_schema(object())  # type: ignore[arg-type]


def test_normalize_schema_rejects_callable_values() -> None:
    """Callable non-type schema values should raise a targeted error."""

    def factory() -> None:
        return None

    with pytest.raises(InvalidSchemaError, match="callable schema inputs"):
        normalize_schema(factory)  # type: ignore[arg-type]


def test_build_response_format_wraps_schema() -> None:
    """Structured output payloads should use the expected shape."""
    response_format = build_response_format(Contact, name="contact")
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "contact"
