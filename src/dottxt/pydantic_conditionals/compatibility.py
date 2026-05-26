"""Compatibility transforms for JSON Schema conditional keywords.

Converts ``dependentRequired`` and ``dependentSchemas`` into
``if``/``then``/``else`` constructs.
"""

from __future__ import annotations

from typing import Any


def _convert_dependent_required_to_if_then(
    dependent_required: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Convert ``dependentRequired`` entries into if/then rules.

    Each trigger field produces one rule:
    ``{"if": {"required": [trigger]}, "then": {"required": [dependents...]}}``
    """
    rules: list[dict[str, Any]] = []
    for trigger, dependents in dependent_required.items():
        rules.append(
            {
                "if": {"required": [trigger]},
                "then": {"required": dependents},
            }
        )
    return rules


def _convert_dependent_schemas_to_if_then(
    dependent_schemas: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert ``dependentSchemas`` entries into if/then rules.

    Each trigger field produces one rule:
    ``{"if": {"required": [trigger]}, "then": sub_schema}``
    """
    rules: list[dict[str, Any]] = []
    for trigger, sub_schema in dependent_schemas.items():
        rules.append(
            {
                "if": {"required": [trigger]},
                "then": sub_schema,
            }
        )
    return rules


def _apply_recursive(value: Any) -> Any:
    """Recursively apply if-then-else compatibility mode to any nested schema."""
    if isinstance(value, dict):
        return apply_compatibility_mode(value)
    if isinstance(value, list):
        return [_apply_recursive(item) for item in value]
    return value


def apply_compatibility_mode(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a schema to eliminate dependentRequired and dependentSchemas.

    Converts these keywords into equivalent if/then/else constructs.
    """

    schema = dict(schema)
    collected_rules: list[dict[str, Any]] = []

    dep_req = schema.pop("dependentRequired", None)
    if dep_req:
        collected_rules.extend(_convert_dependent_required_to_if_then(dep_req))

    dep_schemas = schema.pop("dependentSchemas", None)
    if dep_schemas:
        collected_rules.extend(_convert_dependent_schemas_to_if_then(dep_schemas))

    for key, value in schema.items():
        schema[key] = _apply_recursive(value)

    if collected_rules:
        schema.setdefault("allOf", [])
        schema["allOf"] = schema["allOf"] + collected_rules

    return schema
