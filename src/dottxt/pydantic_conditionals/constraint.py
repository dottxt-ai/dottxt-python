"""Constraint building utilities for JSON Schema generation.

Converts field constraints (plain values, ``Field(...)`` objects, and
``Constraint`` markers) into their JSON Schema representations.
"""

from __future__ import annotations

from typing import Any

from pydantic._internal._known_annotated_metadata import collect_known_metadata
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema


class Constraint:
    """Marker for a field constraint with optional value and required flag."""

    def __init__(
        self,
        value: FieldInfo | Any = None,
        has_value: bool = False,
        required: bool = False,
    ) -> None:
        self.value: FieldInfo | Any = value
        self.has_value: bool = has_value
        self.required: bool = required


def _build_constraint_value(val: Any) -> dict[str, Any]:
    """Convert a constraint value into its JSON Schema representation.

    If *val* is a Pydantic ``FieldInfo`` (created via ``Field(...)``), the
    known validation metadata (e.g. ``pattern``, ``ge``, ``le``) is extracted
    and mapped to the corresponding JSON Schema keywords.

    For plain values (strings, numbers, etc.), a ``{"const": val}`` schema is
    returned, which requires the field to exactly match the given value.
    """
    if isinstance(val, FieldInfo):
        core_schema, _ = collect_known_metadata(val.metadata)
        json_schema: dict[str, Any] = {}

        # ValidationsMapping maps Pydantic core-schema validation keys to
        # their JSON Schema equivalents, grouped by type category.
        mapping = GenerateJsonSchema.ValidationsMapping.__dict__
        for key in mapping:
            if key.startswith("_"):
                continue

            for core_key, json_schema_key in mapping[key].items():
                if core_key in core_schema:
                    json_schema[json_schema_key] = core_schema[core_key]

        return json_schema

    return {"const": val}


def _build_properties(constraints: dict[str, Any]) -> dict[str, Any]:
    """Build a ``{"properties": {...}}`` dict from field-constraint pairs."""
    if not constraints:
        return {}

    return {
        "type": "object",
        "properties": {k: _build_constraint_value(v) for k, v in constraints.items()},
        "additionalProperties": True,
    }


def _build_required(required: list[str]) -> dict[str, Any]:
    """Build a ``{"required": [...]}`` dict from field names."""
    if not required:
        return {}

    return {"type": "object", "required": required, "additionalProperties": True}


def build_constraints(constraints: dict[str, Constraint]) -> dict[str, Any]:
    """Convert a dict of ``Constraint`` objects into a JSON Schema fragment.

    Produces a dict with ``"properties"`` and/or ``"required"`` keys based on
    which constraints have values or the required flag set.
    """
    property_constraints: dict[str, Any] = {}
    required_constraints: list[str] = []

    for key, constraint in constraints.items():
        if constraint.required:
            required_constraints.append(key)

        if constraint.has_value:
            property_constraints[key] = constraint.value

    return {
        **_build_properties(property_constraints),
        **_build_required(required_constraints),
    }


def compute_constraints(
    existing_constraints: dict[str, Constraint],
    new_constraints: dict[str, Constraint | Any],
) -> dict[str, Constraint]:
    """Merge *new_constraints* into *existing_constraints*.

    Plain values are wrapped in ``Constraint(value=..., has_value=True)``.
    When a key already exists, the required flag is OR-ed and the value is
    overwritten if the new constraint carries one.
    """
    result = existing_constraints
    for key, constraint in new_constraints.items():
        if constraint is None:
            # At this point, if a constraint is None it can either be absent or null
            # None in `where`s is handeled elsewhere
            constraint = Constraint(value=None, has_value=True)
        elif not isinstance(constraint, Constraint):
            constraint = Constraint(value=constraint, has_value=True, required=True)

        if key in result:
            result[key].required = result[key].required or constraint.required

            if constraint.has_value:
                result[key].value = constraint.value
                result[key].has_value = True
        else:
            result[key] = constraint

    return result
