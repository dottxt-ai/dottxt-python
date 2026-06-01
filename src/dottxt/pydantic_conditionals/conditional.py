"""Conditional JSON Schema generation for Pydantic models.

This module provides tools to add JSON Schema conditional keywords
(``if``/``then``/``else``) and ``dependentSchemas`` to Pydantic model
schemas using a fluent API.  These conditional rules are expressed in
the generated JSON Schema output, enabling conditional validation that
goes beyond standard Pydantic validators.

See https://json-schema.org/understanding-json-schema/reference/conditionals
for details on JSON Schema conditional keywords.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel
from pydantic.annotated_handlers import GetJsonSchemaHandler

from .constraint import Constraint, build_constraints, compute_constraints

CONDITIONALS_REF_MARKER = "$$ref"


def _update_conditionals_ref_marker(
    schema: dict[str, Any], from_str: str, to_str: str
) -> dict[str, Any]:
    """Update the original schema with new ref markers temporarily"""

    if isinstance(schema, list):
        for i in range(len(schema)):
            schema[i] = _update_conditionals_ref_marker(schema[i], from_str, to_str)

    elif isinstance(schema, dict):
        ref = schema.pop(from_str, None)
        if ref:
            schema[to_str] = ref

        for k, v in schema.items():
            schema[k] = _update_conditionals_ref_marker(v, from_str, to_str)
    else:
        assert (
            schema is None
            or isinstance(schema, bool)
            or isinstance(schema, int)
            or isinstance(schema, float)
            or isinstance(schema, str)
        )

    return schema


def _insert_conditionals_ref_marker(schema: dict[str, Any]) -> dict[str, Any]:
    return _update_conditionals_ref_marker(schema, "$ref", CONDITIONALS_REF_MARKER)


def _remove_conditionals_ref_marker(schema: dict[str, Any]) -> dict[str, Any]:
    return _update_conditionals_ref_marker(schema, CONDITIONALS_REF_MARKER, "$ref")


def _normalize_require_fields(field: str | tuple[str] | list[str]) -> list[str]:
    """Validate and normalize a field-or-list argument into a list of strings."""
    if isinstance(field, str):
        return [field]
    if isinstance(field, tuple):
        return list(field)
    if isinstance(field, list):
        return field
    raise ValueError("Parameter must be a list or str.")


class QueryBuilder:
    """Builds a single conditional JSON Schema rule using a fluent interface."""

    def __init__(
        self, conditions: dict[str, Any] | None, *, schema: dict[str, Any] | None = None
    ) -> None:

        if conditions is None:
            conditions = {}
        for key, con in conditions.items():
            if con is None:
                # If you have a None in an `where` you are testing if the type is `null`
                conditions[key] = Constraint(value=None, has_value=True, required=True)
        self.conditions: list[dict[str, Constraint]] = (
            [compute_constraints({}, conditions)] if conditions else []
        )
        self.then_constraints: dict[str, Any] | None = None
        self.else_constraints: dict[str, Any] | None = None
        self.if_schema: dict[str, Any] | None = schema
        self.then_schema: dict[str, Any] | None = None
        self.else_schema: dict[str, Any] | None = None

        if self.conditions and self.if_schema is not None:
            raise ValueError(
                "You cannot provide a schema and a list of conditions to a where."
            )

    @classmethod
    def from_schema(cls, schema: type[BaseModel]) -> QueryBuilder:
        schema_dict = schema.model_json_schema()
        schema_dict["additionalProperties"] = True
        return cls(conditions=None, schema=schema_dict)

    def __and__(self, other: QueryBuilder | DependentSchemaBuilder) -> QueryBuilder:
        """Combine this condition with another using logical AND.

        The resulting condition will match only if both this condition and the
        other condition are met.

        Args:
            other: Another QueryBuilder or DependentSchemaBuilder instance to
                   combine with.

        Returns:
            A QueryBuilder instance representing the combined condition.
        """
        if not isinstance(other, (QueryBuilder, DependentSchemaBuilder)):
            raise ValueError("Can only combine with another QueryBuilder.")

        if isinstance(other, DependentSchemaBuilder):
            other = other._convert_to_if_then_else()

        if len(self.conditions) > 1 or len(other.conditions) > 1:
            raise ValueError("Cannot mix AND and OR conditions.")

        if other.then_constraints or other.else_constraints:
            raise ValueError(
                "Cannot combine with a QueryBuilder that has constraints "
                "or requirements."
            )

        if (self.if_schema is not None and other.conditions) or (
            isinstance(other, QueryBuilder)
            and other.if_schema is not None
            and self.conditions
        ):
            raise ValueError(
                "Cannot combine where clauses specified with schemas with "
                "ones specified with conditions."
            )

        conditions = self.conditions[0] if len(self.conditions) == 1 else {}
        if other.conditions:
            conditions.update(other.conditions[0])
        if conditions:
            self.conditions = [conditions]

        if self.if_schema is not None and other.if_schema is not None:
            self.if_schema = {"allOf": [self.if_schema, other.if_schema]}

        return self

    def __or__(self, other: QueryBuilder | DependentSchemaBuilder) -> QueryBuilder:
        """Combine this condition with another using logical OR.

        The resulting condition will match if either this condition or the other
        condition is met.

        Args:
            other: Another QueryBuilder or DependentSchemaBuilder instance to
                   combine with.

        Returns:
            A QueryBuilder instance representing the combined condition.
        """

        if not isinstance(other, (QueryBuilder, DependentSchemaBuilder)):
            raise ValueError("Can only combine with another QueryBuilder.")

        if (
            isinstance(other, QueryBuilder)
            and self.if_schema is not None
            and other.conditions
        ) or (
            isinstance(other, QueryBuilder)
            and other.if_schema is not None
            and self.conditions
        ):
            raise ValueError(
                "Cannot combine where clauses specified with schemas with "
                "ones specified with conditions."
            )

        if isinstance(other, DependentSchemaBuilder):
            other = other._convert_to_if_then_else()

        if other.then_constraints or other.else_constraints:
            raise ValueError(
                "Cannot combine with a QueryBuilder that has constraints or "
                "requirements."
            )

        self.conditions += other.conditions
        if self.if_schema is not None and other.if_schema is not None:
            self.if_schema = {"anyOf": [self.if_schema, other.if_schema]}

        return self

    def require(self, field: str | list[str]) -> Self:
        """Mark one or more fields as required when the condition is met.

        Args:
            field: A field name or list of field names to require.

        Returns:
            This builder instance, for method chaining.
        """
        new_constraints = {}
        for field_norm in _normalize_require_fields(field):
            new_constraints[field_norm] = Constraint(required=True)
        return self.constrain(**new_constraints)

    def then_apply(self, schema: type[BaseModel]) -> Self:
        """Mixes in this schema when the condition is met (``then`` clause).

        This does not combine with ``constrain`` or other ``then_apply`` methods.
        The last ``then_apply`` will take precedence.

        Returns:
            This builder instance, for method chaining.
        """
        self.then_schema = schema.model_json_schema()
        self.then_schema["additionalProperties"] = True
        return self

    def then_apply_only(self, schema: type[BaseModel]) -> Self:
        """Mixes in this schema when the condition is met (``then`` clause).
        ``additionalProperties`` is set to ``False``.

        This does not combine with ``constrain`` or other ``then_apply`` methods.
        The last ``then_apply`` will take precedence.

        Returns:
            This builder instance, for method chaining.
        """
        self.then_schema = schema.model_json_schema()
        self.then_schema["additionalProperties"] = False
        return self

    def else_apply(self, schema: type[BaseModel]) -> Self:
        """Mixes in this schema when the condition is *not* met (``else`` clause).

        This does not combine with ``otherwise`` or other ``else_apply`` methods.
        The last ``else_apply`` will take precedence.

        Returns:
            This builder instance, for method chaining.
        """
        self.else_schema = schema.model_json_schema()
        self.else_schema["additionalProperties"] = True
        return self

    def else_apply_only(self, schema: type[BaseModel]) -> Self:
        """Mixes in this schema when the condition is *not* met (``else`` clause).
        ``additionalProperties`` is set to ``False``.

        This does not combine with ``otherwise`` or other ``else_apply`` methods.
        The last ``else_apply`` will take precedence.

        Returns:
            This builder instance, for method chaining.
        """
        self.else_schema = schema.model_json_schema()
        self.else_schema["additionalProperties"] = False
        return self

    def constrain(self, **kwargs: Any) -> Self:
        """Add constraints to fields when the condition is met (``then`` clause).

        Each keyword argument maps a field name to a constraint value.
        Constraint values can be plain values (for exact-match ``const``
        checks) or ``Field(...)`` objects (for rich validation constraints
        like ``pattern``, ``ge``, ``le``, etc.).

        Returns:
            This builder instance, for method chaining.
        """
        self.then_constraints = compute_constraints(self.then_constraints or {}, kwargs)
        return self

    def otherwise(self, **kwargs: Any) -> Self:
        """Add constraints for when the condition is *not* met (``else`` clause).

        Accepts the same arguments as :meth:`constrain`.

        Returns:
            This builder instance, for method chaining.
        """
        self.else_constraints = compute_constraints(self.else_constraints or {}, kwargs)
        return self

    def _build(self) -> dict[str, Any] | None:
        """Compile this builder into a JSON Schema conditional fragment.

        Returns a dict containing ``if``, and optionally ``then`` and/or
        ``else`` keys, ready to be merged into a JSON Schema object.
        Returns ``None`` if no constraints or requirements were specified.
        """
        result: dict[str, Any] = {}

        if self.then_schema:
            result["then"] = self.then_schema
        elif self.then_constraints:
            result["then"] = build_constraints(self.then_constraints)

        if self.else_schema:
            result["else"] = self.else_schema
        elif self.else_constraints:
            result["else"] = build_constraints(self.else_constraints)

        if not result:
            return None

        if self.if_schema:
            result["if"] = self.if_schema
            return result

        if_results = [build_constraints(condition) for condition in self.conditions]

        if len(if_results) == 1:
            result["if"] = if_results[0]
        else:
            result["if"] = {"anyOf": if_results}

        return result


class DependentSchemaBuilder:
    """Builds a ``dependentSchemas`` entry for a single trigger field."""

    def __init__(self, trigger_field: str) -> None:
        self.trigger_field = trigger_field
        self.then_constraints: dict[str, Any] | None = None
        self.then_schema: dict[str, Any] | None = None

    def require(self, field: str | list[str]) -> Self:
        """Mark one or more fields as required when the trigger field is present."""
        new_constraints = {}
        for field_norm in _normalize_require_fields(field):
            new_constraints[field_norm] = Constraint(required=True)
        return self.constrain(**new_constraints)

    def then_apply(self, schema: type[BaseModel]) -> Self:
        """Mixes in this schema when the condition is met (``then`` clause).

        This does not combine with ``constrain`` or other ``then_apply`` methods.
        The last ``then_apply`` will take precedence.

        Returns:
            This builder instance, for method chaining.
        """
        self.then_schema = schema.model_json_schema()
        self.then_schema["additionalProperties"] = True
        return self

    def else_apply(self, schema: type[BaseModel]) -> QueryBuilder:
        """Mixes in this schema when the condition is *not* met (``else`` clause).

        This does not combine with ``otherwise`` or other ``else_apply`` methods.
        The last ``else_apply`` will take precedence.

        Returns:
            QueryBuilder instance, for method chaining.
        """
        qb = self._convert_to_if_then_else()
        return qb.else_apply(schema)

    def constrain(self, **kwargs: Any) -> Self:
        """Add property constraints when the trigger field is present."""
        self.then_constraints = compute_constraints(self.then_constraints or {}, kwargs)
        return self

    def otherwise(self, **kwargs: Any) -> QueryBuilder:
        """Add constraints for when the condition is *not* met (``else`` clause).

        Accepts the same arguments as :meth:`constrain`.

        Returns:
            QueryBuilder instance, for method chaining.
        """
        qb = self._convert_to_if_then_else()
        return qb.otherwise(**kwargs)

    def __and__(self, other: QueryBuilder | DependentSchemaBuilder) -> QueryBuilder:
        """Combine this condition with another using logical AND.

        The resulting condition will match only if both this condition and the
        other condition are met.

        Args:
            other: Another QueryBuilder or DependentSchemaBuilder instance to
                   combine with.

        Returns:
            A QueryBuilder instance representing the combined condition.
        """
        qb = self._convert_to_if_then_else()
        return qb & other

    def __or__(self, other: QueryBuilder | DependentSchemaBuilder) -> QueryBuilder:
        """Combine this condition with another using logical OR.

        The resulting condition will match if either this condition or the other
        condition is met.

        Args:
            other: Another QueryBuilder or DependentSchemaBuilder instance to
                   combine with.

        Returns:
            A QueryBuilder instance representing the combined condition.
        """
        qb = self._convert_to_if_then_else()
        return qb | other

    def _convert_to_if_then_else(self) -> QueryBuilder:
        qb = QueryBuilder(None)
        qb.conditions = [{self.trigger_field: Constraint(required=True)}]
        qb.then_constraints = self.then_constraints
        qb.then_schema = self.then_schema

        return qb

    def _build(self) -> tuple[str, dict[str, Any]] | None:
        """Return ``(trigger_field, sub_schema)`` or ``None``."""

        sub_schema = None
        if self.then_schema:
            sub_schema = self.then_schema
        elif self.then_constraints:
            sub_schema = build_constraints(self.then_constraints)

        if not sub_schema:
            return None

        return self.trigger_field, sub_schema


class RequiredWith:
    """Field annotation expressing a ``dependentRequired`` constraint.

    When attached to a field via ``Annotated``, it declares that whenever any
    of the specified *trigger* fields are present, the annotated field must
    also be present.

    Args:
        *fields: One or more field names whose presence requires this field.

    Example::

        class FileOperation(ConditionalModel, BaseModel):
            content: str | None = None
            create_parents: Annotated[bool | None, RequiredWith("content")] = None

        # Produces: {"dependentRequired": {"content": ["create_parents"]}}
    """

    def __init__(self, *fields: str) -> None:
        if not fields:
            raise ValueError("RequiredWith requires at least one field name.")
        self.fields = fields


class ConditionalModel:
    """Mixin that adds conditional rules to a Pydantic model's JSON Schema.

    Subclass this alongside ``BaseModel`` and define a ``model_conditions``
    class variable containing one or more :class:`QueryBuilder`,
    or :class:`DependentSchemaBuilder` rules::

        class Address(ConditionalModel, BaseModel):
            street: str
            country: str
            postal_code: str

            model_conditions: ClassVar = (
                when(country="USA").require("postal_code"),
            )

    When ``model_json_schema()`` is called, the conditional rules are injected
    into the generated schema using JSON Schema's ``if``/``then``/``else``
    keywords. Multiple conditions are combined using ``allOf``.
    """

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        source_type: Any,
        handler: GetJsonSchemaHandler,
    ) -> dict[str, Any]:
        result_schema = handler(source_type)
        # --- dependentRequired from RequiredWith annotations ---
        if hasattr(cls, "model_fields"):
            dependent_required: dict[str, list[str]] = {}
            for field_name, field_info in cls.model_fields.items():
                for metadata in field_info.metadata:
                    if not isinstance(metadata, RequiredWith):
                        continue

                    for field in metadata.fields:
                        dependent_required.setdefault(field, [])
                        dependent_required[field].append(field_name)
            if dependent_required:
                result_schema["dependentRequired"] = dependent_required

        # --- if/then/else and dependentSchemas from model_conditions ---
        if not hasattr(cls, "model_conditions"):
            return result_schema

        conditions = cls.model_conditions
        if not isinstance(conditions, (tuple, list)):
            conditions = (conditions,)

        if_then_else_results: list[dict[str, Any]] = []
        dependent_schemas: dict[str, Any] = {}

        for condition in conditions:
            if isinstance(condition, QueryBuilder):
                condition_result = condition._build()
                if not condition_result:
                    continue

                if_then_else_results.append(
                    _insert_conditionals_ref_marker(condition_result)
                )

            elif isinstance(condition, DependentSchemaBuilder):
                dependent_schema_result = condition._build()
                if not dependent_schema_result:
                    continue

                trigger, sub_schema = dependent_schema_result
                dependent_schemas.setdefault(trigger, [])
                dependent_schemas[trigger].append(
                    _insert_conditionals_ref_marker(sub_schema)
                )

        # Merge if/then/else results into the schema
        if len(if_then_else_results) > 1:
            result_schema.setdefault("allOf", [])
            result_schema["allOf"] = result_schema["allOf"] + if_then_else_results
        elif len(if_then_else_results) == 1:
            result_schema = {**result_schema, **if_then_else_results[0]}

        # Merge dependentSchemas into the schema
        if dependent_schemas:
            dependent_schemas_result: dict[str, Any] = {}
            for trigger, subschemas in dependent_schemas.items():
                if len(subschemas) == 1:
                    dependent_schemas_result[trigger] = subschemas[0]
                else:
                    dependent_schemas_result[trigger] = {"allOf": subschemas}

            result_schema["dependentSchemas"] = dependent_schemas_result

        return result_schema

    @classmethod
    def model_json_schema(
        cls,
        *args,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate JSON Schema for this model."""
        result = super().model_json_schema(*args, **kwargs)  # type: ignore[misc]
        result = _remove_conditionals_ref_marker(result)

        return result
