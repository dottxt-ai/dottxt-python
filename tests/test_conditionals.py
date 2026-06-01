from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import BaseModel, Field

from dottxt.pydantic_conditionals import (
    ConditionalModel,
    RequiredWith,
    when,
    when_present,
    when_schema,
)
from dottxt.pydantic_conditionals.conditional import (
    Constraint,
    QueryBuilder,
    _normalize_require_fields,
)
from dottxt.pydantic_conditionals.constraint import (
    _build_required,
    compute_constraints,
)


def test_when_require_constrain():
    expected_schema = {
        "if": {
            "additionalProperties": True,
            "properties": {"country": {"const": "USA"}},
            "required": ["country"],
            "type": "object",
        },
        "properties": {
            "street": {"title": "Street", "type": "string"},
            "country": {"title": "Country", "type": "string"},
            "postal_code": {"title": "Postal Code", "type": "string"},
        },
        "required": ["street", "country", "postal_code"],
        "then": {
            "additionalProperties": True,
            "properties": {"postal_code": {"pattern": "^\\d{5}$"}},
            "required": ["postal_code"],
            "type": "object",
        },
        "title": "Address",
        "type": "object",
    }

    class Address(ConditionalModel, BaseModel):
        street: str
        country: str
        postal_code: str
        model_conditions: ClassVar = (
            when(country="USA")
            .require("postal_code")
            .constrain(postal_code=Field(pattern=r"^\d{5}$")),
        )

    schema = Address.model_json_schema()

    assert expected_schema == schema


def test_required_with():
    expected_schema = {
        "dependentRequired": {"content": ["create_parents"]},
        "properties": {
            "content": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Content",
            },
            "create_parents": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "title": "Create Parents",
            },
        },
        "title": "FileOperation",
        "type": "object",
    }

    class FileOperation(ConditionalModel, BaseModel):
        content: str | None = None
        create_parents: Annotated[bool | None, RequiredWith("content")] = None

    schema = FileOperation.model_json_schema()

    print(schema)
    assert expected_schema == schema


def test_when_present():
    expected_schema = {
        "dependentSchemas": {
            "credit_card": {
                "additionalProperties": True,
                "properties": {
                    "billing_address": {
                        "minItems": 1,
                        "minLength": 1,
                        "minProperties": 1,
                    }
                },
                "required": ["billing_address"],
                "type": "object",
            }
        },
        "properties": {
            "payment_method": {"title": "Payment Method", "type": "string"},
            "credit_card": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Credit Card",
            },
            "billing_address": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Billing Address",
            },
        },
        "required": ["payment_method"],
        "title": "Payment",
        "type": "object",
    }

    class Payment(ConditionalModel, BaseModel):
        payment_method: str
        credit_card: str | None = None
        billing_address: str | None = None

        model_conditions: ClassVar = (
            when_present("credit_card")
            .require("billing_address")
            .constrain(billing_address=Field(min_length=1)),
        )

    schema = Payment.model_json_schema()

    print(schema)
    assert expected_schema == schema


def test_or():
    expected_schema = {
        "if": {
            "anyOf": [
                {
                    "additionalProperties": True,
                    "properties": {"country": {"const": "USA"}},
                    "required": ["country"],
                    "type": "object",
                },
                {
                    "additionalProperties": True,
                    "properties": {"country": {"const": "Canada"}},
                    "required": ["country"],
                    "type": "object",
                },
            ]
        },
        "properties": {
            "country": {"title": "Country", "type": "string"},
            "postal_code": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Postal Code",
            },
        },
        "required": ["country"],
        "then": {
            "additionalProperties": True,
            "required": ["postal_code"],
            "type": "object",
        },
        "title": "Shipping",
        "type": "object",
    }

    class Shipping(ConditionalModel, BaseModel):
        country: str
        postal_code: str | None = None

        model_conditions: ClassVar = (
            (when(country="USA") | when(country="Canada")).require("postal_code"),
        )

    schema = Shipping.model_json_schema()

    print(schema)
    assert expected_schema == schema


def test_and():
    expected_schema = {
        "if": {
            "additionalProperties": True,
            "properties": {
                "country": {"const": "USA"},
                "state": {"const": "CA"},
            },
            "required": ["country", "state"],
            "type": "object",
        },
        "properties": {
            "country": {"title": "Country", "type": "string"},
            "state": {"title": "State", "type": "string"},
            "postal_code": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Postal Code",
            },
        },
        "required": ["country", "state"],
        "then": {
            "additionalProperties": True,
            "required": ["postal_code"],
            "type": "object",
        },
        "title": "AndExample",
        "type": "object",
    }

    class AndExample(ConditionalModel, BaseModel):
        country: str
        state: str
        postal_code: str | None = None
        model_conditions: ClassVar = (
            (when(country="USA") & when(state="CA")).require("postal_code"),
        )

    schema = AndExample.model_json_schema()

    print(schema)
    assert expected_schema == schema


def test_when_require_constrain_otherwise():
    expected_schema = {
        "else": {
            "additionalProperties": True,
            "properties": {"is_intl": {"const": True}},
            "required": ["is_intl"],
            "type": "object",
        },
        "if": {
            "additionalProperties": True,
            "properties": {"country": {"const": "USA"}},
            "required": ["country"],
            "type": "object",
        },
        "properties": {
            "street": {"title": "Street", "type": "string"},
            "country": {"title": "Country", "type": "string"},
            "postal_code": {"title": "Postal Code", "type": "string"},
            "is_intl": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "title": "Is Intl",
            },
        },
        "required": ["street", "country", "postal_code"],
        "then": {
            "additionalProperties": True,
            "properties": {
                "postal_code": {"pattern": "^\\d{5}$"},
                "is_intl": {"const": None},
            },
            "required": ["postal_code"],
            "type": "object",
        },
        "title": "Address",
        "type": "object",
    }

    class Address(ConditionalModel, BaseModel):
        street: str
        country: str
        postal_code: str
        is_intl: bool | None = None
        model_conditions: ClassVar = (
            when(country="USA")
            .require("postal_code")
            .constrain(postal_code=Field(pattern=r"^\d{5}$"))
            .constrain(is_intl=None)
            .otherwise(is_intl=True),
        )

    schema = Address.model_json_schema()

    print(schema)
    assert expected_schema == schema


def test_when_present_otherwise():
    expected_schema = {
        "else": {
            "additionalProperties": True,
            "properties": {"price": {"minimum": 10}},
            "required": ["price"],
            "type": "object",
        },
        "if": {
            "additionalProperties": True,
            "required": ["has_discount"],
            "type": "object",
        },
        "properties": {
            "has_discount": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "title": "Has Discount",
            },
            "price": {"default": 0.0, "title": "Price", "type": "number"},
        },
        "then": {
            "additionalProperties": True,
            "properties": {"price": {"minimum": 0}},
            "required": ["price"],
            "type": "object",
        },
        "title": "WhenPresentOtherwise",
        "type": "object",
    }

    class WhenPresentOtherwise(ConditionalModel, BaseModel):
        has_discount: bool | None = None
        price: float = 0.0
        model_conditions: ClassVar = (
            when_present("has_discount")
            .constrain(price=Field(ge=0))
            .otherwise(price=Field(ge=10)),
        )

    schema = WhenPresentOtherwise.model_json_schema()

    print(schema)
    assert expected_schema == schema


def test_when_or_when_present():
    expected_schema = {
        "if": {
            "anyOf": [
                {
                    "additionalProperties": True,
                    "properties": {"country": {"const": "USA"}},
                    "required": ["country"],
                    "type": "object",
                },
                {"additionalProperties": True, "required": ["state"], "type": "object"},
            ]
        },
        "properties": {
            "country": {"title": "Country", "type": "string"},
            "state": {"title": "State", "type": "string"},
            "postal_code": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Postal Code",
            },
        },
        "required": ["country", "state"],
        "then": {
            "additionalProperties": True,
            "required": ["postal_code"],
            "type": "object",
        },
        "title": "MixedCondition",
        "type": "object",
    }

    class MixedCondition(ConditionalModel, BaseModel):
        country: str
        state: str
        postal_code: str | None = None
        model_conditions: ClassVar = (
            (when(country="USA") | when_present("state")).require("postal_code"),
        )

    schema = MixedCondition.model_json_schema()

    print(schema)
    assert expected_schema == schema


def test_when_require():
    expected_schema = {
        "if": {
            "additionalProperties": True,
            "properties": {"country": {"const": "USA"}},
            "required": ["country"],
            "type": "object",
        },
        "properties": {
            "street": {"title": "Street", "type": "string"},
            "country": {"title": "Country", "type": "string"},
            "postal_code": {"default": "", "title": "Postal Code", "type": "string"},
        },
        "required": ["street", "country"],
        "then": {
            "additionalProperties": True,
            "required": ["postal_code"],
            "type": "object",
        },
        "title": "Address",
        "type": "object",
    }

    class Address(ConditionalModel, BaseModel):
        street: str
        country: str
        postal_code: str = ""
        model_conditions: ClassVar = when(country="USA").require("postal_code")

    schema = Address.model_json_schema()

    print(schema)
    assert expected_schema == schema


def test_bare_when():
    expected_schema = {
        "properties": {"foo": {"title": "Foo", "type": "string"}},
        "required": ["foo"],
        "title": "Foo",
        "type": "object",
    }

    class Foo(ConditionalModel, BaseModel):
        foo: str
        model_conditions: ClassVar = (when(foo="Foo"),)

    schema = Foo.model_json_schema()
    assert expected_schema == schema


def test_mixin_schemas():
    expected_schema = {
        "else": {
            "additionalProperties": True,
            "properties": {"baz": {"title": "Baz", "type": "string"}},
            "required": ["baz"],
            "title": "Baz",
            "type": "object",
        },
        "if": {
            "additionalProperties": True,
            "properties": {"foo": {"const": "Foo"}},
            "required": ["foo"],
            "type": "object",
        },
        "properties": {"foo": {"title": "Foo", "type": "string"}},
        "required": ["foo"],
        "then": {
            "additionalProperties": True,
            "properties": {"bar": {"title": "Bar", "type": "string"}},
            "required": ["bar"],
            "title": "Bar",
            "type": "object",
        },
        "title": "Foo",
        "type": "object",
    }

    class Baz(BaseModel):
        baz: str

    class Bar(BaseModel):
        bar: str

    class Foo(ConditionalModel, BaseModel):
        foo: str
        model_conditions: ClassVar = when(foo="Foo").then_apply(Bar).else_apply(Baz)

    schema = Foo.model_json_schema()
    print(schema)
    assert expected_schema == schema


def test_populated_or_empty():
    expected_schema = {
        "else": {
            "additionalProperties": True,
            "properties": {
                "foo": {
                    "default": [],
                    "items": {"type": "string"},
                    "maxItems": 0,
                    "title": "Foo",
                    "type": "array",
                }
            },
            "title": "FooElse",
            "type": "object",
        },
        "if": {
            "additionalProperties": True,
            "properties": {"should_foo": {"const": True}},
            "required": ["should_foo"],
            "type": "object",
        },
        "properties": {
            "should_foo": {"title": "Should Foo", "type": "boolean"},
            "foo": {"items": {"type": "string"}, "title": "Foo", "type": "array"},
        },
        "required": ["should_foo", "foo"],
        "then": {
            "additionalProperties": True,
            "properties": {
                "foo": {
                    "items": {"type": "string"},
                    "minItems": 1,
                    "title": "Foo",
                    "type": "array",
                }
            },
            "required": ["foo"],
            "title": "FooThen",
            "type": "object",
        },
        "title": "Foo",
        "type": "object",
    }

    class FooThen(BaseModel):
        foo: list[str] = Field(min_length=1)

    class FooElse(BaseModel):
        foo: list[str] = Field([], max_length=0)

    class Foo(ConditionalModel, BaseModel):
        should_foo: bool
        foo: list[str]
        model_conditions: ClassVar = (
            when(should_foo=True).then_apply(FooThen).else_apply(FooElse)
        )

    schema = Foo.model_json_schema()
    print(schema)
    assert expected_schema == schema


def test_drop_fields():
    expected_schema = {
        "if": {
            "additionalProperties": True,
            "properties": {"should_include_bar": {"const": True}},
            "required": ["should_include_bar"],
            "type": "object",
        },
        "properties": {
            "should_include_bar": {"title": "Should Include Bar", "type": "boolean"},
            "foo": {"title": "Foo", "type": "string"},
        },
        "required": ["should_include_bar", "foo"],
        "then": {
            "additionalProperties": True,
            "properties": {"bar": {"title": "Bar", "type": "string"}},
            "required": ["bar"],
            "title": "FooThen",
            "type": "object",
        },
        "title": "Foo",
        "type": "object",
    }

    class FooThen(BaseModel):
        bar: str

    class Foo(ConditionalModel, BaseModel):
        should_include_bar: bool
        foo: str
        model_conditions: ClassVar = when(should_include_bar=True).then_apply(FooThen)

    schema = Foo.model_json_schema()
    print(schema)
    assert expected_schema == schema


def test_multi_schema_then():
    expected_schema = {
        "if": {
            "additionalProperties": True,
            "properties": {"should_include_bar": {"const": True}},
            "required": ["should_include_bar"],
            "type": "object",
        },
        "properties": {
            "should_include_bar": {"title": "Should Include Bar", "type": "boolean"},
            "foo": {"title": "Foo", "type": "string"},
        },
        "required": ["should_include_bar", "foo"],
        "then": {
            "additionalProperties": True,
            "properties": {"baz": {"title": "Baz", "type": "string"}},
            "required": ["baz"],
            "title": "FooThenTwo",
            "type": "object",
        },
        "title": "Foo",
        "type": "object",
    }

    class FooThenTwo(BaseModel):
        baz: str

    class FooThen(BaseModel):
        bar: str

    class Foo(ConditionalModel, BaseModel):
        should_include_bar: bool
        foo: str
        model_conditions: ClassVar = (
            when(should_include_bar=True).then_apply(FooThen).then_apply(FooThenTwo)
        )

    schema = Foo.model_json_schema()
    print(schema)
    assert expected_schema == schema


def test_then_apply_with_ref():
    expected_schema = {
        "else": {
            "$defs": {
                "AAA": {
                    "properties": {"foo": {"title": "Foo", "type": "string"}},
                    "required": ["foo"],
                    "title": "AAA",
                    "type": "object",
                }
            },
            "additionalProperties": True,
            "properties": {
                "baz": {"title": "Baz", "type": "string"},
                "foo": {"$ref": "#/$defs/AAA"},
            },
            "required": ["baz", "foo"],
            "title": "Baz",
            "type": "object",
        },
        "if": {
            "additionalProperties": True,
            "properties": {"foo": {"const": "Foo"}},
            "required": ["foo"],
            "type": "object",
        },
        "properties": {"foo": {"title": "Foo", "type": "string"}},
        "required": ["foo"],
        "then": {
            "additionalProperties": True,
            "properties": {"bar": {"title": "Bar", "type": "string"}},
            "required": ["bar"],
            "title": "Bar",
            "type": "object",
        },
        "title": "Foo",
        "type": "object",
    }

    class AAA(BaseModel):
        foo: str

    class Baz(ConditionalModel, BaseModel):
        baz: str
        foo: AAA

    class Bar(BaseModel):
        bar: str

    class Foo(ConditionalModel, BaseModel):
        foo: str
        model_conditions: ClassVar = when(foo="Foo").then_apply(Bar).else_apply(Baz)

    schema = Foo.model_json_schema()
    print(schema)
    assert expected_schema == schema


def test_when_apply():
    class Foo_if(BaseModel):
        foo: Literal[None]

    class Foo_then(BaseModel):
        bar: Literal["No foo"]

    class Foo_else(BaseModel):
        foo: str
        bar: Literal["There is a foo"]

    class Foo(ConditionalModel, BaseModel):
        foo: str | None = None
        bar: str
        not_there: int
        model_conditions: ClassVar = (
            when_schema(Foo_if).then_apply_only(Foo_then).else_apply_only(Foo_else)
        )

    schema = Foo.model_json_schema()

    expected = {
        "else": {
            "additionalProperties": False,
            "properties": {
                "foo": {"title": "Foo", "type": "string"},
                "bar": {"const": "There is a foo", "title": "Bar", "type": "string"},
            },
            "required": ["foo", "bar"],
            "title": "Foo_else",
            "type": "object",
        },
        "if": {
            "additionalProperties": True,
            "properties": {"foo": {"const": None, "title": "Foo", "type": "null"}},
            "required": ["foo"],
            "title": "Foo_if",
            "type": "object",
        },
        "properties": {
            "foo": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Foo",
            },
            "bar": {"title": "Bar", "type": "string"},
            "not_there": {"title": "Not There", "type": "integer"},
        },
        "required": ["bar", "not_there"],
        "then": {
            "additionalProperties": False,
            "properties": {
                "bar": {"const": "No foo", "title": "Bar", "type": "string"},
            },
            "required": ["bar"],
            "title": "Foo_then",
            "type": "object",
        },
        "title": "Foo",
        "type": "object",
    }

    assert schema == expected


def test_normalize_require_fields_variants_and_error() -> None:
    assert _normalize_require_fields("a") == ["a"]
    assert _normalize_require_fields(("a",)) == ["a"]
    assert _normalize_require_fields(["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError, match="Parameter must be a list or str"):
        _normalize_require_fields(123)  # type: ignore[arg-type]


def test_build_required_empty() -> None:
    assert _build_required([]) == {}


def test_querybuilder_none_constraint_and_schema_conflict_error() -> None:
    qb = when(country=None)
    built = qb.require("postal_code")._build()
    assert built is not None
    assert built["if"]["properties"]["country"]["const"] is None

    with pytest.raises(ValueError, match="cannot provide a schema and a list"):
        QueryBuilder({"field": "x"}, schema={"type": "object"})


def test_querybuilder_and_or_guard_paths() -> None:
    class A(BaseModel):
        a: Literal["a"]

    class B(BaseModel):
        b: Literal["b"]

    with pytest.raises(ValueError, match="Can only combine"):
        _ = when(a="x") & 5  # type: ignore[operator]

    with pytest.raises(ValueError, match="Can only combine"):
        _ = when(a="x") | 5  # type: ignore[operator]

    with pytest.raises(ValueError, match="Cannot mix AND and OR"):
        _ = (when(a="x") | when(b="y")) & when(c="z")

    with pytest.raises(ValueError, match="has constraints"):
        _ = when(a="x") & when(b="y").constrain(q="v")

    with pytest.raises(ValueError, match="has constraints"):
        _ = when(a="x") | when(b="y").constrain(q="v")

    with pytest.raises(ValueError, match="Cannot combine where clauses"):
        _ = when_schema(A) & when(a="x")

    with pytest.raises(ValueError, match="Cannot combine where clauses"):
        _ = when_schema(A) | when(a="x")

    and_schema = (when_schema(A).require("a") & when_schema(B))._build()
    assert and_schema is not None
    assert "allOf" in and_schema["if"]

    or_schema = (when_schema(A).require("a") | when_schema(B))._build()
    assert or_schema is not None
    assert "anyOf" in or_schema["if"]


def test_dependent_builder_paths_and_requiredwith_error() -> None:
    class ThenSchema(BaseModel):
        out: str

    builder = when_present("trigger").then_apply(ThenSchema)
    built_pair = builder._build()
    assert built_pair is not None
    assert built_pair[0] == "trigger"
    assert built_pair[1]["additionalProperties"] is True

    assert when_present("trigger")._build() is None

    with pytest.raises(ValueError, match="at least one field"):
        RequiredWith()


class _ElseSchema(BaseModel):
    flag: bool


class _MarkerModel(ConditionalModel, BaseModel):
    value: Annotated[str, "metadata-that-is-not-requiredwith"]
    model_conditions: ClassVar = (when_present("x"),)


class _DependentMergeModel(ConditionalModel, BaseModel):
    x: str | None = None
    a: str | None = None
    b: str | None = None
    model_conditions: ClassVar = (
        when_present("x").require("a"),
        when_present("x").require("b"),
    )


class _IfThenListModel(ConditionalModel, BaseModel):
    first: str | None = None
    second: str | None = None
    model_conditions: ClassVar = (
        when(first="x").require("second"),
        when(second="y").require("first"),
    )


def test_conditional_model_branches_for_continue_and_merge() -> None:
    marker_schema = _MarkerModel.model_json_schema()
    assert "dependentSchemas" not in marker_schema

    dep_schema = _DependentMergeModel.model_json_schema()
    assert dep_schema["dependentSchemas"]["x"]["allOf"]

    if_then_schema = _IfThenListModel.model_json_schema()
    assert len(if_then_schema["allOf"]) >= 2


def test_dependent_builder_else_and_boolean_ops() -> None:
    qb_else = when_present("f").else_apply(_ElseSchema)
    built_else = qb_else._build()
    assert built_else is not None
    assert built_else["else"]["additionalProperties"] is True

    qb_and = when(a="x").require("a") & when_present("f")
    assert qb_and._build() is not None

    qb_or = when_present("f") | when(v="x")
    assert qb_or._build() is None

    assert ((when_present("f") & when(v="x"))._build()) is None


def test_then_apply_only_without_if_schema() -> None:
    class ThenOnly(BaseModel):
        out: str

    built = when(v="x").then_apply_only(ThenOnly)._build()
    assert built is not None
    assert built["then"]["additionalProperties"] is False


def test_conditional_mixin_without_model_fields() -> None:
    # ConditionalModel should pass through the downstream schema unchanged when
    # there is no conditional metadata to inject.
    payload = ConditionalModel.__get_pydantic_json_schema__(
        object,
        lambda _: {"x": 1},  # type: ignore[arg-type]
    )
    assert payload == {"x": 1}


class _MixedConditionsModel(ConditionalModel, BaseModel):
    value: str | None = None
    model_conditions: ClassVar = (object(), when(value="x").require("value"))


def test_conditional_model_ignores_unknown_conditions() -> None:
    schema = _MixedConditionsModel.model_json_schema()
    assert "if" in schema


def test_compute_constraints_existing_key_without_new_value() -> None:
    existing = {"x": Constraint(value="old", has_value=True, required=False)}
    result = compute_constraints(existing, {"x": Constraint(required=True)})
    assert result["x"].required is True
    assert result["x"].value == "old"
