from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, Field

from dottxt.pydantic_conditionals import (
    ConditionalModel,
    RequiredWith,
    when,
    when_present,
    when_schema,
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


def test_compatibility_mode():
    class FileOp(ConditionalModel, BaseModel):
        content: str | None = None
        create_parents: Annotated[bool | None, RequiredWith("content")] = None

    schema = FileOp.model_json_schema(compatibility_mode=True)

    assert "dependentRequired" not in schema

    assert "allOf" in schema
    # Should keep if/then/else (not convert to anyOf/allOf/not)
    rule = schema["allOf"][0]
    assert "if" in rule
    assert "then" in rule


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
            "not_there": {"default": 0, "title": "Not There", "type": "integer"},
        },
        "required": ["bar"],
        "then": {
            "additionalProperties": False,
            "properties": {
                "foo": {"const": None, "title": "Foo", "type": "null"},
                "bar": {"const": "No foo", "title": "Bar", "type": "string"},
            },
            "required": ["foo", "bar"],
            "title": "Foo_then",
            "type": "object",
        },
        "title": "Foo",
        "type": "object",
    }

    assert schema == expected
