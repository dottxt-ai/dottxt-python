# Pydantic Conditional Schemas

`dottxt.pydantic_conditionals` adds JSON Schema conditionals to Pydantic models.
Use it when you need schema-level `if`/`then`/`else` or dependency rules that are
enforced through a Pydantic response model passed to `DotTxt.generate(...)`.

## Imports

`ConditionalModel` must be listed before `BaseModel` in class inheritance:
`class MyModel(ConditionalModel, BaseModel): ...`.

```python
from typing import Annotated, ClassVar

from pydantic import BaseModel, Field

from dottxt import DotTxt
from dottxt.pydantic_conditionals import (
    ConditionalModel,
    RequiredWith,
    when,
    when_present,
)
```

## Pattern 1: Value-Based Conditional (`when`) with `generate(...)`

Use `when(...)` to trigger constraints when a field has a specific value.

```python
class Address(ConditionalModel, BaseModel):
    country: str
    postal_code: str | None = None

    model_conditions: ClassVar = (
        when(country="USA")
        .require("postal_code")
        .constrain(postal_code=Field(pattern=r"^\d{5}$")),
    )


client = DotTxt()
result = client.generate(
    model="openai/gpt-oss-20b",
    input=(
        "Return a JSON object for a US shipping address. "
        "Set country to USA and include a valid 5-digit postal_code."
    ),
    response_format=Address,
)
print(result.model_dump())
```

## Pattern 2: Presence-Based Conditional (`when_present`) with `generate(...)`

Use `when_present(...)` to trigger constraints when a field exists, independent
of its value.

```python
class Payment(ConditionalModel, BaseModel):
    credit_card: str | None = None
    billing_address: str | None = None

    model_conditions: ClassVar = (
        when_present("credit_card").require("billing_address"),
    )


client = DotTxt()
result = client.generate(
    model="openai/gpt-oss-20b",
    input=(
        "User lives at 1 main street and has credit card number `12345678910`."
        "Return a JSON object representing a payment payload. "
        "Include credit_card and billing_address fields."
    ),
    response_format=Payment,
)
print(result.model_dump())
```

## Pattern 3: Annotation-Based Dependency (`RequiredWith`)

Use `RequiredWith` when one field must be present whenever another field is
present.

```python
class FileOperation(ConditionalModel, BaseModel):
    content: str | None = None
    create_parents: Annotated[bool | None, RequiredWith("content")] = None
```

## Runnable Examples

- [examples/pydantic_conditionals_when.py](../examples/pydantic_conditionals_when.py)
- [examples/pydantic_conditionals_when_present.py](../examples/pydantic_conditionals_when_present.py)

## References

- JSON Schema conditionals: <https://json-schema.org/understanding-json-schema/reference/conditionals>
- JSON Schema object dependencies: <https://json-schema.org/understanding-json-schema/reference/conditionals#dependentrequired>
- Pydantic JSON schema docs: <https://docs.pydantic.dev/latest/concepts/json_schema/>
- Pydantic `Field` constraints: <https://docs.pydantic.dev/latest/concepts/fields/>
