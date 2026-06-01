from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from dottxt import DotTxt
from dottxt.pydantic_conditionals import ConditionalModel, when


class Address(ConditionalModel, BaseModel):
    country: str
    postal_code: str | None = None

    model_conditions: ClassVar = (
        when(country="USA")
        .require("postal_code")
        .constrain(postal_code=Field(pattern=r"^\d{5}$")),
    )


if __name__ == "__main__":
    client = DotTxt()
    result = client.generate(
        model="openai/gpt-oss-20b",
        input=(
            "Return a JSON object for a US shipping address. "
            "Set country to USA and include a valid 5-digit postal_code."
        ),
        response_format=Address,
    )
    print(result)
    print(result.model_dump())
