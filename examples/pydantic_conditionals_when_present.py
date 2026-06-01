from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from dottxt import DotTxt
from dottxt.pydantic_conditionals import ConditionalModel, when_present


class Payment(ConditionalModel, BaseModel):
    credit_card: str | None = None
    billing_address: str | None = None

    model_conditions: ClassVar = (
        when_present("credit_card").require("billing_address"),
    )


if __name__ == "__main__":
    client = DotTxt()
    result = client.generate(
        model="openai/gpt-oss-20b",
        input=(
            "Return a JSON object representing a payment payload. "
            "Include credit_card and billing_address fields."
        ),
        response_format=Payment,
    )
    print(result)
    print(result.model_dump())
