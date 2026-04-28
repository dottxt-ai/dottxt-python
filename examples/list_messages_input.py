"""Generate structured output from a list of chat messages."""

from typing import Literal

from pydantic import BaseModel, Field

from dottxt import DotTxt


class IncidentSummary(BaseModel):
    """Structured incident summary."""

    severity: Literal["low", "medium", "high"]
    team: str = Field(max_length=32)


def main() -> None:
    """Run the example."""
    client = DotTxt()
    messages = [
        {
            "role": "system",
            "content": (
                "You triage production incidents and route them to the right team."
            ),
        },
        {
            "role": "user",
            "content": "Checkout errors are blocking purchases across the storefront.",
        },
    ]
    result = client.generate(
        model="openai/gpt-oss-20b",
        input=messages,
        response_format=IncidentSummary,
    )
    print(result)
    # Example output:
    # severity='high' team='checkout'


if __name__ == "__main__":
    main()
