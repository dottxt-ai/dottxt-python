"""Generate structured output from a Pydantic model."""

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
    result = client.generate(
        model="openai/gpt-oss-20b",
        input=(
            "Summarize this incident: checkout errors are blocking purchases. "
            "Return a JSON object with keys severity and team."
        ),
        response_format=IncidentSummary,
    )
    print(result)
    # Example output:
    # severity='high' team='checkout'
    print(result.model_dump())
    # Example output:
    # {'severity': 'high', 'team': 'checkout'}


if __name__ == "__main__":
    main()
