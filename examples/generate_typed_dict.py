"""Generate structured output using a TypedDict schema input."""

from __future__ import annotations

from typing import Literal, TypedDict

from dottxt import DotTxt


class IncidentPayload(TypedDict):
    """TypedDict schema for incident summaries."""

    severity: Literal["low", "medium", "high"]
    team: str


def main() -> None:
    """Run the example."""
    client = DotTxt()
    result = client.generate(
        model="openai/gpt-oss-20b",
        input=(
            "Summarize this incident: checkout errors are blocking purchases. "
            "Return a JSON object with keys severity and team."
        ),
        response_format=IncidentPayload,
    )
    print(result)
    # Example output:
    # {'severity': 'high', 'team': 'checkout'}


if __name__ == "__main__":
    main()
