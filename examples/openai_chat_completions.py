"""Use the OpenAI-compatible client surface with structured output."""

from typing import Literal

from pydantic import BaseModel, Field

from dottxt import DotTxt as OpenAI


class IncidentSummary(BaseModel):
    """Structured incident summary."""

    severity: Literal["low", "medium", "high"]
    team: str = Field(max_length=32)


def main() -> None:
    """Run the example."""
    client = OpenAI()
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this incident: checkout errors are blocking purchases."
                ),
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "incident_summary",
                "schema": IncidentSummary.model_json_schema(),
            },
        },
    )
    print(completion.choices[0].message.content)
    # Example output:
    # {"severity":"high","team":"checkout"}


if __name__ == "__main__":
    main()
