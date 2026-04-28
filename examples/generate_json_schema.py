"""Generate structured output from a JSON Schema string."""

from dottxt import DotTxt

schema = """
{
  "type": "object",
  "properties": {
    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
    "team": {"type": "string", "maxLength": 32}
  },
  "required": ["severity", "team"],
  "additionalProperties": false
}
"""


def main() -> None:
    """Run the example."""
    client = DotTxt()
    result = client.generate(
        model="openai/gpt-oss-20b",
        input=(
            "Summarize this incident: checkout errors are blocking purchases. "
            "Return a JSON object with keys severity and team."
        ),
        response_format=schema,
    )
    print(result)
    # Example output:
    # {'severity': 'high', 'team': 'checkout'}


if __name__ == "__main__":
    main()
