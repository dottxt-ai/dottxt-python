"""Generate structured output using Genson inference and schema merging."""

from __future__ import annotations

from dottxt import DotTxt


def _build_inferred_schema() -> object:
    """Infer a schema from example payloads."""
    from genson import SchemaBuilder

    builder = SchemaBuilder()
    builder.add_object(
        {
            "severity": "high",
            "team": "checkout",
            "impact_score": 9,
        }
    )
    builder.add_object(
        {
            "severity": "medium",
            "team": "payments",
            "impact_score": 6,
        }
    )
    return builder


def _build_merged_schema() -> object:
    """Merge multiple schema fragments into one schema."""
    from genson import SchemaBuilder

    builder = SchemaBuilder()
    builder.add_schema(
        {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                "team": {"type": "string", "maxLength": 32},
            },
            "required": ["severity", "team"],
        }
    )
    builder.add_schema(
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "maxLength": 120},
            },
            "required": ["summary"],
        }
    )
    builder.add_schema({"type": "object", "additionalProperties": False})
    return builder


def main() -> None:
    """Run the example."""
    try:
        inferred_schema = _build_inferred_schema()
        merged_schema = _build_merged_schema()
    except ImportError as exc:
        raise SystemExit("Install genson first: uv add genson") from exc

    client = DotTxt()
    inferred_result = client.generate(
        model="openai/gpt-oss-20b",
        input=(
            "Return an incident object for: checkout errors are blocking purchases. "
            "Output must be a JSON object with severity, team, and integer "
            "impact_score. Keep team concise, like 'checkout'."
        ),
        response_format=inferred_schema,
    )
    print("Inferred schema result:")
    print(inferred_result)
    # Example output:
    # {'severity': 'high', 'team': 'checkout', 'impact_score': 9}

    schema = merged_schema.to_schema()
    schema["required"] = list(schema.get("properties", {}).keys())

    merged_result = client.generate(
        model="openai/gpt-oss-20b",
        input=(
            "Summarize this incident: checkout errors are blocking purchases. "
            "Output must be a JSON object with severity, team, and summary. "
            "Keep summary under 20 words."
        ),
        response_format=schema,
    )
    print("Merged schema result:")
    print(merged_result)
    # Example output:
    # {'severity': 'high', 'team': 'checkout', 'summary': 'Checkout errors...'}


if __name__ == "__main__":
    main()
