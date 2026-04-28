"""List models available to the configured dottxt API key."""

from dottxt import DotTxt


def main() -> None:
    """Run the example."""
    client = DotTxt()
    models = client.models.list()
    for model in models.data:
        print(model.id)
    # Example output:
    # openai/gpt-oss-20b
    # openai/gpt-4.1-mini


if __name__ == "__main__":
    main()
