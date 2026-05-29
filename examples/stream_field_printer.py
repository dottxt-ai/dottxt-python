"""Print each leaf field and value as it arrives from the model.

Smallest possible demo of the patch-stream interface: define a schema,
iterate, print. No buffering, no closing brace.

Usage:
    DOTTXT_API_KEY=sk-... python examples/stream_field_printer.py
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from dottxt import AsyncDotTxt


class Engineer(BaseModel):
    """A software engineer profile."""

    name: str = Field(max_length=32)
    role: Literal["backend", "frontend", "ml", "infra"]
    years_experience: int = Field(ge=0, le=50)
    favorite_languages: list[str] = Field(min_length=1, max_length=4)


async def main() -> None:
    """Run the example."""
    client = AsyncDotTxt()
    try:
        stream = client.stream(
            model="openai/gpt-oss-20b",
            response_format=Engineer,
            input="Generate a profile for a senior backend engineer.",
        )
        async for event in stream:
            if not event.is_leaf:
                continue
            print(f"{event.field:>24} = {event.value!r}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
