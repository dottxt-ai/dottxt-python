"""Rebuild the full document from raw patch ops as they arrive.

The SDK already hands you ``event.snapshot`` — an independent deep copy of
the document so far — so most consumers never touch the raw ops. This
example folds ``event.op`` into a document of its own, re-rendering it as it
grows. Use this as a template when you need to drive your own state from the
ops.

The gateway only ever emits RFC 6902 ``add`` ops, so we reuse the SDK's own
``apply_add`` helper rather than pulling in a full JSON Patch library.

Usage:
    DOTTXT_API_KEY=sk-... python examples/stream_reconstruct.py
"""

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field

from dottxt import AsyncDotTxt, apply_add


class Address(BaseModel):
    """A postal address."""

    city: str = Field(max_length=32)
    country: str = Field(max_length=32)


class Person(BaseModel):
    """A person with a nested address and a list of skills.

    Nesting and an array mean ops arrive for ``/address/city`` and
    ``/skills/0`` as well as for top-level scalars.
    """

    name: str = Field(max_length=32)
    age: int = Field(ge=0, le=120)
    address: Address
    skills: list[str] = Field(min_length=1, max_length=4)


async def main() -> None:
    """Run the example."""
    client = AsyncDotTxt()
    # Start from an empty object; the stream's first op is the root seed.
    doc: Any = {}
    try:
        stream = client.stream(
            model="openai/gpt-oss-20b",
            response_format=Person,
            input="Generate a profile for a backend engineer based in Paris.",
        )
        async for event in stream:
            doc = apply_add(doc, event.op["path"], event.op["value"])
            label = "(root)" if event.field == "" else event.field
            print(f"+ {label}")
            print(json.dumps(doc, indent=2))
            print("-" * 40)
    finally:
        await client.close()

    print("final document:")
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
