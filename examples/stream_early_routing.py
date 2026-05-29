"""Route on /intent before /reply finishes.

The schema is ordered ``intent`` → ``urgency`` → ``reply``. Because dottxt
streams fields in schema order, the routing decision fires the moment
``intent`` arrives, typically tens of milliseconds in, while the model
continues generating the (much longer) ``reply``.

What to watch in the output: the ``-> dispatched ...`` and
``-> paged oncall ...`` lines arrive well before the final reply line.
The elapsed-time prefix on the reply line is the punchline, how much later
the full message lands compared to when routing was already settled.

Usage:
    DOTTXT_API_KEY=sk-... python examples/stream_early_routing.py
"""

import asyncio
import time
from typing import Literal

from pydantic import BaseModel, Field

from dottxt import AsyncDotTxt


class SupportTicket(BaseModel):
    """A triaged support reply.

    Field order is significant: earlier fields arrive first and unblock
    downstream work that does not depend on later fields.
    """

    intent: Literal["billing", "technical", "account", "feedback"]
    urgency: Literal["low", "medium", "high", "critical"]
    reply: str = Field(max_length=400)


async def route_to_billing(ticket_id: str) -> None:
    """Dispatch the ticket to the billing queue (stub)."""
    print(f"  -> dispatched {ticket_id} to billing queue")


async def route_to_technical(ticket_id: str) -> None:
    """Dispatch the ticket to the technical queue (stub)."""
    print(f"  -> dispatched {ticket_id} to technical queue")


async def page_oncall(ticket_id: str) -> None:
    """Page the on-call engineer (stub)."""
    print(f"  -> paged oncall for {ticket_id}")


async def main() -> None:
    """Run the example."""
    ticket_id = "TKT-8821"
    user_message = (
        "I was charged twice for my subscription this month and the second "
        "charge doesn't appear in my invoice list. Please refund the duplicate."
    )

    client = AsyncDotTxt()
    started = time.monotonic()
    try:
        stream = client.stream(
            model="openai/gpt-oss-20b",
            response_format=SupportTicket,
            input=[
                {
                    "role": "system",
                    "content": "Triage support tickets and draft a reply.",
                },
                {"role": "user", "content": user_message},
            ],
            max_tokens=400,
        )
        async for event in stream:
            match event.field:
                # Fire-and-forget: routing kicks off while /reply is still
                # streaming.
                case "intent" if event.value == "billing":
                    asyncio.create_task(route_to_billing(ticket_id))
                case "intent" if event.value == "technical":
                    asyncio.create_task(route_to_technical(ticket_id))
                case "urgency" if event.value == "critical":
                    asyncio.create_task(page_oncall(ticket_id))
                case "reply":
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    print(f"reply ({elapsed_ms}ms): {event.value}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
