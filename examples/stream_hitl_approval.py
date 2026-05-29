"""Mid-stream human approval, no checkpointer, no resume.

The proposed action arrives as a fact (``action``) before the reply is
generated. We prompt the operator between receiving the fact and acting on
it. If the operator declines, the rest of the stream is consumed but the
``reply`` is never sent.

Usage:
    DOTTXT_API_KEY=sk-... python examples/stream_hitl_approval.py
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from dottxt import AsyncDotTxt


class AgentDecision(BaseModel):
    """An agent's proposed action and customer-facing reply.

    ``action`` precedes ``reply`` so the operator can approve or reject
    while the reply text is still streaming.
    """

    action: Literal[
        "answer_only",
        "open_ticket",
        "issue_refund",
        "delete_account",
    ]
    reply: str = Field(max_length=300)


HIGH_RISK_ACTIONS = {"issue_refund", "delete_account"}


async def ask_human(question: str) -> bool:
    """Prompt the operator on stdin; return True if they approve."""
    # input() is blocking; run it off the event loop so other tasks
    # (e.g. background dispatching) keep running while we wait.
    answer = await asyncio.to_thread(input, f"{question} [y/N]: ")
    return answer.strip().lower() in {"y", "yes"}


async def send_reply(reply: str) -> None:
    """Send the customer-facing reply (stub)."""
    print(f"sent reply: {reply}")


async def main() -> None:
    """Run the example."""
    client = AsyncDotTxt()
    user_message = "Please close my account permanently. I am leaving."

    approved = True
    proposed_action: str | None = None

    try:
        stream = client.stream(
            model="openai/gpt-oss-20b",
            response_format=AgentDecision,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer support agent. Decide on an "
                        "action and draft a reply."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            max_tokens=300,
        )
        async for event in stream:
            if not event.is_leaf:
                continue
            match event.field:
                case "action":
                    proposed_action = event.value
                    print(f"proposed action: {event.value}")
                    if event.value in HIGH_RISK_ACTIONS:
                        approved = await ask_human(f"Approve action '{event.value}'?")
                        if not approved:
                            print("operator declined — reply will not be sent")
                case "reply" if approved:
                    await send_reply(event.value)
                case "reply":
                    print(f"discarded reply (action '{proposed_action}' declined)")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
