"""Fan out research tasks as a planner emits each step.

The planner schema has a top-level ``steps`` array. Each item streams in as
a separate field (``steps/0``, ``steps/1``, ...). We launch a research
coroutine the moment each step arrives, so step 0's research is already
underway while step 1 is still being generated.

What to watch in the output
---------------------------

The interesting comparison is **total wall-clock time vs. the sum of the
per-step research times**. Each step's stub research takes ~1.0-1.8s
(``1.0 + 0.2 * index``); if research ran serially after generation
completed, you'd pay generation_time + Σ research_time. Because each
``research()`` task is scheduled on arrival and the planner keeps
emitting while they run, total time is closer to
``max(generation_time, generation_time_until_last_step + slowest_research)``
 (i.e. roughly one research interval longer than generation, not the sum).

Look for:

- ``[t+...ms] received step N`` lines arriving while earlier
  ``[step N-1] started`` lines are still mid-flight (overlap is visible).
- The final ``all N steps researched in Xms total`` line: ``X`` should be
  noticeably less than ``generation_ms + Σ (1000 + 200·i)`` for emitted
  steps. With 4 steps that sum is ~4.6s of research; total tends to land
  near generation_time + ~1.6s rather than generation_time + ~4.6s.

Usage:
    DOTTXT_API_KEY=sk-... python examples/stream_fanout.py
"""

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from dottxt import AsyncDotTxt


class Plan(BaseModel):
    """An ordered research plan."""

    topic: str = Field(max_length=80)
    steps: list[str] = Field(min_length=3, max_length=5)


async def research(step_index: int, step: str) -> dict[str, Any]:
    """Pretend to research a single step (sleep + return)."""
    started = time.monotonic()
    print(f"  [step {step_index}] started: {step!r}")
    # Simulate a real lookup. Different durations make the overlap visible.
    await asyncio.sleep(1.0 + 0.2 * step_index)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    print(f"  [step {step_index}] done in {elapsed_ms}ms")
    return {"step": step, "elapsed_ms": elapsed_ms}


async def main() -> None:
    """Run the example."""
    client = AsyncDotTxt()
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    started = time.monotonic()

    try:
        stream = client.stream(
            model="openai/gpt-oss-20b",
            response_format=Plan,
            input=(
                "Plan three to five research steps to answer the question: "
                "'What are the trade-offs between RAG and fine-tuning for "
                "domain-specific assistants?' Each step should be a short "
                "imperative sentence."
            ),
            max_tokens=400,
        )
        async for event in stream:
            if not event.is_leaf:
                continue
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if event.field.startswith("steps/"):
                index = int(event.field.split("/", 1)[1])
                print(f"[t+{elapsed_ms:>5}ms] received step {index}")
                tasks.append(asyncio.create_task(research(index, event.value)))
            elif event.field == "topic":
                print(f"[t+{elapsed_ms:>5}ms] topic: {event.value}")

        results = await asyncio.gather(*tasks)
    finally:
        await client.close()

    total_ms = int((time.monotonic() - started) * 1000)
    sum_research_ms = sum(r["elapsed_ms"] for r in results)
    print()
    print(f"all {len(results)} steps researched in {total_ms}ms total")
    print(
        f"sum of per-step research times: {sum_research_ms}ms "
        f"(serial would take at least this long; overlap saved "
        f"{max(0, sum_research_ms - total_ms)}ms)"
    )


if __name__ == "__main__":
    asyncio.run(main())
