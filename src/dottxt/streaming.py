"""Patch-stream consumer for the dottxt ``stream: "patch"`` endpoint.

The gateway emits one RFC 6902 ``add`` op per JSON token as the model
generates a structured response. This module yields ``PatchEvent`` objects
that carry the raw op together with a snapshot of the document built up to
and including that op.
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from dottxt.schemas import SchemaInput, build_chat_payload


class PatchStreamError(RuntimeError):
    """Raised when the upstream patch stream returns a non-200 status."""

    def __init__(self, *, status_code: int, body: str) -> None:
        super().__init__(f"patch stream failed: {status_code} {body[:500]}")
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class PatchEvent:
    """One wire op plus the document reconstructed up to and including it.

    ``op`` is the raw RFC 6902 operation as received, currently always an
    ``add``. ``snapshot`` is the document after the op has been applied; it
    is an independent deep copy, so callers may stash events without later
    ops mutating earlier snapshots.

    The ``field`` / ``value`` properties demux the op for the
    common pattern of reacting to one structured-output field at a time.
    """

    op: dict[str, Any]
    snapshot: dict[str, Any] | list[Any]

    @property
    def field(self) -> str:
        """JSON Pointer for this op with the leading ``/`` stripped.

        Top-level keys read as ``"intent"``, array items as ``"steps/0"``,
        nested fields as ``"address/city"``. Returns ``""`` for the root op.
        """
        path = self.op.get("path", "")
        return path[1:] if path.startswith("/") else path

    @property
    def value(self) -> Any:
        """The op's ``value`` (a leaf for leaf ops, a container for seeds)."""
        return self.op.get("value")


def apply_add(doc: Any, path: str, value: Any) -> Any:
    """Apply an RFC 6902 ``add`` op in place and return the (possibly new) root.

    Supports the ``path == ""`` root replacement and ``-`` for array append;
    numeric path segments index into arrays, everything else keys into
    objects. Mutates ``doc`` for non-root paths.
    """
    if path == "":
        return value
    parts = path[1:].split("/")
    cur = doc
    for p in parts[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    last = parts[-1]
    if isinstance(cur, list):
        idx = len(cur) if last == "-" else int(last)
        cur.insert(idx, value)
    else:
        cur[last] = value
    return doc


async def stream(
    *,
    base_url: str,
    api_key: str,
    model: str,
    response_format: SchemaInput,
    input: str | list[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    timeout: float = 60.0,
    extra: dict[str, Any] | None = None,
) -> AsyncIterator[PatchEvent]:
    """Yield ``PatchEvent``\\ s from a patch-streamed chat completion.

    Sends ``stream: "patch"`` to ``{base_url}/chat/completions`` and reads
    the NDJSON response, yielding one event per op as it arrives. Each
    event carries the raw op plus an independent snapshot of the document
    after the op has been applied.
    """
    body = build_chat_payload(
        model=model,
        response_format=response_format,
        input=input,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        extra=extra,
    )
    body["stream"] = "patch"

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode("utf-8", errors="replace")
                raise PatchStreamError(status_code=resp.status_code, body=detail)
            doc: Any = None
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    doc, event = _handle_line(line, doc)
                    yield event
            if buf.strip():
                doc, event = _handle_line(buf, doc)
                yield event


def _handle_line(line: str, doc: Any) -> tuple[Any, PatchEvent]:
    """Parse one NDJSON line, apply the op, and build the PatchEvent.

    Returns the updated root and an event whose ``op`` and ``snapshot`` are
    both independent of the live doc, so subsequent ops mutating the doc
    cannot retroactively change a yielded event.
    """
    op = json.loads(line)
    if op.get("op") == "add":
        doc = apply_add(doc, op["path"], op["value"])
    return doc, PatchEvent(op=copy.deepcopy(op), snapshot=copy.deepcopy(doc))
