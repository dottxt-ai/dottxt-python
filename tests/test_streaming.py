"""Tests for the patch-stream consumer."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from dottxt.streaming import PatchEvent, PatchStreamError, apply_add, stream


def test_patch_event_field_for_array_index_and_nested_path() -> None:
    """Array indices and nested object paths keep their joined segments."""
    arr = PatchEvent(
        op={"op": "add", "path": "/steps/0", "value": "verify"},
        snapshot={"steps": ["verify"]},
    )
    nested = PatchEvent(
        op={"op": "add", "path": "/address/city", "value": "Paris"},
        snapshot={"address": {"city": "Paris"}},
    )
    assert arr.field == "steps/0"
    assert nested.field == "address/city"


def test_apply_add_replaces_root_for_empty_path() -> None:
    """An op with ``path == ""`` replaces the document root."""
    assert apply_add(None, "", {}) == {}
    assert apply_add({"old": 1}, "", []) == []


def test_apply_add_sets_top_level_key() -> None:
    """Top-level keys are set on the root object in place."""
    doc: dict[str, Any] = {}
    apply_add(doc, "/intent", "billing")
    assert doc == {"intent": "billing"}


def test_apply_add_inserts_at_array_index() -> None:
    """Numeric path segments insert into arrays at that index."""
    doc: dict[str, list[str]] = {"steps": []}
    apply_add(doc, "/steps/0", "verify")
    apply_add(doc, "/steps/1", "refund")
    assert doc == {"steps": ["verify", "refund"]}


def test_apply_add_appends_with_dash() -> None:
    """The ``-`` path segment appends to an array."""
    doc: dict[str, list[int]] = {"xs": [1, 2]}
    apply_add(doc, "/xs/-", 3)
    assert doc == {"xs": [1, 2, 3]}


def test_apply_add_nested_object() -> None:
    """Nested object keys keep their segments joined by ``/``."""
    doc: dict[str, dict[str, str]] = {"address": {}}
    apply_add(doc, "/address/city", "Paris")
    assert doc == {"address": {"city": "Paris"}}


def _ndjson_response(ops: list[dict[str, Any]]) -> httpx.Response:
    """Build a fake NDJSON streaming response."""
    body = "".join(json.dumps(op) + "\n" for op in ops).encode()
    return httpx.Response(
        200,
        content=body,
        headers={"content-type": "application/x-ndjson"},
    )


def _install_mock_transport(monkeypatch: Any, handler: Any) -> None:
    """Route httpx.AsyncClient through a MockTransport for the duration of a test."""
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("dottxt.streaming.httpx.AsyncClient", fake_async_client)


@pytest.mark.asyncio
async def test_stream_yields_expected_events(monkeypatch: Any) -> None:
    """End-to-end: a canned NDJSON stream produces ops + growing snapshots."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return _ndjson_response(
            [
                {"op": "add", "path": "", "value": {}},
                {"op": "add", "path": "/intent", "value": "billing"},
                {"op": "add", "path": "/urgency", "value": "high"},
                {"op": "add", "path": "/steps", "value": []},
                {"op": "add", "path": "/steps/0", "value": "verify"},
                {"op": "add", "path": "/steps/1", "value": "refund"},
                {"op": "add", "path": "/reply", "value": "Done."},
            ]
        )

    _install_mock_transport(monkeypatch, handler)

    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "urgency": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "string"}},
            "reply": {"type": "string"},
        },
    }
    events = [
        event
        async for event in stream(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="openai/gpt-oss-20b",
            response_format=schema,
            input="go",
        )
    ]

    # One event per wire op — including structural ops (root seed, empty
    # containers).
    assert [e.op["path"] for e in events] == [
        "",
        "/intent",
        "/urgency",
        "/steps",
        "/steps/0",
        "/steps/1",
        "/reply",
    ]
    # Snapshot grows monotonically and reflects each op's effect.
    assert events[0].snapshot == {}
    assert events[1].snapshot == {"intent": "billing"}
    assert events[3].snapshot == {
        "intent": "billing",
        "urgency": "high",
        "steps": [],
    }
    assert events[5].snapshot == {
        "intent": "billing",
        "urgency": "high",
        "steps": ["verify", "refund"],
    }
    assert events[-1].snapshot == {
        "intent": "billing",
        "urgency": "high",
        "steps": ["verify", "refund"],
        "reply": "Done.",
    }
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["body"]["stream"] == "patch"
    assert captured["body"]["model"] == "openai/gpt-oss-20b"
    assert captured["auth"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_stream_snapshots_are_independent(monkeypatch: Any) -> None:
    """Each event's snapshot is a deep copy — later ops don't mutate earlier ones."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson_response(
            [
                {"op": "add", "path": "", "value": {}},
                {"op": "add", "path": "/steps", "value": []},
                {"op": "add", "path": "/steps/0", "value": "a"},
            ]
        )

    _install_mock_transport(monkeypatch, handler)

    events = [
        e
        async for e in stream(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="m",
            response_format={"type": "object"},
            input="go",
        )
    ]
    # The /steps event captured the empty list before /steps/0 was added.
    assert events[1].snapshot == {"steps": []}
    # And the op carried in event 0 (root seed) still shows the seed value,
    # not the final document state.
    assert events[0].op == {"op": "add", "path": "", "value": {}}


@pytest.mark.asyncio
async def test_stream_raises_on_non_200(monkeypatch: Any) -> None:
    """A non-200 response surfaces as PatchStreamError with the body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            content=b'{"error":"Forbidden","message":"no access"}',
            headers={"content-type": "application/json"},
        )

    _install_mock_transport(monkeypatch, handler)

    with pytest.raises(PatchStreamError) as info:
        async for _ in stream(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="m",
            response_format={"type": "object"},
            input="go",
        ):
            pass
    assert info.value.status_code == 403
    assert "Forbidden" in info.value.body


@pytest.mark.asyncio
async def test_stream_passes_list_input_unchanged(monkeypatch: Any) -> None:
    """When ``input`` is already a list of messages, it is forwarded as-is."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ndjson_response(
            [
                {"op": "add", "path": "", "value": {}},
                {"op": "add", "path": "/x", "value": 1},
            ]
        )

    _install_mock_transport(monkeypatch, handler)

    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "go"},
    ]
    events = [
        e
        async for e in stream(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="m",
            response_format={"type": "object"},
            input=messages,
        )
    ]
    assert events[-1].snapshot == {"x": 1}
    assert captured["body"]["messages"] == messages


@pytest.mark.asyncio
async def test_stream_passes_generation_params(monkeypatch: Any) -> None:
    """temperature / max_tokens / seed land in the request body."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ndjson_response([{"op": "add", "path": "", "value": {}}])

    _install_mock_transport(monkeypatch, handler)

    _ = [
        e
        async for e in stream(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="m",
            response_format={"type": "object"},
            input="go",
            temperature=0.2,
            max_tokens=128,
            seed=7,
            extra={"top_p": 0.9},
        )
    ]
    body = captured["body"]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 128
    assert body["seed"] == 7
    assert body["top_p"] == 0.9


@pytest.mark.asyncio
async def test_stream_tolerates_blank_lines_and_trailing_op(
    monkeypatch: Any,
) -> None:
    """Blank lines are skipped; a trailing op without a newline is flushed."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b'{"op":"add","path":"","value":{}}\n'
            b'{"op":"add","path":"/a","value":1}\n'
            b"\n"
            b"\n"
            b'{"op":"add","path":"/b","value":2}\n'
            b'{"op":"add","path":"/c","value":3}'
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/x-ndjson"},
        )

    _install_mock_transport(monkeypatch, handler)

    events = [
        e
        async for e in stream(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="m",
            response_format={"type": "object"},
            input="go",
        )
    ]
    assert [e.op["path"] for e in events] == ["", "/a", "/b", "/c"]
    assert events[-1].snapshot == {"a": 1, "b": 2, "c": 3}


@pytest.mark.asyncio
async def test_async_dottxt_stream_yields_patch_events(
    monkeypatch: Any,
) -> None:
    """AsyncDotTxt.stream forwards base_url + api_key and yields PatchEvents."""
    from dottxt import AsyncDotTxt

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        return _ndjson_response(
            [
                {"op": "add", "path": "", "value": {}},
                {"op": "add", "path": "/intent", "value": "billing"},
            ]
        )

    _install_mock_transport(monkeypatch, handler)

    client = AsyncDotTxt(
        api_key="sk-async-test",
        base_url="https://api.example.com/v1",
    )
    try:
        events = [
            e
            async for e in client.stream(
                model="m",
                response_format={"type": "object"},
                input="go",
            )
        ]
    finally:
        await client.close()

    assert all(isinstance(e, PatchEvent) for e in events)
    assert events[-1].snapshot == {"intent": "billing"}
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-async-test"
