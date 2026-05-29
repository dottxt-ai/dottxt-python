# dottxt Python Client

The `dottxt` package exposes two clients based on the [OpenAI Python
SDK](https://github.com/openai/openai-python):

- `DotTxt`: synchronous client
- `AsyncDotTxt`: asynchronous client

Both clients expose the same surface:

- `models.list()`: list models available for the API key provided
- `generate(...)`: main structured-output method that accepts various output types
  to constrain the generation and returns parsed data
- `chat.completions.create(...)`: fully OpenAI-compatible chat completions

Both `generate` and `chat.completions.create` allow you to generate constrained text
based on an input and an output type, but `generate` includes helper features for
better convenience.

## Installation

```bash
pip install dottxt
```

Requires Python 3.10+.

## Initialization of the Client

The client takes two arguments. Each is read from the constructor first, then
from the environment.

- `api_key` (`str | None`): falls back to `DOTTXT_API_KEY`. Required, the
  constructor raises `ValueError` if neither is set.
- `base_url` (`str | None`): falls back to `DOTTXT_BASE_URL`, then to
  `https://api.dottxt.ai/v1`.

Any other keyword argument is forwarded to the underlying OpenAI SDK client.

`DotTxt` and `AsyncDotTxt` take the same arguments and expose the same
surface; the async version just returns awaitables.

```python
from dottxt import DotTxt, AsyncDotTxt

sync_client = DotTxt()
async_client = AsyncDotTxt()
```

## Listing Available Models

`models.list()` returns an OpenAI SDK `SyncPage[Model]` (or `AsyncPage[Model]` on
`AsyncDotTxt`). The models are under `.data` and each has an `id` property.

```python
from dottxt import DotTxt

client = DotTxt()
page = client.models.list()

for model in page.data:
    print(model.id)
# openai/gpt-oss-20b
# Qwen/Qwen3.5-35B-A3B-FP8
```

Use the `id` values as the `model=` argument of `generate(...)` or
`chat.completions.create(...)`. You can also set `DOTTXT_MODEL` to one of them
to use it as a default for the CLI.

## Generating Text

The `generate` method is the main way of generating constrained text. It requires
providing a model ID, a text input and the desired response format.

Parameters:

- `model` (`str`): a model ID, it must correspond to one of the models
  returned by the `models.list` method
- `input` (`str | list[dict]`): the prompt — a plain string or a list of
  OpenAI-style messages (see [Input](#input))
- `response_format` (`Any`): the output type to use to constrain the response
  (see [Output Types](#output-types))
- `temperature` (`float | None`), `max_tokens` (`int | None`), `seed`
  (`int | None`): optional arguments to control the generation parameters
- any other keyword is forwarded to the API endpoint. As the client is based
  on the OpenAI sdk, those keywords must be those used by
  `chat.completions.create`

### Input

`input` accepts two types of values:

- a plain string, wrapped as a single user message
- a list of OpenAI-style message dicts, passed through to the API unchanged. See the
  [OpenAI chat messages reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages)
  for the supported roles and fields.

```python
from typing import Literal
from pydantic import BaseModel
from dottxt import DotTxt

class IncidentSummary(BaseModel):
    severity: Literal["low", "medium", "high"]
    team: str

client = DotTxt()

# Plain string
client.generate(
    model="openai/gpt-oss-20b",
    input="Summarize this incident: checkout errors are blocking purchases.",
    response_format=IncidentSummary,
)

# Messages list (e.g. to add a system prompt or prior turns)
client.generate(
    model="openai/gpt-oss-20b",
    input=[
        {"role": "system", "content": "You are an incident-response assistant."},
        {"role": "user", "content": "Summarize: checkout errors are blocking purchases."},
    ],
    response_format=IncidentSummary,
)
```

### Output Types

`response_format` accepts numerous types that evaluate to a JSON Schema:

- a JSON Schema as a `str` or `dict`
- a Pydantic `BaseModel` subclass
- a `TypedDict` or `dataclass`
- an `Enum` class, `Literal[...]`, `Union[...]` or `Optional[...]`
- typing containers: `list[T]`, `dict[K, V]`, `tuple[...]`
- any object exposing `to_json() -> str` (e.g.
  [Genson](https://github.com/wolverdude/GenSON))

`generate(...)` returns decoded JSON (`dict`, `list`, `str`, ...) for all
response formats except Pydantic, for which it returns a validated model
instance.

Raw `list` instances are not supported — pass `list[T]` or a JSON Schema object.

### Errors

- `dottxt.InvalidOutputError`: raised by `generate(...)` when the completion
  cannot be parsed into the requested structure. It exposes:
  - `model`: model identifier used for generation
  - `raw_output`: raw completion text returned by the model
  - `finish_reason`: completion finish reason when available (a `"length"`
    value is called out in the message as a likely truncation)
  - `original_error`: underlying `json.JSONDecodeError` or Pydantic
    `ValidationError`
- `dottxt.schemas.InvalidSchemaError`: raised when `response_format` cannot be
  normalized to a JSON Schema object.
- Transport and API errors (rate limits, auth failures, ...) propagate from
  the OpenAI SDK. See the
  [OpenAI Python error classes](https://github.com/openai/openai-python#handling-errors).

### Examples

String input with a Pydantic model:

```python
from typing import Literal
from pydantic import BaseModel
from dottxt import DotTxt

class IncidentSummary(BaseModel):
    severity: Literal["low", "medium", "high"]
    team: str

client = DotTxt()
result = client.generate(
    model="openai/gpt-oss-20b",
    input="Summarize this incident: checkout errors are blocking purchases.",
    response_format=IncidentSummary,
)
print(result) # IncidentSummary(severity='high', team='checkout')
print(result.model_dump()) # {'severity': 'high', 'team': 'checkout'}
```

Messages input with a JSON Schema dict:

```python
from dottxt import DotTxt

schema = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "team": {"type": "string", "maxLength": 32},
    },
    "required": ["severity", "team"],
    "additionalProperties": False,
}

client = DotTxt()
result = client.generate(
    model="openai/gpt-oss-20b",
    input=[
        {"role": "system", "content": "You are an incident-response assistant."},
        {"role": "user", "content": "Summarize: checkout errors are blocking purchases."},
    ],
    response_format=schema,
)
print(result) # {'severity': 'high', 'team': 'checkout'}
```

## Streaming Fields (Patch Stream)

`AsyncDotTxt.stream(...)` yields `PatchEvent` objects as the model fills in a
schema-constrained response. It is built on the gateway's `stream: "patch"`
mode, which emits RFC 6902 JSON Patch operations in schema order, so
downstream work can start the moment a field arrives, without waiting for
the closing brace.

Parameters mirror `generate(...)`:

- `model` (`str`)
- `input` (`str | list[dict]`)
- `response_format` (`Any`) — any schema input accepted by `generate(...)`
- `temperature`, `max_tokens`, `seed`, `timeout` — optional
- `extra` (`dict | None`) — extra chat-completions body fields

Each `PatchEvent` carries:

- `event.op` — the raw RFC 6902 operation (`{"op": "add", "path": ..., "value": ...}`)
- `event.snapshot` — an independent deep copy of the JSON object built up to
  and including this op
- `event.is_leaf` / `event.field` / `event.value` convenience demux for
  the common case of reacting to one field at a time. `is_leaf` is `True`
  for non-structural adds (skipping the root seed and empty-container init
  ops); `field` is the JSON Pointer with the leading `/` stripped
  (`"intent"`, `"steps/0"`, `"address/city"`).

```python
import asyncio
from typing import Literal
from pydantic import BaseModel
from dottxt import AsyncDotTxt

class SupportTicket(BaseModel):
    # Field order = arrival order. Put what unblocks downstream work first.
    intent: Literal["billing", "technical", "account"]
    urgency: Literal["low", "medium", "high", "critical"]
    reply: str

async def main():
    client = AsyncDotTxt()
    stream = client.stream(
        model="openai/gpt-oss-20b",
        response_format=SupportTicket,
        input="I was charged twice this month, please refund the duplicate.",
    )
    async for event in stream:
        if not event.is_leaf:
            continue
        match event.field:
            case "intent":
                asyncio.create_task(dispatch_to_queue(event.value))
            case "urgency" if event.value == "critical":
                asyncio.create_task(page_oncall())
            case "reply":
                await send(event.value)

asyncio.run(main())
```

The routing decision fires the moment `intent` arrives, typically tens of
milliseconds in while `reply` continues to stream. If you need the full
object so far (e.g. to log progress or hand a partial object to another
service), use `event.snapshot`.

Errors:

- `dottxt.PatchStreamError`: raised when the gateway returns a non-200
  status. Exposes `status_code` and `body`.

## OpenAI-Compatible Text Generation

If you prefer the standard OpenAI SDK surface, you can call
`chat.completions.create(...)` directly. The client passes the call through
unchanged and returns the raw chat completion object, parsing and
validation are up to the caller.

For structured output, pass the wrapped OpenAI-style `response_format`
payload yourself:

```python
from typing import Literal
from pydantic import BaseModel
from dottxt import DotTxt

class IncidentSummary(BaseModel):
    severity: Literal["low", "medium", "high"]
    team: str

client = DotTxt()
completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Summarize this incident: checkout errors are blocking purchases.",
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "incident_summary",
            "schema": IncidentSummary.model_json_schema(),
        },
    },
)
print(completion.choices[0].message.content)
# {"severity":"high","team":"checkout"}
```

See the [OpenAI chat completions
reference](https://platform.openai.com/docs/api-reference/chat/create).

## Closing the Client

Both clients hold an underlying HTTP connection pool. Call `client.close()`
(or `await client.close()` on `AsyncDotTxt`) when you're done with it.

## Examples

Runnable examples live in the [`examples/`](../examples) directory:

- [`generate_pydantic.py`](../examples/generate_pydantic.py): generate with a
  Pydantic model
- [`generate_json_schema.py`](../examples/generate_json_schema.py): generate
  with a JSON Schema string
- [`list_models.py`](../examples/list_models.py): list available models
- [`openai_chat_completions.py`](../examples/openai_chat_completions.py): use
  the OpenAI-compatible `chat.completions.create` surface
- [`stream_field_printer.py`](../examples/stream_field_printer.py): minimal
  `stream` demo — print each leaf field and value as it lands
- [`stream_early_routing.py`](../examples/stream_early_routing.py): route on
  `/intent` while `/reply` is still streaming
- [`stream_hitl_approval.py`](../examples/stream_hitl_approval.py): approve a
  proposed action mid-stream and discard the reply if the operator declines
- [`stream_fanout.py`](../examples/stream_fanout.py): fan research tasks out
  on each `/steps/N` as the planner emits them
