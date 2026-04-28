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

- `api_key` (`str | None`): falls back to `DOTTXT_API_KEY`. Required — the
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

## OpenAI-Compatible Text Generation

If you prefer the standard OpenAI SDK surface, you can call
`chat.completions.create(...)` directly. The client passes the call through
unchanged and returns the raw chat completion object — parsing and
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
