# dottxt Python Library

The [.txt](https://dottxt.ai/) Python library provides access to the .txt REST
API from Python 3.10+ applications.

**[Request API access here.](https://h1xbpbfsf0w.typeform.com/to/fwQNWmS8?typeform-source=sdk.readme)**

It provides two client surfaces:

- `DotTxt` for sync access with DotTxt helpers and OpenAI-compatible
  namespaces
- `AsyncDotTxt` for async access with the same helper semantics

The API uses:

- base URL: `https://api.dottxt.ai/v1`
- auth: `Authorization: Bearer $DOTTXT_API_KEY`
- primary endpoints: `GET /models` and `POST /chat/completions`

## Install

```bash
pip install dottxt
```

## Configure

```bash
export DOTTXT_API_KEY="your-api-key"
```

All clients in this package read `DOTTXT_API_KEY` by default.

Optional overrides:

```bash
export DOTTXT_BASE_URL="https://api.dottxt.ai/v1"
export DOTTXT_MODEL="<model-id>"
```

## CLI

Use the `dottxt` CLI for login, model discovery, and one-off generation.

- CLI reference: [docs/cli.md](docs/cli.md)
- Client reference: [docs/client.md](docs/client.md)

## Client Surfaces

Choose the client that matches the shape you want to work with:

- `DotTxt.generate(...)` and `AsyncDotTxt.generate(...)`
  accept JSON Schema as a string/object, plus any typed schema supported
  by Pydantic `TypeAdapter` (for example: Pydantic models, Enums, Literals,
  Unions, Optionals, and typed containers), and objects exposing
  `to_json() -> str` (for example Genson).
  Root `{"type":"structural-tag", ...}` schema objects are accepted without
  JSON Schema metaschema validation.
  They return a validated Pydantic model instance for Pydantic input, or parsed
  JSON for the other schema input types.
- `DotTxt` and `AsyncDotTxt` also expose OpenAI SDK `chat` and `models`
  namespaces for direct SDK access alongside DotTxt helpers.

For constructor kwargs passed through to the OpenAI SDK client
(`DotTxt(..., **client_kwargs)` / `AsyncDotTxt(..., **client_kwargs)`), see
[OpenAI Python base client parameters](https://github.com/openai/openai-python/blob/main/src/openai/_base_client.py).

## Native DotTxt Client

```python
from typing import Literal

from pydantic import BaseModel, Field

from dottxt import DotTxt


class IncidentSummary(BaseModel):
    severity: Literal["low", "medium", "high"]
    team: str = Field(max_length=32)


client = DotTxt()

result = client.generate(
    model="openai/gpt-oss-20b",
    input="Summarize this incident: checkout errors are blocking purchases.",
    response_format=IncidentSummary,
)
print(result)
# Example model output:
# severity='high' team='checkout'
print(result.model_dump())
# Example output:
# {'severity': 'high', 'team': 'checkout'}

models = client.models.list()
print([model.id for model in models.data])
# Example output:
# ['openai/gpt-oss-20b', 'openai/gpt-4.1-mini']
```

### Async Native Client

```python
import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from dottxt import AsyncDotTxt


class IncidentSummary(BaseModel):
    severity: Literal["low", "medium", "high"]
    team: str = Field(max_length=32)


async def main() -> None:
    client = AsyncDotTxt()
    result = await client.generate(
        model="openai/gpt-oss-20b",
        input="Summarize this incident: checkout errors are blocking purchases.",
        response_format=IncidentSummary,
    )
    print(result)
    # Example model output:
    # severity='high' team='checkout'
    print(result.model_dump())
    # Example output:
    # {'severity': 'high', 'team': 'checkout'}

    models = await client.models.list()
    print([model.id for model in models.data])
    # Example output:
    # ['openai/gpt-oss-20b', 'openai/gpt-4.1-mini']


asyncio.run(main())
```

For `DotTxt` and `AsyncDotTxt`, `generate(...)` accepts `response_format` as:

- a Pydantic model class
- a TypedDict type
- a dataclass type
- an Enum class
- a `typing.Literal[...]` type
- a `typing.Union[...]` type
- a `typing.Optional[...]` type
- typed containers such as `list[...]`, `dict[...]`, `tuple[...]`
- a JSON string containing JSON Schema
- a JSON object (`dict`)
- an object exposing `to_json() -> str` that returns JSON Schema

Notes:
- Raw list instances as `response_format` are not supported.
- Root `{"type":"structural-tag", ...}` schema objects bypass metaschema checks.

For direct `chat.completions.create(...)`, pass the wrapped OpenAI-style
`response_format` payload yourself.

Use `DotTxt.models.list()` and `AsyncDotTxt.models.list()` for model listing.

## OpenAI-Compatible Usage

Use `DotTxt` when you want an OpenAI-style client surface with
`chat.completions.create(...)` and `models.list()`.

```python
from typing import Literal

from pydantic import BaseModel, Field

from dottxt import DotTxt as OpenAI


class IncidentSummary(BaseModel):
    severity: Literal["low", "medium", "high"]
    team: str = Field(max_length=32)


client = OpenAI()

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
# Example output:
# {"severity":"high","team":"checkout"}

models = client.models.list()
print([model.id for model in models.data])
# Example output:
# ['openai/gpt-oss-20b', 'openai/gpt-4.1-mini']
```

The compatibility surface expects the wrapped OpenAI-style
`response_format` payload:

- `{"type": "json_schema", "json_schema": {...}}`

## Examples

- [Provide a list of messages as input](examples/list_messages_input.py)
- [Use a JSON Schema to generate](examples/generate_json_schema.py)
- [Use a Pydantic BaseModel to generate](examples/generate_pydantic.py)
- [Use a dataclass type to generate](examples/generate_dataclass.py)
- [Use a TypedDict type to generate](examples/generate_typed_dict.py)
- [Use a Genson schema builder to generate](examples/generate_genson.py)
- [List available models](examples/list_models.py)
- [OpenAI-Compatible chat completions](examples/openai_chat_completions.py)
