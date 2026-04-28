"""CLI for dottxt."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

import click
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from dottxt import __version__
from dottxt.client import DotTxt


def _credentials_path() -> Path:
    """Return the credentials file path for this machine."""
    app_dir = Path(click.get_app_dir("dottxt"))
    return app_dir / "credentials.json"


def _write_credentials(api_key: str) -> None:
    """Persist the API key to local credentials storage."""
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"api_key": api_key}, indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Best effort on platforms where chmod behavior may vary.
        pass


def _read_credentials_api_key() -> str | None:
    """Read API key from local credentials file when available."""
    path = _credentials_path()
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    api_key = payload.get("api_key")
    return api_key if isinstance(api_key, str) and api_key.strip() else None


def _resolve_api_key() -> str | None:
    """Resolve API key from environment, then credentials file."""
    env_key = os.getenv("DOTTXT_API_KEY")
    if env_key is not None:
        normalized_env_key = env_key.strip()
        if normalized_env_key:
            return normalized_env_key
    return _read_credentials_api_key()


def _resolve_default_model() -> str | None:
    """Resolve default model from environment."""
    env_model = os.getenv("DOTTXT_MODEL")
    if env_model is not None:
        normalized_env_model = env_model.strip()
        if normalized_env_model:
            return normalized_env_model
    return None


def _is_model_unavailable_error(exc: Exception) -> bool:
    """Return whether an exception appears to describe an unavailable model."""
    message = str(exc).lower()
    if "model" not in message:
        return False
    unavailable_markers = (
        "not found",
        "not available",
        "unknown model",
        "does not exist",
        "not permitted",
        "access denied",
        "permission denied",
        "invalid model",
    )
    return any(marker in message for marker in unavailable_markers)


def _jsonable(value: Any) -> Any:
    """Convert SDK objects into JSON-serializable primitives."""
    if isinstance(value, dict):
        return {key: _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())
    asdict = getattr(value, "to_dict", None)
    if callable(asdict):
        return _jsonable(asdict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value


def _close_client(client: DotTxt | None) -> None:
    """Close a client instance when available."""
    if client is None:
        return
    client.close()


def _fetch_models(
    api_key: str,
    *,
    json_mode: bool,
    error_prefix: str,
) -> list[dict[str, Any]]:
    """Fetch models and normalize to entries that include an ``id``."""
    client: DotTxt | None = None
    try:
        client = DotTxt(api_key=api_key)
        response = client.models.list()
    except Exception as exc:
        message = _api_error_message(
            exc,
            auth_message="Invalid API key. Please verify your key and try again.",
            connection_message=(
                "Unable to reach the API while fetching models. Please try again."
            ),
            rate_limit_message=(
                "API rate limit reached while fetching models. Please try again."
            ),
            status_prefix="API returned status",
            status_suffix="while fetching models.",
            fallback_prefix=error_prefix,
            include_status_detail=False,
        )
        _fail(message, json_mode=json_mode)
    finally:
        _close_client(client)

    model_data = _jsonable(getattr(response, "data", []))
    if not isinstance(model_data, list):
        return []
    return [
        item
        for item in model_data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def _validate_api_key(api_key: str, *, json_mode: bool) -> list[str]:
    """Validate API key by attempting to fetch model IDs."""
    models = _fetch_models(
        api_key,
        json_mode=json_mode,
        error_prefix="Unable to authenticate API key",
    )
    return [item["id"] for item in models]


def _display_path(path: Path) -> str:
    """Return a user-facing path with home abbreviated to ~ when possible."""
    home = Path.home()
    try:
        relative = path.relative_to(home)
    except ValueError:
        return str(path)
    return str(Path("~") / relative)


def _read_stdin_prompt() -> str:
    """Read prompt text from stdin."""
    if click.get_text_stream("stdin").isatty():
        return ""
    return click.get_text_stream("stdin").read().strip()


def _stdin_is_tty() -> bool:
    """Return whether stdin is connected to a TTY."""
    return sys.stdin.isatty()


def _read_stdin() -> str:
    """Read full stdin content until EOF."""
    return sys.stdin.read()


def _emit_verbose(ctx: click.Context, message: str, *, data: Any | None = None) -> None:
    """Print human-oriented verbose diagnostics to stderr."""
    if not bool(ctx.obj["verbose"]):
        return
    if data is None:
        click.echo(f"[verbose] {message}", err=True)
        return
    if isinstance(data, dict):
        compact_data = ", ".join(f"{key}={value}" for key, value in data.items())
    else:
        compact_data = json.dumps(data, separators=(",", ":"))
    click.echo(f"[verbose] {message} {compact_data}", err=True)


def _emit(data: Any, *, json_mode: bool) -> None:
    """Print command output."""
    if json_mode:
        click.echo(json.dumps(data))
        return
    if isinstance(data, str):
        click.echo(data)
        return
    click.echo(json.dumps(data, indent=2))


def _emit_error(message: str, *, json_mode: bool) -> None:
    """Print a machine-readable error message when requested."""
    if json_mode:
        click.echo(json.dumps({"error": {"message": message}}), err=True)


def _fail(message: str, *, json_mode: bool) -> NoReturn:
    """Terminate command execution with JSON-only errors when requested."""
    _emit_error(message, json_mode=json_mode)
    if json_mode:
        raise click.exceptions.Exit(1)
    raise click.ClickException(message)


def _status_error_message(prefix: str, exc: APIStatusError) -> str:
    """Build a detailed message for API status failures."""
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail: str | None = None
    if isinstance(body, dict):
        error_payload = body.get("error")
        if isinstance(error_payload, dict):
            error_message = error_payload.get("message")
            if isinstance(error_message, str) and error_message.strip():
                detail = error_message.strip()
        if detail is None:
            detail = json.dumps(body, separators=(",", ":"))
    elif isinstance(body, str) and body.strip():
        detail = body.strip()
    if status_code is None:
        message = f"{prefix} failed with API status error."
    else:
        message = f"{prefix} failed with API status {status_code}."
    if detail:
        return f"{message} {detail}"
    return message


def _api_error_message(
    exc: Exception,
    *,
    auth_message: str,
    connection_message: str,
    rate_limit_message: str,
    status_prefix: str,
    fallback_prefix: str,
    include_status_detail: bool,
    status_suffix: str = "",
) -> str:
    """Map known API exceptions to user-facing messages."""
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return auth_message
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return connection_message
    if isinstance(exc, RateLimitError):
        return rate_limit_message
    if isinstance(exc, APIStatusError):
        if include_status_detail:
            return _status_error_message(status_prefix, exc)
        return f"{status_prefix} {exc.status_code} {status_suffix}".strip()
    return f"{fallback_prefix}: {exc}"


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option("--version", is_flag=True, help="Print version and exit.")
@click.option("--json", "json_mode", is_flag=True, help="Machine-readable output.")
@click.option("--verbose", is_flag=True, help="Expanded diagnostics.")
@click.pass_context
def main(
    ctx: click.Context,
    version: bool,
    json_mode: bool,
    verbose: bool,
) -> None:
    """dottxt command line interface."""
    ctx.ensure_object(dict)
    ctx.obj["json_mode"] = json_mode
    ctx.obj["verbose"] = verbose

    if version:
        _emit(__version__, json_mode=json_mode)
        ctx.exit(0)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


@main.command()
@click.pass_context
def login(ctx: click.Context) -> None:
    """Store an API key locally."""
    json_mode = bool(ctx.obj["json_mode"])

    credentials_path = _credentials_path()
    resolved_api_key = os.getenv("DOTTXT_API_KEY")
    if resolved_api_key is not None:
        resolved_api_key = resolved_api_key.strip()
    source = "env"
    if not resolved_api_key:
        if _stdin_is_tty():
            resolved_api_key = click.prompt(
                "API key",
                hide_input=True,
                err=True,
            ).strip()
            source = "prompt"
        else:
            resolved_api_key = _read_stdin().strip()
            source = "stdin"
    if not resolved_api_key:
        _fail(
            (
                "No API key provided. Set DOTTXT_API_KEY, enter it interactively, "
                "or pipe it on stdin."
            ),
            json_mode=json_mode,
        )

    available_model_ids = _validate_api_key(resolved_api_key, json_mode=json_mode)
    _write_credentials(resolved_api_key)
    payload = {
        "status": "ok",
        "path": str(credentials_path),
        "source": source,
        "models_available": len(available_model_ids),
    }
    _emit_verbose(ctx, "Stored credentials.", data=payload)
    if json_mode:
        _emit(payload, json_mode=json_mode)
        return
    _emit("Logged in.", json_mode=False)


@main.command()
@click.pass_context
def logout(ctx: click.Context) -> None:
    """Delete locally stored credentials."""
    json_mode = bool(ctx.obj["json_mode"])
    path = _credentials_path()

    removed = False
    if path.exists():
        path.unlink()
        removed = True

    payload = {"status": "ok", "removed": removed, "path": str(path)}
    _emit_verbose(ctx, "Processed logout.", data=payload)
    if json_mode:
        _emit(payload, json_mode=json_mode)
        return
    _emit("Logged out.", json_mode=False)


@main.command()
@click.option("--author", type=str, help="Filter by model author.")
@click.pass_context
def models(ctx: click.Context, author: str | None) -> None:
    """List available models from the API."""
    json_mode = bool(ctx.obj["json_mode"])
    resolved_default_model = _resolve_default_model()
    resolved_api_key = _resolve_api_key()
    if not resolved_api_key:
        _fail(
            "No API key available. Run 'dottxt login' or set DOTTXT_API_KEY.",
            json_mode=json_mode,
        )
    normalized_models = _fetch_models(
        resolved_api_key,
        json_mode=json_mode,
        error_prefix="Unable to list models",
    )
    if author:
        normalized_models = [
            item for item in normalized_models if item.get("author") == author
        ]
    _emit_verbose(
        ctx,
        "Prepared model listing.",
        data={"author_filter": author, "count": len(normalized_models)},
    )
    if json_mode:
        _emit(normalized_models, json_mode=json_mode)
        return

    model_ids = [item["id"] for item in normalized_models]
    for model_id in model_ids:
        if model_id == resolved_default_model:
            click.echo(f"{model_id} * (default)")
            continue
        click.echo(model_id)


@main.command(name="generate")
@click.option(
    "-m",
    "--model",
    envvar="DOTTXT_MODEL",
    required=True,
    show_envvar=True,
    help="Model to use.",
)
@click.option(
    "-s",
    "--schema",
    "schema_file",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="JSON Schema file.",
)
@click.argument("prompt_arg", required=False, metavar="[PROMPT]")
@click.pass_context
def generate(
    ctx: click.Context,
    model: str,
    schema_file: Path,
    prompt_arg: str | None,
) -> None:
    """Generate structured output via the dottxt API.

    PROMPT is literal text and is required unless stdin is piped.
    The model resolves from --model, then DOTTXT_MODEL.
    """
    json_mode = bool(ctx.obj["json_mode"])
    if not schema_file.exists() or not schema_file.is_file():
        message = f"Schema file not found: {schema_file}"
        _fail(message, json_mode=json_mode)

    schema_text = schema_file.read_text(encoding="utf-8")
    try:
        schema_payload = json.loads(schema_text)
    except json.JSONDecodeError as exc:
        message = f"Schema file is not valid JSON: {exc.msg}"
        _fail(message, json_mode=json_mode)

    if prompt_arg is not None:
        final_prompt = prompt_arg
    else:
        final_prompt = _read_stdin_prompt()
    if not final_prompt:
        message = "No prompt provided. Use PROMPT or pipe stdin."
        _fail(message, json_mode=json_mode)

    resolved_api_key = _resolve_api_key()
    if not resolved_api_key:
        _fail(
            "No API key available. Run 'dottxt login' or set DOTTXT_API_KEY.",
            json_mode=json_mode,
        )

    client: DotTxt | None = None
    try:
        client = DotTxt(api_key=resolved_api_key)
        generated = client.generate(
            model=model,
            response_format=schema_text,
            input=final_prompt,
        )
    except Exception as exc:
        if _is_model_unavailable_error(exc):
            _fail(
                (
                    f"Model '{model}' is not available for this API key. "
                    "Run 'dottxt models' to choose an available model, then set "
                    "DOTTXT_MODEL or pass --model."
                ),
                json_mode=json_mode,
            )
        message = _api_error_message(
            exc,
            auth_message=f"Unable to authenticate API key: {exc}",
            connection_message=f"Unable to reach API: {exc}",
            rate_limit_message=f"Rate limited by API: {exc}",
            status_prefix="Generation",
            fallback_prefix="Generation failed",
            include_status_detail=True,
        )
        _emit_verbose(
            ctx,
            "Generation request failed.",
            data={"error_type": type(exc).__name__, "detail": message},
        )
        _fail(message, json_mode=json_mode)
    finally:
        _close_client(client)

    response: dict[str, Any] = {
        "model": model,
        "prompt": final_prompt,
        "schema_summary": {
            "type": schema_payload.get("type")
            if isinstance(schema_payload, dict)
            else None,
            "keys": list(schema_payload.keys())
            if isinstance(schema_payload, dict)
            else [],
        },
        "result": _jsonable(generated),
    }
    _emit_verbose(
        ctx,
        "Generated response.",
        data={
            "model": model,
            "schema_file": str(schema_file),
            "prompt_length": len(final_prompt),
            "cwd": os.getcwd(),
        },
    )
    if json_mode:
        _emit(response, json_mode=json_mode)
        return
    _emit(response["result"], json_mode=False)
