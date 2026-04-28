"""Focused tests for CLI helper behavior and error paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import click
import pytest
from click.testing import CliRunner
from openai import APIStatusError

from dottxt import cli as cli_module
from dottxt.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Return a CLI runner instance."""
    return CliRunner()


def _write_credentials_file(tmp_path: Path, content: str) -> None:
    """Write raw credentials payload under the expected config path."""
    credentials_file = tmp_path / "dottxt" / "credentials.json"
    credentials_file.parent.mkdir(parents=True, exist_ok=True)
    credentials_file.write_text(content, encoding="utf-8")


def test_read_credentials_api_key_returns_none_when_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing credentials file should resolve to None."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert cli_module._read_credentials_api_key() is None


def test_read_credentials_api_key_returns_none_for_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid credentials JSON should resolve to None."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _write_credentials_file(tmp_path, "{")

    assert cli_module._read_credentials_api_key() is None


def test_read_credentials_api_key_returns_none_for_blank_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Blank API key values should resolve to None."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _write_credentials_file(tmp_path, '{"api_key":"   "}')

    assert cli_module._read_credentials_api_key() is None


def test_read_credentials_api_key_returns_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Valid credentials payload should return API key."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _write_credentials_file(tmp_path, '{"api_key":"file-key"}')

    assert cli_module._read_credentials_api_key() == "file-key"


def test_resolve_api_key_prefers_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment API key should take precedence over file key."""
    monkeypatch.setenv("DOTTXT_API_KEY", "env-key")
    monkeypatch.setattr(cli_module, "_read_credentials_api_key", lambda: "file-key")

    assert cli_module._resolve_api_key() == "env-key"


def test_resolve_api_key_falls_back_to_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentials file value should be used when env key is absent."""
    monkeypatch.delenv("DOTTXT_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "_read_credentials_api_key", lambda: "file-key")

    assert cli_module._resolve_api_key() == "file-key"


def test_resolve_api_key_strips_whitespace_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only env keys should fall back to credentials file."""
    monkeypatch.setenv("DOTTXT_API_KEY", "   ")
    monkeypatch.setattr(cli_module, "_read_credentials_api_key", lambda: "file-key")

    assert cli_module._resolve_api_key() == "file-key"


def test_resolve_default_model_prefers_non_empty_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DOTTXT_MODEL should resolve as the default model."""
    monkeypatch.setenv("DOTTXT_MODEL", "openai/gpt-4.1-mini")

    model = cli_module._resolve_default_model()

    assert model == "openai/gpt-4.1-mini"


def test_resolve_default_model_ignores_blank_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank DOTTXT_MODEL should resolve to no default model."""
    monkeypatch.setenv("DOTTXT_MODEL", "   ")

    model = cli_module._resolve_default_model()

    assert model is None


def test_write_credentials_tolerates_chmod_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Credential writes should succeed even if chmod is unsupported."""

    def _raise_oserror(*_args: object, **_kwargs: object) -> None:
        raise OSError("no chmod")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli_module.os, "chmod", _raise_oserror)

    cli_module._write_credentials("test-key")

    credentials_file = tmp_path / "dottxt" / "credentials.json"
    assert credentials_file.exists()


def test_jsonable_handles_model_dump_branch() -> None:
    """_jsonable should use model_dump() when available."""

    class ModelDumpObject:
        def model_dump(self) -> dict[str, int]:
            return {"x": 1}

    assert cli_module._jsonable(ModelDumpObject()) == {"x": 1}


def test_jsonable_handles_dict_method_branch() -> None:
    """_jsonable should use dict() when model_dump() is absent."""

    class DictObject:
        def dict(self) -> dict[str, int]:
            return {"y": 2}

    assert cli_module._jsonable(DictObject()) == {"y": 2}


def test_jsonable_handles_to_dict_method_branch() -> None:
    """_jsonable should use to_dict() when present."""

    class ToDictObject:
        def to_dict(self) -> dict[str, int]:
            return {"z": 3}

    assert cli_module._jsonable(ToDictObject()) == {"z": 3}


def test_jsonable_handles_dunder_dict_branch() -> None:
    """_jsonable should serialize plain objects via __dict__."""

    class PlainObject:
        def __init__(self) -> None:
            self.value = 4

    assert cli_module._jsonable(PlainObject()) == {"value": 4}


def test_display_path_keeps_non_home_paths_literal() -> None:
    """Paths outside home should be returned without tilde rewriting."""
    target = Path("/tmp/credentials.json")

    assert cli_module._display_path(target) == "/tmp/credentials.json"


def test_close_client_noop_when_none() -> None:
    """_close_client should no-op for None."""
    cli_module._close_client(None)


def test_close_client_closes_instance() -> None:
    """_close_client should call close() when client exists."""

    class Closable:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    instance = Closable()
    cli_module._close_client(instance)  # type: ignore[arg-type]
    assert instance.closed is True


def test_models_handles_constructor_error(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models should surface SDK constructor failures cleanly."""

    class FailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_module, "DotTxt", FailingDotTxt)

    result = runner.invoke(main, ["models"])

    assert result.exit_code == 1
    assert "Unable to list models: boom" in result.output


def test_models_handles_non_list_data_payload(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models should normalize non-list data payloads to an empty list."""

    class NonListDotTxt:
        def __init__(self, *, api_key: str) -> None:
            self.models = SimpleNamespace(
                list=lambda: SimpleNamespace(data={"id": "m"})
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_module, "DotTxt", NonListDotTxt)

    result = runner.invoke(main, ["--json", "models"])

    assert result.exit_code == 0
    assert result.output.strip() == "[]"


def test_validate_api_key_returns_empty_models_for_non_list_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation should return an empty model list for non-list data payloads."""

    class NonListDotTxt:
        def __init__(self, *, api_key: str) -> None:
            self.models = SimpleNamespace(
                list=lambda: SimpleNamespace(data={"id": "m"})
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "DotTxt", NonListDotTxt)

    result = cli_module._validate_api_key("test-key", json_mode=False)

    assert result == []


def test_models_supports_missing_author_field(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models should allow entries without author and still list IDs."""

    class PartialAuthorDotTxt:
        def __init__(self, *, api_key: str) -> None:
            self.models = SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[
                        {"id": "openai/gpt-oss-20b", "author": "openai"},
                        {"id": "vendor/model-no-author"},
                    ]
                )
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_module, "DotTxt", PartialAuthorDotTxt)

    default_result = runner.invoke(main, ["models"])
    filtered_result = runner.invoke(main, ["models", "--author", "openai"])

    assert default_result.exit_code == 0
    assert default_result.output.strip().splitlines() == [
        "openai/gpt-oss-20b",
        "vendor/model-no-author",
    ]
    assert filtered_result.exit_code == 0
    assert filtered_result.output.strip().splitlines() == ["openai/gpt-oss-20b"]


def test_generate_handles_constructor_error(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generate should surface SDK constructor failures cleanly."""

    class FailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            raise RuntimeError("boom")

    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"type":"object"}', encoding="utf-8")
    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_module, "DotTxt", FailingDotTxt)

    result = runner.invoke(
        main,
        ["generate", "-m", "openai/gpt-oss-20b", "-s", str(schema_file), "hello"],
    )

    assert result.exit_code == 1
    assert "Generation failed: boom" in result.output


def test_validate_api_key_handles_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auth failures should return a specific invalid-key error."""

    class FakeAuthError(Exception):
        pass

    class AuthFailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            self.models = SimpleNamespace(list=self._list_models)

        def _list_models(self) -> object:
            raise FakeAuthError()

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "AuthenticationError", FakeAuthError)
    monkeypatch.setattr(cli_module, "DotTxt", AuthFailingDotTxt)

    with pytest.raises(click.ClickException, match="Invalid API key"):
        cli_module._validate_api_key("bad-key", json_mode=False)


def test_validate_api_key_handles_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection failures should return a retryable connectivity error."""

    class FakeConnectionError(Exception):
        pass

    class ConnectionFailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            self.models = SimpleNamespace(list=self._list_models)

        def _list_models(self) -> object:
            raise FakeConnectionError()

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "APIConnectionError", FakeConnectionError)
    monkeypatch.setattr(cli_module, "DotTxt", ConnectionFailingDotTxt)

    with pytest.raises(click.ClickException, match="Unable to reach the API"):
        cli_module._validate_api_key("bad-key", json_mode=False)


def test_validate_api_key_handles_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate-limit failures should return a specific retry message."""

    class FakeRateLimitError(Exception):
        pass

    class RateLimitFailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            self.models = SimpleNamespace(list=self._list_models)

        def _list_models(self) -> object:
            raise FakeRateLimitError()

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "RateLimitError", FakeRateLimitError)
    monkeypatch.setattr(cli_module, "DotTxt", RateLimitFailingDotTxt)

    with pytest.raises(click.ClickException, match="API rate limit reached"):
        cli_module._validate_api_key("bad-key", json_mode=False)


def test_validate_api_key_handles_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP status failures should include the returned status code."""

    class FakeStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"status={status_code}")
            self.status_code = status_code

    class StatusFailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            self.models = SimpleNamespace(list=self._list_models)

        def _list_models(self) -> object:
            raise FakeStatusError(503)

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "APIStatusError", FakeStatusError)
    monkeypatch.setattr(cli_module, "DotTxt", StatusFailingDotTxt)

    with pytest.raises(click.ClickException, match="status 503"):
        cli_module._validate_api_key("bad-key", json_mode=False)


def test_generate_surfaces_api_status_code_and_body_message(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generate should include status code and API body detail on status errors."""

    class FakeStatusError(Exception):
        def __init__(self, status_code: int, body: dict[str, object]) -> None:
            super().__init__("status error")
            self.status_code = status_code
            self.body = body

    class FailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            return None

        def generate(self, **_kwargs: object) -> object:
            raise FakeStatusError(
                502,
                {"error": {"message": "bad gateway from upstream model"}},
            )

        def close(self) -> None:
            return None

    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"type":"object"}', encoding="utf-8")
    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_module, "DotTxt", FailingDotTxt)
    monkeypatch.setattr(cli_module, "APIStatusError", FakeStatusError)

    result = runner.invoke(
        main,
        ["--verbose", "generate", "-m", "test-model", "-s", str(schema_file), "hello"],
    )

    assert result.exit_code == 1
    assert "Generation failed with API status 502." in result.output
    assert "bad gateway from upstream model" in result.output
    assert "[verbose] Generation request failed." in result.output


def test_generate_surfaces_connection_failures(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generate should map API connection exceptions to a clear message."""

    class FakeConnectionError(Exception):
        pass

    class FailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            return None

        def generate(self, **_kwargs: object) -> object:
            raise FakeConnectionError("network down")

        def close(self) -> None:
            return None

    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"type":"object"}', encoding="utf-8")
    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_module, "DotTxt", FailingDotTxt)
    monkeypatch.setattr(cli_module, "APIConnectionError", FakeConnectionError)

    result = runner.invoke(
        main,
        ["generate", "-m", "test-model", "-s", str(schema_file), "hello"],
    )

    assert result.exit_code == 1
    assert "Unable to reach API: network down" in result.output


def test_generate_surfaces_authentication_failures(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generate should map authentication exceptions to auth-specific errors."""

    class FakeAuthError(Exception):
        pass

    class FailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            return None

        def generate(self, **_kwargs: object) -> object:
            raise FakeAuthError("bad token")

        def close(self) -> None:
            return None

    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"type":"object"}', encoding="utf-8")
    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_module, "DotTxt", FailingDotTxt)
    monkeypatch.setattr(cli_module, "AuthenticationError", FakeAuthError)

    result = runner.invoke(
        main,
        ["generate", "-m", "test-model", "-s", str(schema_file), "hello"],
    )

    assert result.exit_code == 1
    assert "Unable to authenticate API key: bad token" in result.output


def test_generate_surfaces_rate_limit_failures(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generate should map rate limit exceptions to clear CLI output."""

    class FakeRateLimitError(Exception):
        pass

    class FailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            return None

        def generate(self, **_kwargs: object) -> object:
            raise FakeRateLimitError("too many requests")

        def close(self) -> None:
            return None

    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"type":"object"}', encoding="utf-8")
    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_module, "DotTxt", FailingDotTxt)
    monkeypatch.setattr(cli_module, "RateLimitError", FakeRateLimitError)

    result = runner.invoke(
        main,
        ["generate", "-m", "test-model", "-s", str(schema_file), "hello"],
    )

    assert result.exit_code == 1
    assert "Rate limited by API: too many requests" in result.output


def test_status_error_message_supports_string_body_without_status() -> None:
    """Status error helper should support plain string bodies."""

    class FakeStatusError(Exception):
        def __init__(self, body: str) -> None:
            self.status_code = None
            self.body = body

    message = cli_module._status_error_message(
        "Generation",
        cast(APIStatusError, FakeStatusError("oops")),
    )

    assert message == "Generation failed with API status error. oops"


def test_status_error_message_falls_back_to_serialized_body_dict() -> None:
    """Status error helper should serialize dict payloads without message field."""

    class FakeStatusError(Exception):
        def __init__(self, status_code: int, body: dict[str, object]) -> None:
            self.status_code = status_code
            self.body = body

    message = cli_module._status_error_message(
        "Generation",
        cast(APIStatusError, FakeStatusError(502, {"foo": "bar"})),
    )

    assert message == 'Generation failed with API status 502. {"foo":"bar"}'


def test_status_error_message_handles_blank_string_body_without_detail() -> None:
    """Status error helper should omit detail when body is blank text."""

    class FakeStatusError(Exception):
        def __init__(self, status_code: int, body: str) -> None:
            self.status_code = status_code
            self.body = body

    message = cli_module._status_error_message(
        "Generation",
        cast(APIStatusError, FakeStatusError(503, "")),
    )

    assert message == "Generation failed with API status 503."


def test_status_error_message_falls_back_when_error_message_is_blank() -> None:
    """Blank nested error messages should fall back to serialized payload."""

    class FakeStatusError(Exception):
        def __init__(self, status_code: int, body: dict[str, object]) -> None:
            self.status_code = status_code
            self.body = body

    message = cli_module._status_error_message(
        "Generation",
        cast(APIStatusError, FakeStatusError(502, {"error": {"message": "   "}})),
    )

    assert (
        message == 'Generation failed with API status 502. {"error":{"message":"   "}}'
    )
