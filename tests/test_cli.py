"""Tests for dottxt CLI behavior."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner, Result

from dottxt import cli as cli_module
from dottxt.cli import main

VALID_SCHEMA = '{"type":"object"}'
STUB_RESULT = {"status": "stub", "message": "SDK call not wired yet."}
MODEL_DATA = [
    {
        "id": "openai/gpt-oss-20b",
        "author": "openai",
        "price_input": 0.2,
        "price_output": 0.8,
        "quality_index": 82,
    },
    {
        "id": "openai/gpt-4.1-mini",
        "author": "openai",
        "price_input": 0.4,
        "price_output": 1.2,
        "quality_index": 88,
    },
    {
        "id": "anthropic/claude-sonnet-4",
        "author": "anthropic",
        "price_input": 0.6,
        "price_output": 1.8,
        "quality_index": 90,
    },
]


class FakeDotTxt:
    """Stub DotTxt client used by CLI tests."""

    init_api_keys: list[str] = []
    generate_calls: list[dict[str, object]] = []
    models_calls: int = 0
    close_calls: int = 0

    def __init__(self, *, api_key: str) -> None:
        """Initialize fake client with provided key."""
        self.__class__.init_api_keys.append(api_key)
        self.models = SimpleNamespace(list=self._list_models)

    def _list_models(self) -> object:
        """Return fake model listing response."""
        self.__class__.models_calls += 1
        return SimpleNamespace(data=MODEL_DATA)

    def generate(
        self,
        *,
        model: str,
        response_format: str,
        input: str,
    ) -> object:
        """Record generate arguments and return deterministic payload."""
        self.__class__.generate_calls.append(
            {
                "model": model,
                "response_format": response_format,
                "input": input,
            }
        )
        return STUB_RESULT

    def close(self) -> None:
        """Record close calls."""
        self.__class__.close_calls += 1


@pytest.fixture
def runner() -> CliRunner:
    """Return a CLI test runner."""
    return CliRunner()


@pytest.fixture(autouse=True)
def patch_dotxt_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch CLI SDK usage with a deterministic fake client."""
    FakeDotTxt.init_api_keys.clear()
    FakeDotTxt.generate_calls.clear()
    FakeDotTxt.models_calls = 0
    FakeDotTxt.close_calls = 0
    monkeypatch.delenv("DOTTXT_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "DotTxt", FakeDotTxt)
    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: "test-key")


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    """Create and return a valid schema file."""
    return _create_schema(tmp_path, content=VALID_SCHEMA)


def _create_schema(tmp_path: Path, *, content: str, name: str = "schema.json") -> Path:
    """Create a schema file under tmp_path."""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _parse_json_output(output: str) -> object:
    """Parse JSON from output that may include prompt prefixes."""
    lines = output.strip().splitlines()
    for idx in range(len(lines)):
        candidate = "\n".join(lines[idx:])
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"No JSON payload found in output: {output!r}")


def _invoke(
    runner: CliRunner,
    args: Sequence[str],
    *,
    input_text: str | None = None,
) -> Result:
    """Invoke the CLI with optional stdin input."""
    return runner.invoke(main, list(args), input=input_text)


def _invoke_json(
    runner: CliRunner,
    args: Sequence[str],
    *,
    input_text: str | None = None,
    expected_exit_code: int = 0,
) -> object:
    """Invoke CLI and parse JSON output."""
    result = _invoke(runner, args, input_text=input_text)
    assert result.exit_code == expected_exit_code
    return _parse_json_output(result.output)


def test_version_json_mode_returns_raw_value(runner: CliRunner) -> None:
    """Version should return a raw JSON string value."""
    payload = _invoke_json(runner, ["--json", "--version"])
    assert isinstance(payload, str)


def test_main_without_subcommand_shows_help(runner: CliRunner) -> None:
    """Root command without subcommand should show help and exit zero."""
    result = _invoke(runner, [])

    assert result.exit_code == 0
    assert "Usage: main [OPTIONS] COMMAND [ARGS]..." in result.output


def test_login_fails_without_api_key(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Login should fail when API key is omitted."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = _invoke(runner, ["login"])

    assert result.exit_code == 1
    assert "No API key provided" in result.output


def test_login_persists_credentials(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Login should persist credentials from stdin."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = _invoke(runner, ["login"], input_text="test-key\n")

    assert result.exit_code == 0
    assert "Logged in." in result.output
    credentials_file = tmp_path / "dottxt" / "credentials.json"
    assert credentials_file.exists()
    payload = json.loads(credentials_file.read_text(encoding="utf-8"))
    assert payload["api_key"] == "test-key"
    assert FakeDotTxt.models_calls == 1
    assert FakeDotTxt.close_calls == 1


def test_login_verbose_prints_full_payload(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """JSON login should print payload details while verbose stays orthogonal."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    payload = _invoke_json(
        runner,
        ["--json", "--verbose", "login"],
        input_text="test-key\n",
    )

    assert isinstance(payload, dict)
    assert payload["status"] == "ok"
    assert payload["source"] == "stdin"
    assert payload["models_available"] == 3


def test_login_uses_env_api_key_when_available(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Login should use DOTTXT_API_KEY when set."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("DOTTXT_API_KEY", "env-key")

    payload = _invoke_json(runner, ["--json", "login"])

    assert isinstance(payload, dict)
    assert payload["status"] == "ok"
    assert payload["source"] == "env"
    credentials_file = tmp_path / "dottxt" / "credentials.json"
    stored = json.loads(credentials_file.read_text(encoding="utf-8"))
    assert stored["api_key"] == "env-key"


def test_login_empty_env_falls_back_to_stdin(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Empty env value should not block stdin credential input."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("DOTTXT_API_KEY", "   ")

    payload = _invoke_json(
        runner,
        ["--json", "login"],
        input_text="stdin-key\n",
    )

    assert isinstance(payload, dict)
    assert payload["source"] == "stdin"
    credentials_file = tmp_path / "dottxt" / "credentials.json"
    stored = json.loads(credentials_file.read_text(encoding="utf-8"))
    assert stored["api_key"] == "stdin-key"


def test_login_interactive_prompt_used_when_tty(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Login should prompt interactively when stdin is a TTY."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    monkeypatch.setattr(cli_module, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(
        cli_module.click,
        "prompt",
        lambda *_args, **_kwargs: "prompt-key",
    )

    payload = _invoke_json(runner, ["--json", "login"])

    assert isinstance(payload, dict)
    assert payload["status"] == "ok"
    assert payload["source"] == "prompt"
    credentials_file = tmp_path / "dottxt" / "credentials.json"
    stored = json.loads(credentials_file.read_text(encoding="utf-8"))
    assert stored["api_key"] == "prompt-key"


def test_login_stores_credentials_with_restricted_permissions(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Credentials file should be created with owner-only permissions."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = _invoke(runner, ["login"], input_text="test-key\n")

    assert result.exit_code == 0
    credentials_file = tmp_path / "dottxt" / "credentials.json"
    permissions = credentials_file.stat().st_mode & 0o777
    assert permissions == 0o600


def test_login_fails_when_api_key_authentication_fails(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Login should fail and avoid writes when API key validation fails."""

    class FailingDotTxt:
        def __init__(self, *, api_key: str) -> None:
            self.models = SimpleNamespace(list=self._list_models)

        def _list_models(self) -> object:
            raise RuntimeError("unauthorized")

        def close(self) -> None:
            return None

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli_module, "DotTxt", FailingDotTxt)
    result = _invoke(runner, ["login"], input_text="bad-key\n")

    assert result.exit_code == 1
    assert "Unable to authenticate API key" in result.output
    credentials_file = tmp_path / "dottxt" / "credentials.json"
    assert not credentials_file.exists()


def test_logout_verbose_prints_full_payload(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """JSON logout should print payload details while verbose stays orthogonal."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    payload = _invoke_json(runner, ["--json", "--verbose", "logout"])

    assert isinstance(payload, dict)
    assert payload["status"] == "ok"
    assert payload["removed"] is False


def test_logout_default_removes_credentials_and_prints_message(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default logout should remove credentials and print a short message."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    login_result = _invoke(runner, ["login"], input_text="test-key\n")
    logout_result = _invoke(runner, ["logout"])

    assert login_result.exit_code == 0
    assert logout_result.exit_code == 0
    assert logout_result.output.strip() == "Logged out."
    credentials_file = tmp_path / "dottxt" / "credentials.json"
    assert not credentials_file.exists()


def test_models_default_print_only_model_ids(runner: CliRunner) -> None:
    """Default models output should include only model ids."""
    result = _invoke(runner, ["models", "--author", "openai"])

    assert result.exit_code == 0
    assert result.output.strip().splitlines() == [
        "openai/gpt-oss-20b",
        "openai/gpt-4.1-mini",
    ]
    assert FakeDotTxt.models_calls == 1
    assert FakeDotTxt.init_api_keys == ["test-key"]
    assert FakeDotTxt.close_calls == 1


def test_models_marks_env_default_model(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models output should mark DOTTXT_MODEL as the default."""
    monkeypatch.setenv("DOTTXT_MODEL", "openai/gpt-4.1-mini")

    result = _invoke(runner, ["models", "--author", "openai"])

    assert result.exit_code == 0
    assert result.output.strip().splitlines() == [
        "openai/gpt-oss-20b",
        "openai/gpt-4.1-mini * (default)",
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["--json", "models", "--author", "openai"],
        ["--json", "--verbose", "models", "--author", "openai"],
    ],
)
def test_models_verbose_and_json_return_full_records(
    runner: CliRunner,
    args: list[str],
) -> None:
    """Verbose/json models output should include full records."""
    payload = _invoke_json(runner, args)

    assert isinstance(payload, list)
    assert len(payload) == 2
    assert all(item["author"] == "openai" for item in payload)


def test_models_json_without_filter_returns_all_records(runner: CliRunner) -> None:
    """JSON models output without author filter should return all records."""
    payload = _invoke_json(runner, ["--json", "models"])

    assert isinstance(payload, list)
    assert len(payload) == 3


def test_models_fails_without_available_api_key(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models should fail when no API key can be resolved."""
    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: None)

    result = _invoke(runner, ["models"])

    assert result.exit_code == 1
    assert "No API key available" in result.output


@pytest.mark.parametrize(
    ("args", "expect_full_payload"),
    [
        (["generate", "-s", "{schema}", "x y"], False),
        (["--verbose", "generate", "-s", "{schema}", "x y"], False),
        (["--json", "generate", "-s", "{schema}", "x y"], True),
    ],
)
def test_generate_output_modes(
    runner: CliRunner,
    schema_file: Path,
    args: list[str],
    expect_full_payload: bool,
) -> None:
    """Generate should switch between result-only and full payload by mode."""
    resolved_args = [arg.replace("{schema}", str(schema_file)) for arg in args]
    command_index = resolved_args.index("generate")
    resolved_args[command_index + 1 : command_index + 1] = [
        "-m",
        "openai/gpt-oss-20b",
    ]
    payload = _invoke_json(runner, resolved_args)

    assert isinstance(payload, dict)
    if expect_full_payload:
        assert payload["prompt"] == "x y"
        assert payload["result"] == STUB_RESULT
    else:
        assert payload == STUB_RESULT
    assert FakeDotTxt.generate_calls[-1]["model"] == "openai/gpt-oss-20b"
    assert FakeDotTxt.generate_calls[-1]["input"] == "x y"
    assert FakeDotTxt.close_calls >= 1


def test_generate_prompt_precedence_and_stdin_fallback(
    runner: CliRunner,
    schema_file: Path,
) -> None:
    """Generate should prefer positional prompt and otherwise read stdin."""
    from_flag_payload = _invoke_json(
        runner,
        [
            "--json",
            "generate",
            "-m",
            "openai/gpt-oss-20b",
            "-s",
            str(schema_file),
            "from flag",
        ],
        input_text="from-stdin",
    )
    from_stdin_payload = _invoke_json(
        runner,
        ["--json", "generate", "-m", "openai/gpt-oss-20b", "-s", str(schema_file)],
        input_text="from-stdin",
    )

    assert isinstance(from_flag_payload, dict)
    assert isinstance(from_stdin_payload, dict)
    assert from_flag_payload["prompt"] == "from flag"
    assert from_stdin_payload["prompt"] == "from-stdin"


def test_generate_prompt_positional_is_literal_text(
    runner: CliRunner,
    schema_file: Path,
) -> None:
    """Generate should treat positional prompt as literal text."""
    payload = _invoke_json(
        runner,
        [
            "--json",
            "generate",
            "-m",
            "openai/gpt-oss-20b",
            "-s",
            str(schema_file),
            "generate",
        ],
    )

    assert isinstance(payload, dict)
    assert payload["prompt"] == "generate"


def test_generate_reads_schema_text_and_forwards_to_sdk(
    runner: CliRunner,
    schema_file: Path,
) -> None:
    """Generate should pass schema file text to the SDK response format."""
    schema_text = schema_file.read_text(encoding="utf-8")

    result = _invoke(
        runner,
        ["generate", "-m", "openai/gpt-oss-20b", "-s", str(schema_file), "x y"],
    )

    assert result.exit_code == 0
    assert FakeDotTxt.generate_calls[-1]["response_format"] == schema_text


def test_generate_uses_model_from_env_when_flag_not_provided(
    runner: CliRunner,
    schema_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate should read DOTTXT_MODEL when --model is omitted."""
    monkeypatch.setenv("DOTTXT_MODEL", "openai/gpt-4.1-mini")

    result = _invoke(runner, ["generate", "-s", str(schema_file), "x y"])

    assert result.exit_code == 0
    assert FakeDotTxt.generate_calls[-1]["model"] == "openai/gpt-4.1-mini"


def test_generate_flag_model_overrides_env_model(
    runner: CliRunner,
    schema_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate should prioritize --model over DOTTXT_MODEL."""
    monkeypatch.setenv("DOTTXT_MODEL", "openai/gpt-4.1-mini")

    result = _invoke(
        runner,
        ["generate", "-m", "openai/gpt-oss-20b", "-s", str(schema_file), "x y"],
    )

    assert result.exit_code == 0
    assert FakeDotTxt.generate_calls[-1]["model"] == "openai/gpt-oss-20b"


def test_generate_stdin_supports_file_input(
    runner: CliRunner,
    schema_file: Path,
    tmp_path: Path,
) -> None:
    """Generate should support file-based prompt input via stdin."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("from-file", encoding="utf-8")

    payload = _invoke_json(
        runner,
        ["--json", "generate", "-m", "openai/gpt-oss-20b", "-s", str(schema_file)],
        input_text=prompt_file.read_text(encoding="utf-8"),
    )

    assert isinstance(payload, dict)
    assert payload["prompt"] == "from-file"


def test_generate_missing_prompt_without_stdin_returns_error(
    runner: CliRunner,
    schema_file: Path,
) -> None:
    """Generate should fail when prompt is missing and stdin is empty."""
    result = _invoke(
        runner,
        ["generate", "-m", "openai/gpt-oss-20b", "-s", str(schema_file)],
    )

    assert result.exit_code == 1
    assert "No prompt provided" in result.output


def test_generate_fails_without_available_api_key(
    runner: CliRunner,
    schema_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate should fail when no API key can be resolved."""
    monkeypatch.setattr(cli_module, "_resolve_api_key", lambda: None)

    result = _invoke(
        runner,
        ["generate", "-m", "openai/gpt-oss-20b", "-s", str(schema_file), "x y"],
    )

    assert result.exit_code == 1
    assert "No API key available" in result.output


def test_generate_reports_targeted_error_for_unavailable_model(
    runner: CliRunner,
    schema_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate should provide targeted guidance when model is unavailable."""

    class ModelUnavailableDotTxt:
        def __init__(self, *, api_key: str) -> None:
            return None

        def generate(
            self,
            *,
            model: str,
            response_format: str,
            input: str,
        ) -> object:
            raise RuntimeError(f"Model '{model}' not available for this key")

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "DotTxt", ModelUnavailableDotTxt)

    result = _invoke(
        runner,
        ["generate", "-m", "openai/gpt-oss-20b", "-s", str(schema_file), "x y"],
    )

    assert result.exit_code == 1
    assert "is not available for this API key" in result.output
    assert "Run 'dottxt models' to choose an available model" in result.output


def test_generate_help_mentions_optional_prompt_argument(runner: CliRunner) -> None:
    """Generate help should document that prompt can be supplied via stdin."""
    result = _invoke(runner, ["generate", "--help"])

    assert result.exit_code == 0
    assert "Usage: main generate [OPTIONS] [PROMPT]" in result.output
    assert "required unless stdin is piped" in result.output


def test_generate_error_paths_and_usage_codes(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Generate should map validation and usage failures to expected exit codes."""
    invalid_schema = _create_schema(tmp_path, name="invalid.json", content='{"type": ')

    invalid_json = _invoke(
        runner,
        ["generate", "-m", "openai/gpt-oss-20b", "-s", str(invalid_schema), "x y"],
    )
    missing_schema_usage = _invoke(
        runner,
        ["generate", "-m", "openai/gpt-oss-20b", "x y"],
    )
    usage_error = _invoke(runner, ["generate", "--unknown-flag"])

    assert invalid_json.exit_code == 1
    assert "not valid JSON" in invalid_json.output
    assert missing_schema_usage.exit_code == 2
    assert "Missing option '-s' / '--schema'" in missing_schema_usage.output
    assert usage_error.exit_code == 2


def test_generate_requires_model_when_env_default_missing(
    runner: CliRunner,
    schema_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate should require --model when DOTTXT_MODEL is unset."""
    monkeypatch.delenv("DOTTXT_MODEL", raising=False)

    result = _invoke(runner, ["generate", "-s", str(schema_file), "x y"])

    assert result.exit_code == 2
    assert "Missing option '-m' / '--model'" in result.output


def test_json_error_output_is_machine_readable(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """JSON mode errors should be emitted as a single JSON payload."""
    missing_schema = tmp_path / "missing.json"
    result = _invoke(
        runner,
        [
            "--json",
            "generate",
            "-m",
            "openai/gpt-oss-20b",
            "-s",
            str(missing_schema),
            "x y",
        ],
    )

    assert result.exit_code == 1
    assert "Error:" not in result.output
    payload = _parse_json_output(result.output)
    assert isinstance(payload, dict)
    assert payload == {
        "error": {"message": f"Schema file not found: {missing_schema}"},
    }


def test_read_stdin_prompt_returns_empty_string_for_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_read_stdin_prompt should return empty string when stdin is a TTY."""

    class _TTYStream:
        def isatty(self) -> bool:
            return True

        def read(self) -> str:
            return "ignored"

    monkeypatch.setattr(
        cli_module.click,
        "get_text_stream",
        lambda _name: _TTYStream(),
    )
    assert cli_module._read_stdin_prompt() == ""


def test_display_path_rewrites_home_prefix() -> None:
    """_display_path should shorten home-prefixed paths with ~."""
    home = Path.home()
    target = home / "dottxt" / "credentials.json"

    assert cli_module._display_path(target) == "~/dottxt/credentials.json"


def test_emit_verbose_without_data_prints_single_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_emit_verbose should print only message text when data is absent."""
    context = cli_module.click.Context(main, obj={"verbose": True})

    cli_module._emit_verbose(context, "hello world")

    captured = capsys.readouterr()
    assert captured.err.strip() == "[verbose] hello world"


def test_emit_verbose_non_dict_data_is_compact_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_emit_verbose should serialize non-dict payloads as compact JSON."""
    context = cli_module.click.Context(main, obj={"verbose": True})

    cli_module._emit_verbose(context, "payload", data=[1, 2, 3])

    captured = capsys.readouterr()
    assert captured.err.strip() == "[verbose] payload [1,2,3]"
