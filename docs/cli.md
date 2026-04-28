# dottxt CLI

CLI behavior backed by the dottxt SDK.

## Global Flags

- `-h, --help`: Show help
- `--version`: Print version
- `--json`: Emit machine-readable JSON
- `--verbose`: Emit expanded diagnostics on `stderr`

`--json` error format:

```json
{"error": {"message": "..."}}
```

In `--json` mode, stdout is machine-readable. Failures emit only that JSON
object and exit with code `1`.
In `--json` mode, success JSON is printed to stdout and error JSON is printed
to stderr.

## Exit Codes

- `0`: Success
- `1`: Runtime/config/input errors
- `2`: Usage/argument errors from `click`

## Commands

### `dottxt login`

Store API credentials.

- Resolution order: `DOTTXT_API_KEY`, then stdin (non-TTY), then interactive
  prompt (TTY only)
- Stdin example: `dottxt login < key.txt`
- Fails when no key is available from any source

### `dottxt logout`

Delete locally stored credentials.

### `dottxt models`

List available models from the API.

- Default: one model ID per line (`* (default)` appears when `DOTTXT_MODEL` matches)
- `--json`: full records
- `--verbose`: debug diagnostics on `stderr` (stdout unchanged)
- `--author NAME`: filter by model author
- Set a shell default model: `export DOTTXT_MODEL=<model-id>`

### `dottxt generate`

Run one generation.

- `-m, --model TEXT`: model id (required unless `DOTTXT_MODEL` is set)
- `DOTTXT_MODEL`: optional default model id for `generate` (`--model` overrides)
- `-s, --schema FILE`: schema file path (required)
- `[PROMPT]`: literal prompt text

Prompt rules:

- If `[PROMPT]` is provided, it is used.
- Otherwise, stdin is read.
- Command fails if neither positional prompt nor stdin is provided.

Output rules:

- `--json`: full generation payload on stdout
- `--verbose`: debug diagnostics on `stderr` (stdout unchanged)
- If the selected model is unavailable for your key, generate returns a
  targeted error with guidance to run `dottxt models` and set `DOTTXT_MODEL`
  or pass `--model`
