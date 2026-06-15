# Need to Include in Docs

Production-readiness work completed in this pass. Items below introduce or change
user-facing behavior and should be documented.

## Type checking (PEP 561)
- The package now ships a `py.typed` marker. Downstream `mypy`/`pyright` will
  type-check against Agentflow's annotations.

## Configurable LLM call timeout
- All LLM clients now apply a default request timeout (`600s`) so a stalled
  provider cannot hang a run indefinitely.
- Override globally via the `AGENTFLOW_LLM_TIMEOUT` environment variable
  (seconds), or programmatically:
  - `from agentflow.core.llm import set_default_llm_timeout, get_default_llm_timeout, DEFAULT_LLM_TIMEOUT_SECONDS`
  - `set_default_llm_timeout(120.0)` / `set_default_llm_timeout(None)` to reset.
- An explicit per-client `timeout=` still takes precedence.

## CompiledGraph async context manager
- `CompiledGraph` supports `async with`:
  ```python
  async with await build_and_compile_graph() as graph:
      await graph.ainvoke(input_data)
  # aclose() runs automatically on exit, even if the body raises
  ```
- `aclose()` is now idempotent (second call returns `{"status": "already_closed"}`).

## Circuit breaker for LLM calls (opt-in)
- Complements retry + `fallback_models`: once a `(provider, model)` fails
  `circuit_breaker_threshold` times in a row, its circuit opens and that target
  is skipped (straight to the next fallback) for `circuit_breaker_reset_timeout`
  seconds, instead of being retried on every call.
- Configure via `RetryConfig`:
  - `circuit_breaker_enabled: bool = False`
  - `circuit_breaker_threshold: int = 5`
  - `circuit_breaker_reset_timeout: float = 30.0`

## Secret redaction for logs
- New helpers in `agentflow.utils`:
  - `mask_secrets(text)` — redacts API keys, `Bearer` tokens, `key=value`
    secrets, and signed-URL credential query params.
  - `SecretRedactionFilter` — a `logging.Filter`; add it to a handler to cover
    all loggers that propagate to it.
  - `install_secret_redaction(logger_name="agentflow")` — convenience installer.

## ConsolePublisher logging option
- `ConsolePublisher` is a dev/debug, opt-in publisher (use a real transport in
  production). It writes to stdout by default; pass `{"use_logger": True}` to
  route events through the `agentflow.publisher` logger instead of stdout.

## Project / repo
- Dependencies now have version bounds (e.g. `pydantic>=2,<3`).
- mypy runs in pre-commit/CI (phased adoption; see `CONTRIBUTING.md`).
- Test coverage gate raised to 80%.
- Added `SECURITY.md` and `CONTRIBUTING.md`.
- Added Dependabot config and a CodeQL workflow.
