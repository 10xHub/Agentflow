# Contributing to Agentflow

Thanks for your interest in improving `10xscale-agentflow`. This guide covers the
core Python framework that lives in this folder. For the API server, TypeScript
client, docs, or playground, see the `CONTRIBUTING`/`CLAUDE.md` in their
respective packages.

- Package (PyPI): `10xscale-agentflow`
- Requires: Python >= 3.12
- The importable package is the nested `agentflow/` directory; this folder is the
  repo root for the core library.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

We use [`uv`](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
# from this folder (the core library root)
uv sync --dev          # create .venv and install the package + dev tools
uv run pre-commit install   # enable the git hooks (optional but recommended)
```

If you work on optional subsystems, install the matching extras, e.g.:

```bash
uv pip install -e ".[google-genai,openai,mcp,pg_checkpoint]"
```

## Before you open a pull request

Run the same checks CI runs. All must pass:

```bash
uv run pre-commit run --all-files     # ruff format + lint, bandit, mypy, hooks
uv run pytest --cov --cov-branch      # tests + coverage gate (>= 80%)
```

You can also run pieces individually:

```bash
uv run ruff check . && uv run ruff format .
uv run mypy agentflow/
uv run pytest tests/graph             # one area
```

### What the gates enforce

- **Formatting & linting:** `ruff` (line length 100, target py312). Most issues
  are auto-fixed by `ruff format` / `ruff check --fix`.
- **Types:** `mypy` runs in pre-commit. The codebase is on *phased* typing: a set
  of modules with pre-existing errors is listed under `[[tool.mypy.overrides]]`
  in `pyproject.toml` with `ignore_errors = true`. New code is type-checked.
  Improving a listed module's types and removing it from that list is a welcome
  contribution; please don't add new modules to it.
- **Security:** `bandit`.
- **Coverage:** `pytest` fails under 80% line coverage. New code needs tests.

## Tests

- Tests live in `tests/`, mirroring the package layout (`graph/`, `state/`,
  `storage/`, `publisher/`, `prebuilt/`, `evaluation/`, `testing/`, plus
  `chaos/`, `benchmarks/`, `integration/`).
- Markers: `asyncio`, `integration` (needs real databases — Redis/Postgres),
  `slow`. Integration tests are skipped unless their backends are available.
- Prefer the in-repo test helpers in `agentflow.qa.testing` (`TestAgent`,
  `MockMCPClient`, `MockToolRegistry`) to exercise graphs without live LLM calls.

## Import paths (read this before referencing symbols)

The package is organised into `core/`, `storage/`, `runtime/`, `qa/`. There are
**no** top-level `agentflow.graph` / `agentflow.state` / `agentflow.checkpointer`
shims — use the canonical paths:

```python
from agentflow.core.graph import StateGraph, Agent, ToolNode, CompiledGraph
from agentflow.core.state import AgentState, Message
from agentflow.core.llm import call_llm, create_llm_client, detect_provider
from agentflow.storage.checkpointer import InMemoryCheckpointer, PgCheckpointer
```

`examples/` uses current import paths and is the most reliable usage reference.

## Optional dependencies

Provider SDKs (OpenAI, Google GenAI), MCP, Postgres, Redis, Qdrant, Mem0, Kafka,
RabbitMQ, OTEL, and a2a are all **extras**. Guard their imports inside the
functions that need them so the core package never hard-imports an optional
dependency. See `agentflow/core/llm/client_factory.py` for the pattern.

## Commit and PR conventions

- Use clear, conventional-style commit subjects (`feat:`, `fix:`, `docs:`,
  `refactor:`, `test:`, `chore:`), matching the existing history.
- Keep changes surgical. This package is `Development Status :: 5 -
  Production/Stable`; avoid renaming exports or moving module boundaries without
  checking every `__init__.py` that re-exports the symbol.
- Update docs/examples when you change public behaviour. Prefer fixing a stale
  doc/example to match the code over the reverse.
- One logical change per PR. Describe the motivation and how you tested it.

## Reporting bugs and security issues

- **Bugs / feature requests:** open an issue at
  https://github.com/10xHub/agentflow/issues with a minimal reproduction.
- **Security vulnerabilities:** do **not** open a public issue — follow
  [`SECURITY.md`](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
