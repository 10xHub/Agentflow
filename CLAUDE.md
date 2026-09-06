# agentflow (core Python library) — Engineering Guide

This file documents the **core Python framework** only (`10xscale-agentflow`, the package that
lives in this folder). For the API/CLI, TS client, docs, or playground, see the CLAUDE.md in
their respective folders and the workspace-root `CLAUDE.md` for the monorepo overview.

- Package name (PyPI): `10xscale-agentflow`
- Version: `1.0.0` (single source of truth: `pyproject.toml`; `agentflow.__version__`
  resolves from installed distribution metadata, so it cannot drift from this file)
- Requires: Python >= 3.12
- Importable top-level package lives at `agentflow/agentflow/` (this folder is the repo root;
  the importable package is the nested `agentflow/` directory).

## What this package is

A graph-based orchestration engine for multi-agent LLM systems. It is **LLM-agnostic**: you bring
the provider SDK (OpenAI / Google GenAI), and Agentflow provides the workflow engine, state,
persistence, tools, memory, evaluation, and event publishing. Inspired by LangGraph but simpler.

## Working principles for this codebase

- **Read before writing.** The public API is large and re-exported through many `__init__.py`
  files. Confirm the real export path before referencing a symbol (see Import Map below).
- **Examples are the source of truth**, not the README. `examples/` uses current import paths;
  the README and several docstrings still show pre-refactor paths (see Known Doc Drift).
- **Surgical edits.** This is `Development Status :: 5 - Production/Stable`. Don't refactor
  module boundaries or rename exports without checking every `__init__.py` that re-exports them.
- **Keep coverage green.** `pytest` enforces `--cov-fail-under=80`. New code needs tests.
- **Optional deps are optional.** Provider SDKs, MCP, Postgres, Redis, Qdrant, Mem0, Kafka,
  RabbitMQ, OTEL, a2a are all extras. Guard imports; never make core import a hard optional dep.

## Package layout (real, current)

The importable package is `agentflow/agentflow/`. Top-level subpackages:

| Subpackage | What lives there |
|---|---|
| `core/` | The engine. `graph/` (StateGraph, Agent, ToolNode, CompiledGraph, Node, Edge), `state/` (AgentState, Message, content blocks, reducers, context managers), `llm/` (provider detection + client factory + `call_llm`), `skills/` (dynamic skill injection), `exceptions/` |
| `storage/` | `checkpointer/` (InMemory, Pg), `store/` (vector/long-term memory: Qdrant, Mem0, embeddings), `media/` (multimodal media processing, offload, resolvers, stores) |
| `runtime/` | `adapters/llm/` (OpenAI / OpenAI-Responses / Google GenAI / Anthropic response converters), `publisher/` (Console, Redis, Kafka, RabbitMQ, OTEL, Composite), `protocols/` (a2a, acp) |
| `prebuilt/` | `agent/` (React, RAG, PlanActReflect, SupervisorTeam, Swarm, StructuredOutput), `tools/` (calculator, fetch, files, handoff, memory, search) |
| `qa/` | `evaluation/` (criteria, datasets, evaluator, reporters, simulators) and `testing/` (TestAgent, mocks, quick tests) |
| `utils/` | constants (START/END/ResponseGranularity), `tool` decorator, `convert_messages`, callbacks, validators, id generators, background tasks, graceful shutdown |

## Import Map (verified) — this is the part that bites people

The package was restructured into `core/`, `storage/`, `runtime/`, `qa/`. **There are no
top-level `agentflow.graph`, `agentflow.state`, `agentflow.checkpointer`, `agentflow.skills`,
`agentflow.evaluation`, `agentflow.testing`, `agentflow.adapters`, or `agentflow.publisher`
shims.** Those paths raise `ModuleNotFoundError`. Use the canonical paths:

```python
# Graph engine
from agentflow.core.graph import Agent, StateGraph, ToolNode, CompiledGraph, Node, Edge, RetryConfig
# or the aggregate: from agentflow.core import StateGraph, Agent, ToolNode, AgentState, Message, ...

# State and messages
from agentflow.core.state import AgentState, Message, TextBlock, ToolResultBlock, add_messages

# LLM client/provider helpers
from agentflow.core.llm import call_llm, create_llm_client, detect_provider

# Skills
from agentflow.core.skills import SkillConfig, SkillMeta, SkillsRegistry

# Persistence
from agentflow.storage.checkpointer import InMemoryCheckpointer, PgCheckpointer, BaseCheckpointer
# Vector / long-term memory
from agentflow.storage.store import QdrantStore, Mem0Store, MemoryConfig, AgentMemoryConfig

# Publishers / converters
from agentflow.runtime.publisher import ConsolePublisher, RedisPublisher, KafkaPublisher, RabbitMQPublisher
from agentflow.runtime.adapters.llm import OpenAIConverter, GoogleGenAIConverter, OpenAIResponsesConverter

# Prebuilt
from agentflow.prebuilt.agent import ReactAgent, RAGAgent, SwarmAgent, SupervisorTeamAgent
from agentflow.prebuilt.tools import safe_calculator, fetch_url, create_handoff_tool, memory_tool

# QA
from agentflow.qa.evaluation import AgentEvaluator, EvalConfig, EvalCase, EvalSet
from agentflow.qa.testing import TestAgent, MockMCPClient, MockToolRegistry

# Utils
from agentflow.utils import tool, convert_messages, Command
from agentflow.utils.constants import START, END, ResponseGranularity
```

Note: the root `agentflow/__init__.py` is intentionally empty. Importing the package does not
eagerly pull in submodules; import the subpackage you need.

## Core concepts

**StateGraph -> CompiledGraph.** Build with `StateGraph()`, `add_node`, `add_edge`,
`add_conditional_edges`, `set_entry_point`; then `.compile(...)` returns a `CompiledGraph`.
`compile()` accepts: `checkpointer`, `store`, `media_store`, `interrupt_before`,
`interrupt_after`, `callback_manager`, `shutdown_timeout` (default 30.0).

**CompiledGraph execution API:** `invoke` / `ainvoke` (run), `stream` / `astream` (incremental),
`stop` / `astop` (interrupt), `override_node`, `attach_remote_tools`, `generate_graph`, `aclose`.
- Input shape: `{"messages": [Message...]}`.
- Config keys: `user_id`, `thread_id`, `run_id`, `recursion_limit` (default 25).
- `response_granularity`: `LOW` (messages only, default), `PARTIAL` (context+summary+messages),
  `FULL` (full state).

**Agent class** (`agentflow.core.graph.Agent`) — the high-level node that wraps LLM calls,
message conversion, and tool integration. Key constructor params:
`model` (required), `output_type="text"`, `system_prompt`, `tool_node` (name or ToolNode),
`extra_messages`, `trim_context`, `tools_tags`, `reasoning_config`, `skills`, `memory`,
`retry_config` (default True), `fallback_models`, `multimodal_config`, `output_schema`.

**Model strings and providers.** `detect_provider(model)` infers the provider from a
`"provider/model"` prefix or the model name, and resolves to `"google"`, `"openai"`, or
`"anthropic"`. Examples: `"gemini/gemini-2.5-flash"`, `"openai/gpt-4o"`, `"gpt-4o-mini"`,
`"claude-opus-5"`, `"anthropic/claude-sonnet-5"`. Google's Vertex AI is selected via
`use_vertex_ai=True`.

Anthropic has three backends, so a boolean flag cannot express them; the selector is the
string `anthropic_backend`: `None` (direct Claude API, the default), `"vertex"`
(`AsyncAnthropicVertex`), or `"bedrock"` (`AsyncAnthropicBedrockMantle`, the Messages-API
endpoint, not the legacy `InvokeModel` client). Bedrock model ids keep their `anthropic.`
prefix (`"anthropic.claude-opus-5"`); it is preserved, not stripped. Install with the
`[anthropic]`, `[anthropic-vertex]`, or `[anthropic-bedrock]` extra.

Anthropic-specific request behaviour, all handled by the provider and not the caller:
`max_tokens` is required and defaulted (16000 non-streaming, 64000 streaming);
`temperature`/`top_p`/`top_k` are stripped for models that reject them with a 400;
`reasoning_config={"effort": ...}` maps to `thinking={"type": "adaptive"}` plus
`output_config.effort`, and `budget_tokens` is never emitted; a trailing assistant turn is
dropped because prefill 400s on current models. `output_type` is limited to `text` and
`json`: the Messages API has no image/audio/video generation endpoint.

Note the SDK is capped at `anthropic>=1.0.0,<2`. The 1.0 release moved to **httpx2**, so an
`http_client` passed through `llm_kwargs` on the Anthropic path must be an httpx2 client,
while the OpenAI path still takes httpx.

**ToolNode.** `ToolNode(tools, client=None, pass_user_info_to_mcp=False)`. First positional arg
is `tools` (an iterable of callables). `client` is an MCP client (fastmcp/mcp). Tools run in
**parallel** when the LLM requests several at once. Define tools as plain functions; injectable
params (`tool_call_id`, `state`, `config`, plus InjectQ-provided deps) are filled automatically.

**State and Message.** `AgentState` is a Pydantic model; subclass it for custom fields.
`Message.text_message(content, role="user")` is the text factory. `Message.tool_message(...)`,
`Message.image_message(...)` exist. There is **no `Message.from_text`** (README shows it; it is
wrong). Content is a list of typed blocks (TextBlock, ImageBlock, ToolCallBlock, ToolResultBlock,
ReasoningBlock, etc.). Reducers (`add_messages`, `replace_messages`, `append_items`) control how
state lists merge.

**Persistence.** `InMemoryCheckpointer` for dev/tests. `PgCheckpointer` (Postgres + Redis dual
layer) for production; requires `[pg_checkpoint]`.

**Memory / store.** 3-layer model: working state -> checkpointer (hot/durable) -> vector store
(Qdrant/Mem0) for long-term. `MemoryConfig` / `AgentMemoryConfig` drive it; `memory_tool` and
`create_memory_preload_node` wire it into a graph.

**Skills.** `SkillConfig(skills_dir=...)` adds dynamic skill injection. Two modes: `on-demand`
(LLM calls `set_skill()` from a trigger table) and `session` (preload a fixed skill from a state
field via `preload_from`).

**Publishers.** Emit execution events to Console, Redis Pub/Sub, Kafka, RabbitMQ, or OTEL.
`CompositePublisher` fans out to several. OTEL publisher provides tracing (`setup_tracing`).

**QA.** `agentflow.qa.evaluation` is a full eval framework (criteria incl. LLM-as-judge,
trajectory matching, rubric, safety, hallucination; datasets; console/JSON/HTML/JUnit reporters;
user simulators). `agentflow.qa.testing` provides `TestAgent`, `MockMCPClient`, `MockToolRegistry`,
`TestContext` for unit-testing graphs without live LLMs.

## Development workflow

This repo root is `agentflow/`; the importable package is `agentflow/agentflow/`. A `.venv` is
already present.

```bash
# from this folder (agentflow/)
.venv/bin/python -m pytest               # full suite (enforces coverage >= 80%)
.venv/bin/python -m pytest tests/graph   # one area
ruff check . && ruff format .            # lint + format (line-length 100, py312)
# editable install with extras for local dev:
pip install -e ".[google-genai,openai,anthropic,mcp,pg_checkpoint]"
```

- Tests live in `tests/` (mirrors package layout: `graph/`, `state/`, `storage/`, `store/`,
  `checkpointer/`, `publisher/`, `prebuilt/`, `evaluation/`, `testing/`, plus `chaos/`,
  `benchmarks/`, `integration/`). Markers: `asyncio`, `integration` (needs real DBs), `slow`.
- Lint config is in `pyproject.toml` `[tool.ruff]` (broad rule set; per-file ignores for a few
  large modules). `mypy` and `bandit` are also configured there.
- `examples/` is organized by feature (react, rag, swarm, supervisor_team, memory, skills, mcp,
  a2a_sdk, evaluation, testing, multimodal, structured_output, ...). Use these as canonical usage.

## Known doc drift (do not copy from these without checking)

- **README.md import paths are stale.** It imports `agentflow.graph`, `agentflow.state`,
  `agentflow.checkpointer` — all removed. Real paths are `agentflow.core.*` / `agentflow.storage.*`.
- **`Message.from_text` does not exist** (README uses it). Use `Message.text_message`.
- **`ToolNode(functions=...)`** keyword is wrong (README MCP example). The param is `tools`.
- A few `examples/` files still use dead paths (`agentflow.state.message`, `agentflow.graph.tool_node`,
  `agentflow.evaluation.*`). Treat those specific files as broken until fixed.

When you touch any of the above, prefer fixing the doc/example to match the code rather than the
reverse, unless the export path itself is the bug.
