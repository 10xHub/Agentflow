# Copilot instructions for 10xScale Agentflow

Use these repo-specific notes to be productive quickly when generating code, docs, or tests.

## Big picture
- 10xScale Agentflow is a lightweight Python framework for building multi-agent workflows with LLM-agnostic orchestration.
- Core primitives live in `agentflow/graph/`: `StateGraph`, `Node`, `Edge`, `ToolNode`, `CompiledGraph`.
- State is a Pydantic model (`state/agent_state.py`); messages are `state/message.py::Message` with multimodal content blocks (`state/message_block.py`).
- Flow: build a `StateGraph` → add nodes/edges (incl. conditional) → `set_entry_point` → `compile()` → use `CompiledGraph.invoke()` or `CompiledGraph.stream()`.
- START/END constants come from `utils/constants.py` (`START="__start__"`, `END="__end__"`). Always reference these.

## Architecture essentials
- **Graph API** (see `graph/state_graph.py`, `graph/compiled_graph.py`, `graph/node.py`):
  - `add_node(name, func|ToolNode)`, `add_edge(from, to)`, `add_conditional_edges(from, condition, path_map)`.
  - `compile(checkpointer, interrupt_before, interrupt_after)` wires DI (InjectQ), checkpointer, callbacks, publisher; default checkpointer is in-memory.
- **Node contracts** (`graph/node.py`):
  - Functions accept `(state: AgentState, config: dict, …deps)` and return updated state, `list[Message]`, or `ModelResponseConverter`.
  - Streaming supported by `.stream(...)` and `EventModel` chunks.
- **Tools via ToolNode** (`graph/tool_node/base.py`):
  - Register plain callables or integrate MCP, Composio, LangChain. Get tool specs via `await tool_node.all_tools()`.
  - Tool functions may declare injectable params like `tool_call_id` and `state` (see examples).
  - Supports parallel tool execution.
- **Persistence & events**:
  - Checkpointers: `InMemoryCheckpointer` (default) and `PgCheckpointer` (Postgres+Redis, extras required).
  - Publishers emit execution events (`ConsolePublisher` for dev). Events are `publisher/events.py::EventModel`.
- **State management**:
  - `AgentState` has `context` (list of messages with `add_messages` reducer), `context_summary`, and `execution_meta` (internal execution state).
  - Subclass `AgentState` to add custom fields while maintaining framework compatibility.
  - Use `BaseContextManager` for custom context trimming logic.

## Authoritative examples
- Minimal graph pattern: `tests/graph/test_graph.py` shows `add_node` → `set_entry_point` → `add_edge(..., END)` → `compile()` → `invoke()`.
- Tooling + routing: `examples/react/react_weather_agent.py`.
- MCP tools: `examples/react-mcp/`.
- Custom state: `examples/custom-state/`.
- Memory integration: `examples/memory/`.
- Callbacks & validation: `examples/callback-validation/`, `examples/input-validation/`.
- Multiagent patterns: `examples/multiagent/`, `examples/handoff/`.
- Prebuilt agents: `agentflow/prebuilt/agent/` (React, RAG, Swarm, Router, MapReduce, SupervisorTeam, PlanActReflect, etc.).

## Conventions
- **Messages**: Create via `Message.text_message(...)` or `Message.tool_message(...)`. Use `ModelResponseConverter` to wrap LLM responses.
- **Content blocks** (`state/message_block.py`): TextBlock, ImageBlock, AudioBlock, VideoBlock, DocumentBlock, DataBlock, ToolCallBlock, ToolResultBlock, ReasoningBlock, AnnotationBlock, ErrorBlock, RemoteToolCallBlock, MediaRef.
- **LLM adapters** (`adapters/llm/`): `ModelResponseConverter` supports the built-in OpenAI, OpenAI Responses, and Google GenAI converters. Use `convert_messages()` from `utils/converter.py` to prepare messages for LLM APIs.
- **ID generation**: DI-driven via InjectQ keys: `generated_id_type` ("string"|"int"|"bigint") and optional `generated_id` (value/awaitable). See `utils/id_generator.py`.
- **Reducers** (`state/reducers.py`): Use `add_messages`, `replace_messages`, `append_items` for state merging.
- **Conditional routing**: Returns labels that must match `path_map` keys; include `END` to terminate.
- **Tests**: pytest-only from `tests/` (config in `pyproject.toml`; markers: `asyncio`, `integration`; coverage enabled).

## Workflows
- **Setup**: Python ≥ 3.12. Core deps: `injectq`, `pydantic`, `python-dotenv`. Dev deps in `requirements-dev.txt`.
- **Tests**: `make test` (uv) or VS Code task "pytest -q". Coverage: `make test-cov`.
- **Build/publish**: `make build`, `make publish` (or `make testpublish`).
- **Docs**: `make docs-serve` (mkdocs at http://127.0.0.1:8000), `make docs-build`.
- **Examples**: Most examples require an LLM API key (e.g., `OPENAI_API_KEY`) set in `.env` and a valid model string for your chosen provider.

## Optional integrations (extras in `pyproject.toml`)
- `openai`, `google-genai`, `anthropic`: provider SDKs. Anthropic adds `anthropic-vertex` and
  `anthropic-bedrock` for Claude on Vertex AI / Bedrock, selected with `anthropic_backend`.
- `all`: every other extra at once, for development and CI.
- `pg_checkpoint`: Postgres (`asyncpg`) + Redis cache (`redis`); use `checkpointer.PgCheckpointer`.
- `mcp`: Model Context Protocol (FastMCP + mcp); pass an MCP client to `ToolNode`.
- `composio`, `langchain`: adapters for external tool registries via `ToolNode`.
- `redis`, `kafka`, `rabbitmq`: Event publisher backends in `agentflow/publisher/`.
- `qdrant`: Qdrant vector store for RAG patterns.
- `mem0`: Mem0 long-term memory integration.

## Gotchas
- Call `set_entry_point(...)` before `compile()` or you'll get `GraphError`.
- In `add_conditional_edges`, ensure the condition's return values match `path_map` keys exactly (including `END`).
- For `PgCheckpointer`, install extras and run `setup()/asetup()` to initialize schema before use.
- `CompiledGraph.stream/astream` yields `EventModel` chunks; use `content_type` to parse. Use `StreamChunk` for typed streaming responses.
- Always call `await compiled_graph.aclose()` for graceful cleanup of resources (DB connections, publishers, background tasks).
- When using dependency injection, register dependencies in the InjectQ container before calling `compile()`.
- Message IDs are auto-generated based on `generated_id_type` context; ensure consistency for database schemas.

## Environment
Python ≥ 3.12 required. Activate venv using `source .venv/bin/activate` and run pytest from the project root:
```bash
pytest
```