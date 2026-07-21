
# 10xScale Agentflow

[![CI](https://github.com/10xHub/agentflow/actions/workflows/ci.yml/badge.svg)](https://github.com/10xHub/agentflow/actions/workflows/ci.yml)
[![Release](https://github.com/10xHub/agentflow/actions/workflows/release.yml/badge.svg)](https://github.com/10xHub/agentflow/actions/workflows/release.yml)
[![CodeQL](https://github.com/10xHub/agentflow/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/10xHub/agentflow/actions/workflows/github-code-scanning/codeql)
[![codecov](https://codecov.io/gh/10xHub/agentflow/branch/main/graph/badge.svg)](https://codecov.io/gh/10xHub/agentflow)

[![PyPI](https://img.shields.io/pypi/v/10xscale-agentflow?color=blue)](https://pypi.org/project/10xscale-agentflow/)
[![Python](https://img.shields.io/pypi/pyversions/10xscale-agentflow)](https://pypi.org/project/10xscale-agentflow/)
[![License](https://img.shields.io/github/license/10xHub/agentflow)](https://github.com/10xHub/agentflow/blob/main/LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](https://codecov.io/gh/10xHub/agentflow)
[![Tests](https://img.shields.io/badge/tests-2953%20passed-brightgreen.svg)](https://github.com/10xHub/agentflow/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-production%2Fstable-brightgreen.svg)](https://pypi.org/project/10xscale-agentflow/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**10xScale Agentflow** is a lightweight Python framework for building intelligent agents and orchestrating multi-agent workflows. It's an **LLM-agnostic orchestration tool** that works with native SDKs from OpenAI, Google Gemini, Anthropic Claude, or any other provider. You choose your LLM library; 10xScale Agentflow provides the workflow orchestration.

This package is the **core engine**. It ships as part of a complete, end-to-end framework: an API server and CLI, a typed TypeScript/React client, a visual playground, and a full documentation site. Start here, then pick up the rest of the stack as you need it.

---

## 🧩 The Agentflow Ecosystem

Agentflow is not just a Python library. It is a full stack for taking a multi-agent system from a prototype to production.

| Package | What it does | Install | Source |
|---|---|---|---|
| **Core framework**<br>`10xscale-agentflow` | Graph orchestration engine, state and checkpointing, 3-layer memory, parallel tools, MCP, publishers, evaluation | `pip install 10xscale-agentflow` | [`agentflow/`](https://github.com/10xHub/agentflow/tree/main/agentflow) |
| **API server + CLI**<br>`10xscale-agentflow-cli` | Turns a compiled graph into a FastAPI service over REST + WebSocket. JWT auth, RBAC, rate limiting, thread and memory APIs, Docker/K8s build | `pip install 10xscale-agentflow-cli` | [`agentflow-api/`](https://github.com/10xHub/agentflow/tree/main/agentflow-api) |
| **TypeScript client SDK**<br>`@10xscale/agentflow-client` | Typed client for every endpoint, React streaming hooks, client-side tool execution, realtime audio over WebSocket | `npm install @10xscale/agentflow-client` | [`agentflow-client/`](https://github.com/10xHub/agentflow/tree/main/agentflow-client) |
| **Playground** | React + Vite UI to chat with your agents, inspect graphs, threads, and state | `agentflow play` | [`agentflow-playground/`](https://github.com/10xHub/agentflow/tree/main/agentflow-playground) |
| **Documentation** | Tutorials, how-to guides, concepts, and API reference | [agentflow.10xscale.ai](https://agentflow.10xscale.ai/) | [`agentflow-docs/`](https://github.com/10xHub/agentflow/tree/main/agentflow-docs) |

**The 60-second path from install to a running service:**

```bash
pip install 10xscale-agentflow 10xscale-agentflow-cli
agentflow init my-agent && cd my-agent   # scaffold a project
agentflow api                            # REST + WebSocket API on :8000
agentflow play                           # server + visual playground
```

---

## ✨ Key Features

- **⚡ Agent Class** - Build complete agents in 10-30 lines of code (new in v0.5.3!)
- **🎯 LLM-Agnostic Orchestration** - Works with any LLM provider (OpenAI, Gemini, Claude, native SDKs)
- **🤖 Multi-Agent Workflows** - Build complex agent systems with your choice of orchestration patterns
- **📊 Structured Responses** - Get `content`, optional `thinking`, and `usage` in a standardized format
- **🌊 Streaming Support** - Real-time incremental responses with delta updates
- **🎙️ Realtime Audio Agents** - Live audio-to-audio sessions over Gemini Live with barge-in, transcripts, tool calling, and automatic reconnect (`AudioAgent`)
- **🔧 Tool Integration** - Native support for function calling and MCP tools with **parallel execution**
- **🔀 LangGraph-Inspired Engine** - Flexible graph orchestration with nodes, conditional edges, and control flow
- **💾 State Management** - Built-in persistence with in-memory and PostgreSQL+Redis checkpointers
- **🔄 Human-in-the-Loop** - Pause/resume execution for approval workflows and debugging
- **🚀 Production-Ready** - Event publishing (Console, Redis, Kafka, RabbitMQ), metrics, and observability
- **🧩 Dependency Injection** - Clean parameter injection for tools and nodes
- **📦 Prebuilt Patterns** - React, RAG, Swarm, Router, MapReduce, SupervisorTeam, and more

---

## 🌟 What Makes Agentflow Different

<details>
<summary><strong>The design decisions behind the framework</strong> (click to expand)</summary>

**Architecture and scale**
- **Cached checkpointing** — PostgreSQL for durability, Redis as the hot layer, so state persistence stays fast under load.
- **3-layer memory** — short-term working context, session conversation history, and long-term vector memory (Qdrant / Mem0).
- **Custom ID generation** — string, int, or bigint IDs instead of 128-bit UUIDs, which meaningfully shrinks database indexes.

**Tools and context**
- **Parallel tool execution** by default, with local Python tools, remote tools over the TypeScript SDK, agent handoff tools, and MCP servers all in the same registry.
- **Dedicated context manager** — trims context at iteration boundaries, never mid-execution, and is fully replaceable.
- **First-class dependency injection** via InjectQ, so tools and nodes stay testable.

**Control and safety**
- **Human-in-the-loop interrupts** — pause anywhere, resume with full state intact.
- **Callback system** at every execution stage for logging, validation, and prompt-injection defence.
- **Flexible navigation** — conditional routing, `Command` jumps, and handoff tools between agents.

**Operations**
- **Event publishing** to Kafka, RabbitMQ, Redis Pub/Sub, OpenTelemetry, or your own publisher.
- **Background task manager** for prefetching, memory persistence, and cleanup outside the request path.
- **Pydantic-first** — state, messages, and tool calls serialize, validate, and store without glue code.

</details>

---

## 📦 Installation

```bash
pip install 10xscale-agentflow          # or: uv pip install 10xscale-agentflow
```

Provider SDKs and infrastructure integrations are optional extras — install only what you use:

| Extra | Adds |
|---|---|
| `google-genai`, `openai` | Provider SDK adapters |
| `realtime` | Audio-to-audio agents over Gemini Live |
| `mcp` | Model Context Protocol client and tools |
| `pg_checkpoint`, `sqlite_checkpoint` | Durable checkpointing (Postgres + Redis, or SQLite) |
| `qdrant`, `mem0` | Long-term vector memory stores |
| `redis`, `kafka`, `rabbitmq`, `otel` | Event publishers and tracing |
| `images`, `cloud-storage` | Multimodal media handling and offload |

```bash
pip install "10xscale-agentflow[google-genai,openai,mcp,pg_checkpoint]"
```

Then set your provider key. A `.env` file in the working directory is loaded automatically.

```bash
export GEMINI_API_KEY=...     # Google Gemini
export OPENAI_API_KEY=sk-...  # OpenAI, or any OpenAI-compatible endpoint
```

---

## 🚀 Quick Start

Prebuilt agents are the fastest way in. A complete tool-calling agent:

```python
from agentflow.core.state import Message
from agentflow.prebuilt.agent import ReactAgent


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny, 22°C."


app = ReactAgent(
    model="gemini/gemini-2.5-flash",
    system_prompt=[{"role": "system", "content": "You are a helpful assistant."}],
    tools=[get_weather],
).compile()

result = app.invoke(
    {"messages": [Message.text_message("What's the weather in Tokyo?")]},
    config={"thread_id": "1"},
)

for message in result["messages"]:
    print(f"{message.role}: {message.content}")
```

That is the whole program. Message conversion, the tool loop, and parallel tool execution are handled
for you. Swap `ReactAgent` for `RAGAgent`, `SwarmAgent`, `SupervisorTeamAgent`, or `PlanActReflectAgent`
and the shape stays the same.

**Stream it** — same agent, incremental output:

```python
async for chunk in app.astream(
    {"messages": [Message.text_message("What's the weather in Tokyo?")]},
    config={"thread_id": "1"},
):
    print(chunk.model_dump())
```

**Add MCP tools** — pass a `fastmcp` client; remote tools join your local ones:

```python
from fastmcp import Client

mcp_client = Client({
    "mcpServers": {
        "weather": {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"},
    }
})

app = ReactAgent(
    model="gemini/gemini-2.5-flash",
    tools=[get_weather],
    client=mcp_client,
).compile()
```

**Persist conversations** — pass a checkpointer at compile time and reuse the `thread_id`:

```python
from agentflow.storage.checkpointer import InMemoryCheckpointer  # PgCheckpointer in production

app = ReactAgent(model="gemini/gemini-2.5-flash", tools=[get_weather]).compile(
    checkpointer=InMemoryCheckpointer(),
)
```

---

## 🏗️ Building Your Own Graph

Prebuilt agents are graphs. When you need custom control flow, build one directly with the same
primitives — nodes, conditional edges, and an entry point:

```python
from agentflow.core.graph import Agent, StateGraph, ToolNode
from agentflow.core.state import AgentState, Message
from agentflow.utils.constants import END

graph = StateGraph()
graph.add_node("MAIN", Agent(
    model="gemini/gemini-2.5-flash",
    system_prompt=[{"role": "system", "content": "You are a helpful assistant."}],
    tool_node="TOOL",
))
graph.add_node("TOOL", ToolNode([get_weather]))


def route(state: AgentState) -> str:
    if state.context and state.context[-1].tools_calls:
        return "TOOL"
    return END


graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile()
```

Nodes can be plain functions too, so you can call a provider SDK directly and keep full control over
the request. See [`examples/react/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/react)
for that variant.

---

## 🎙️ Realtime Audio Agents

Live audio-to-audio sessions over Gemini Live. The provider owns the turn loop, so the session runs
through `arealtime`: you push into an input queue and consume normalized events.

```python
from agentflow.prebuilt.agent import AudioAgent
from agentflow.core.realtime import LiveInputQueue, RealtimeConfig

app = AudioAgent(
    "gemini-live-2.5-flash-preview",
    realtime_config=RealtimeConfig(model="gemini-live-2.5-flash-preview", voice="Puck"),
    tools=[get_weather],
).compile()

queue = LiveInputQueue()
queue.send_audio(pcm16_bytes)   # non-blocking, safe to call from an audio callback

async for event in app.arealtime(queue, {"thread_id": "t1"}):
    ...                         # AudioDeltaEvent / transcripts / ToolCallEvent / ...

queue.close()
```

Barge-in, persisted transcripts (raw audio is never stored), automatic reconnect with session
resumption, and image/video frame input are handled for you. `system_prompt`, `skills`, and `memory`
work as they do on any other agent. Needs `pip install 10xscale-agentflow[realtime]`.

---

## ⚡ Parallel Tool Execution

When an LLM requests several tools at once, Agentflow runs them concurrently. No configuration, no
code changes — it applies to every agent and graph.

```text
get_weather("NYC") 1.0s + get_news("tech") 1.5s + get_stock("AAPL") 0.8s

Sequential: 1.0 + 1.5 + 0.8 = 3.3s
Agentflow:  max(1.0, 1.5, 0.8) = 1.5s
```

See the [parallel tool execution docs](https://agentflow.10xscale.ai/Concept/graph/tools/#parallel-tool-execution).

---

## 📚 More Examples

Every pattern below is a runnable script in
[`examples/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples):

| Topic | Directory |
|---|---|
| React agents, sync and class-based | [`react/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/react) |
| Streaming and stop/resume | [`react_stream/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/react_stream) |
| MCP servers and tools | [`react-mcp/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/react-mcp), [`github-mcp/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/github-mcp) |
| RAG, memory, and vector stores | [`rag/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/rag), [`memory/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/memory), [`store/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/store) |
| Multi-agent: swarm, supervisor, handoff | [`swarm/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/swarm), [`supervisor_team/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/supervisor_team), [`handoff/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/handoff) |
| Realtime audio, multimodal | [`realtime/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/realtime), [`multimodal/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/multimodal) |
| Structured output, skills, custom state | [`structured_output/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/structured_output), [`skills/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/skills), [`custom-state/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/custom-state) |
| Checkpointing, evaluation, testing | [`checkpointer/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/checkpointer), [`evaluation/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/evaluation), [`testing/`](https://github.com/10xHub/agentflow/tree/main/agentflow/examples/testing) |

Run one:

```bash
export GEMINI_API_KEY=...   # or OPENAI_API_KEY
python examples/react/react_single_class.py
```

---

## 🎯 Use Cases & Patterns

10xScale Agentflow includes prebuilt agent patterns for common scenarios:

### 🤖 Agent Types

- **React Agent** - Reasoning and acting with tool calls
- **RAG Agent** - Retrieval-augmented generation
- **Guarded Agent** - Input/output validation and safety
- **Plan-Act-Reflect** - Multi-step reasoning
- **Audio Agent** - Realtime audio-to-audio sessions (Gemini Live) with barge-in and tool calling

### 🔀 Orchestration Patterns

- **Router Agent** - Route queries to specialized agents
- **Swarm** - Dynamic multi-agent collaboration
- **SupervisorTeam** - Hierarchical agent coordination
- **MapReduce** - Parallel processing and aggregation
- **Sequential** - Linear workflow chains
- **Branch-Join** - Parallel branches with synchronization

### 🔬 Advanced Patterns

- **Deep Research** - Multi-level research and synthesis
- **Network** - Complex agent networks

See the [documentation](https://10xhub.github.io/Agentflow/) for complete examples.

---

## 🔧 Development

### For Library Users

Install 10xScale Agentflow as shown above. The `pyproject.toml` contains all runtime dependencies.

### For Contributors

```bash
# Clone the monorepo and enter the core framework
git clone https://github.com/10xHub/agentflow.git
cd agentflow/agentflow

# Create the environment and install dev dependencies
uv sync --dev
uv run pre-commit install          # optional, mirrors CI

# Run tests (coverage gate: >= 80%)
uv run pytest --cov --cov-branch
uv run pytest tests/graph          # one area

# Lint, format, type-check
uv run ruff check . && uv run ruff format .
uv run mypy agentflow/

# Run an example
uv run python examples/react/react_sync.py
```

### Development Tools

The project uses:
- **pytest** for testing (async support, coverage gate at 80%)
- **ruff** for linting and formatting (line length 100, target py312)
- **mypy** for type checking, applied in phases across the codebase
- **bandit** for security checks
- **pre-commit** to run all of the above before each commit
- **Docusaurus** for the documentation site (`agentflow-docs/`)

Tool configuration lives in `pyproject.toml`.

---

## 🗺️ Roadmap

- ✅ Core graph engine with nodes and edges
- ✅ State management and checkpointing
- ✅ Tool integration (MCP, custom tools, parallel execution)
- ✅ **Parallel tool execution** for improved performance
- ✅ Streaming and event publishing
- ✅ Human-in-the-loop support
- ✅ Prebuilt agent patterns
- ✅ Agent-to-Agent (A2A) communication protocols
- ✅ Observability and tracing (OpenTelemetry)
- ✅ Realtime audio-to-audio agents (Gemini Live)
- 🚧 Remote node execution for distributed processing
- 🚧 More persistence backends (Redis, DynamoDB)
- 🚧 Parallel/branching strategies
- 🚧 Visual graph editor

---

## 🤝 Contributing

**Agentflow is built in the open, and we are actively looking for contributors.**

The framework is production-stable at its core, but it is wide: a graph engine, an API server, a
TypeScript SDK, a React playground, and a documentation site. Every one of those surfaces has room
for people who want to harden it, extend it, or make it easier to learn. If you have ever wanted to
work on agent infrastructure rather than just use it, this is a good place to start.

### Where you can contribute

Contributions are not limited to this package. The whole framework lives in one monorepo, so pick
whichever part matches your skills:

| Area | Stack | Good if you enjoy |
|---|---|---|
| [**Core framework**](https://github.com/10xHub/agentflow/tree/main/agentflow) | Python 3.12, Pydantic, asyncio | Graph execution, state, persistence, memory, LLM adapters, evaluation |
| [**API server + CLI**](https://github.com/10xHub/agentflow/tree/main/agentflow-api) | FastAPI, Typer | REST/WebSocket APIs, auth and RBAC, rate limiting, deployment tooling |
| [**TypeScript client**](https://github.com/10xHub/agentflow/tree/main/agentflow-client) | TypeScript, Vite, Vitest | SDK design, streaming, typed APIs, React hooks |
| [**Playground**](https://github.com/10xHub/agentflow/tree/main/agentflow-playground) | React 19, Redux Toolkit, Tailwind | UI/UX, graph visualization, developer tooling |
| [**Documentation**](https://github.com/10xHub/agentflow/tree/main/agentflow-docs) | Docusaurus, Markdown | Tutorials, guides, reference, and making concepts click |
| [**Examples**](https://github.com/10xHub/agentflow/tree/main/agentflow/examples) | Python | Showing real patterns: RAG, swarms, MCP, multimodal, realtime |

### Where we most need help right now

- **Stabilization** — edge cases in long-running graphs, streaming, and reconnect behaviour
- **Persistence backends** — Redis, DynamoDB, and other checkpointers beyond Postgres
- **Provider coverage** — more LLM adapters and better parity across providers
- **Typing** — the codebase is on phased `mypy` adoption; removing a module from the ignore list is a genuinely welcome PR
- **Docs and examples** — the fastest way to help other people adopt the framework
- **Bug reports** — even an issue with a clean reproduction moves the project forward

### Getting started

```bash
git clone https://github.com/10xHub/agentflow.git
cd agentflow/agentflow
uv sync --dev                      # environment + dev tools
uv run pre-commit install          # optional, matches CI
uv run pytest --cov --cov-branch   # tests + coverage gate
```

Read [CONTRIBUTING.md](https://github.com/10xHub/agentflow/blob/main/agentflow/CONTRIBUTING.md) for
the full workflow, and the [Code of Conduct](https://github.com/10xHub/agentflow/blob/main/agentflow/CODE_OF_CONDUCT.md)
before you participate. Issues labelled **`good first issue`** are a deliberate on-ramp — if none are
open, say hello in [Discussions](https://github.com/10xHub/agentflow/discussions) and we will scope one with you.

---

## 🌟 Contributors

Every feature in this framework exists because someone decided to build it. Thank you to everyone who
has contributed code, docs, examples, issues, and reviews.

<a href="https://github.com/10xHub/agentflow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=10xHub/agentflow" alt="Agentflow contributors" />
</a>

*Your avatar belongs here.* Open a pull request on any package in the monorepo and this wall updates itself.

---

## 🔐 Security

Found a vulnerability? Please do not open a public issue. Follow the disclosure process in
[SECURITY.md](https://github.com/10xHub/agentflow/blob/main/agentflow/SECURITY.md).

---

## 📄 License

MIT License. See [LICENSE](https://github.com/10xHub/agentflow/blob/main/agentflow/LICENSE).

---

## 🔗 Links & Resources

**Documentation**
- [Documentation home](https://agentflow.10xscale.ai/) — tutorials, how-to guides, concepts, API reference
- [Examples directory](https://github.com/10xHub/agentflow/tree/main/agentflow/examples) — runnable code for every major pattern

**Packages**
- PyPI: [`10xscale-agentflow`](https://pypi.org/project/10xscale-agentflow/) (core) · [`10xscale-agentflow-cli`](https://pypi.org/project/10xscale-agentflow-cli/) (API + CLI)
- npm: [`@10xscale/agentflow-client`](https://www.npmjs.com/package/@10xscale/agentflow-client) (TypeScript SDK)

**Project**
- [GitHub repository](https://github.com/10xHub/agentflow) — source for all packages
- [Issues](https://github.com/10xHub/agentflow/issues) — bug reports and feature requests
- [Discussions](https://github.com/10xHub/agentflow/discussions) — questions, ideas, and help
- [Changelog](https://github.com/10xHub/agentflow/blob/main/agentflow/CHANGELOG.md) — release history

---

**Ready to build?** Start with the [documentation](https://agentflow.10xscale.ai/), or jump straight into the
[examples](https://github.com/10xHub/agentflow/tree/main/agentflow/examples). If Agentflow is useful to you,
a ⭐ on the repository helps other developers find it.
