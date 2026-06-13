"""Core components for Agentflow.

This package provides the foundational building blocks for agent workflows:

- ``agentflow.core.graph``      — graph-based workflow engine (StateGraph, Agent, ...)
- ``agentflow.core.exceptions`` — custom exception hierarchy
- ``agentflow.core.skills``     — dynamic skill injection for agents
- ``agentflow.core.state``      — state management, messages, and reducers
"""

from __future__ import annotations

import typing as _t

from . import exceptions, skills, state

# --- Exceptions ---
from .exceptions import (
    GraphError,
    GraphRecursionError,
    MetricsError,
    NodeError,
    ResourceNotFoundError,
    SchemaVersionError,
    SerializationError,
    StorageError,
    TransientStorageError,
)

# --- Skills ---
from .skills import SkillConfig, SkillMeta, SkillsRegistry


# --- Graph (lazy) ---
# The graph engine is imported lazily to avoid an import cycle: ``agentflow.core.graph`` imports
# back into ``agentflow.utils`` and ``agentflow.storage.checkpointer``. Importing it eagerly here
# means that ``import agentflow.utils`` or ``import agentflow.storage.checkpointer`` *as the first
# import* triggers ``agentflow.core`` -> ``graph`` -> back into the half-initialized module and
# raises ImportError. Deferring graph keeps ``from agentflow.core import StateGraph`` working while
# letting those modules be imported in any order. See tests/test_import_order.py.
_GRAPH_EXPORTS = frozenset(
    {
        "Agent",
        "BaseAgent",
        "CompiledGraph",
        "Edge",
        "Node",
        "RetryConfig",
        "StateGraph",
        "ToolNode",
    }
)

if _t.TYPE_CHECKING:
    from . import graph
    from .graph import (
        Agent,
        BaseAgent,
        CompiledGraph,
        Edge,
        Node,
        RetryConfig,
        StateGraph,
        ToolNode,
    )


def __getattr__(name: str) -> _t.Any:
    """Lazily resolve the graph submodule and its exported symbols (PEP 562).

    Uses ``importlib.import_module`` (not ``from . import graph``) so a re-entrant lookup while
    ``graph`` is still importing returns the partial module from ``sys.modules`` directly instead
    of recursing back through this hook via the parent-attribute binding.
    """
    if name == "graph" or name in _GRAPH_EXPORTS:
        import importlib

        graph = importlib.import_module(f"{__name__}.graph")
        globals()["graph"] = graph  # cache so future lookups skip __getattr__
        return graph if name == "graph" else getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _GRAPH_EXPORTS | {"graph"})


# --- State ---
from .state import (
    AgentState,
    AnnotationBlock,
    AnnotationRef,
    AudioBlock,
    BaseContextManager,
    ContentBlock,
    DataBlock,
    DocumentBlock,
    ErrorBlock,
    ExecutionState,
    ExecutionStatus,
    ImageBlock,
    MediaRef,
    Message,
    MessageContextManager,
    ReasoningBlock,
    StreamChunk,
    StreamEvent,
    TextBlock,
    TokenUsages,
    ToolCallBlock,
    ToolResult,
    ToolResultBlock,
    VideoBlock,
    add_messages,
    append_items,
    remove_tool_messages,
    replace_messages,
    replace_value,
)


__all__ = [
    # Graph
    "Agent",
    # State
    "AgentState",
    "AnnotationBlock",
    "AnnotationRef",
    "AudioBlock",
    "BaseAgent",
    "BaseContextManager",
    "CompiledGraph",
    "ContentBlock",
    "DataBlock",
    "DocumentBlock",
    "Edge",
    "ErrorBlock",
    "ExecutionState",
    "ExecutionStatus",
    # Exceptions
    "GraphError",
    "GraphRecursionError",
    "ImageBlock",
    "MediaRef",
    "Message",
    "MessageContextManager",
    "MetricsError",
    "Node",
    "NodeError",
    "ReasoningBlock",
    "ResourceNotFoundError",
    "RetryConfig",
    "SchemaVersionError",
    "SerializationError",
    # Skills
    "SkillConfig",
    "SkillMeta",
    "SkillsRegistry",
    "StateGraph",
    "StorageError",
    "StreamChunk",
    "StreamEvent",
    "TextBlock",
    "TokenUsages",
    "ToolCallBlock",
    "ToolNode",
    "ToolResult",
    "ToolResultBlock",
    "TransientStorageError",
    "VideoBlock",
    "add_messages",
    "append_items",
    # Submodules
    "exceptions",
    "graph",
    "remove_tool_messages",
    "replace_messages",
    "replace_value",
    "skills",
    "state",
]
