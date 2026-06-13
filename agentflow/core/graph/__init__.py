"""Agentflow Graph Module - Core Workflow Engine.

This module provides the foundational components for building and executing
agent workflows in TAF. It implements a graph-based execution model
similar to LangGraph, where workflows are defined as directed graphs of
interconnected nodes that process state and execute business logic.

Architecture Overview:
======================

The graph module follows a builder pattern for workflow construction and
provides a compiled execution environment for runtime performance. The core
components work together to enable complex, stateful agent interactions:

1. **StateGraph**: The primary builder class for constructing workflows
2. **Node**: Executable units that encapsulate functions or tool operations
3. **Edge**: Connections between nodes that define execution flow
4. **CompiledGraph**: The executable runtime form of a constructed graph
5. **ToolNode**: Specialized node for managing and executing tools

Core Components:
================

StateGraph:
    The main entry point for building workflows. Provides a fluent API for
    adding nodes, connecting them with edges, and configuring execution
    behavior. Supports both static and conditional routing between nodes.

Node:
    Represents an executable unit within the graph. Wraps functions or
    ToolNode instances and handles dependency injection, parameter mapping,
    and execution context. Supports both regular and streaming execution modes.

Edge:
    Defines connections between nodes, supporting both static (always followed)
    and conditional (state-dependent) routing. Enables complex branching logic
    and decision trees within workflows.

CompiledGraph:
    The executable runtime form created by compiling a StateGraph. Provides
    synchronous and asynchronous execution methods, state persistence,
    event publishing, and comprehensive error handling.

ToolNode:
    A specialized registry and executor for callable functions from various
    sources including local functions, MCP tools, Composio integrations,
    and LangChain tools. Supports automatic schema generation and unified
    tool execution.

Key Features:
=============

- **State Management**: Persistent, typed state that flows between nodes
- **Dependency Injection**: Automatic injection of framework services
- **Event Publishing**: Comprehensive execution monitoring and debugging
- **Streaming Support**: Real-time incremental result processing
- **Interrupts & Resume**: Pauseable execution with checkpointing
- **Tool Integration**: Unified interface for various tool providers
- **Type Safety**: Generic typing for custom state classes
- **Error Handling**: Robust error recovery and callback mechanisms

Usage Example:
==============

    ```python
    from agentflow.core.graph import StateGraph, ToolNode
    from agentflow.utils import START, END


    # Define workflow functions
    def process_input(state, config):
        # Process user input
        result = analyze_input(state.context[-1].content)
        return [Message.text_message(f"Analysis: {result}")]


    def generate_response(state, config):
        # Generate final response
        response = create_response(state.context)
        return [Message.text_message(response)]


    # Create tools
    def search_tool(query: str) -> str:
        return f"Search results for: {query}"


    tools = ToolNode([search_tool])

    # Build the graph
    graph = StateGraph()
    graph.add_node("process", process_input)
    graph.add_node("search", tools)
    graph.add_node("respond", generate_response)

    # Define flow
    graph.add_edge(START, "process")
    graph.add_edge("process", "search")
    graph.add_edge("search", "respond")
    graph.add_edge("respond", END)

    # Compile and execute
    compiled = graph.compile()
    result = compiled.invoke({"messages": [Message.text_message("Hello, world!")]})

    # Cleanup
    await compiled.aclose()
    ```

Integration Points:
==================

The graph module integrates with other TAF components:

- **State Module**: Provides AgentState and context management
- **Utils Module**: Supplies constants, messages, and helper functions
- **Checkpointer Module**: Enables state persistence and recovery
- **Publisher Module**: Handles event publishing and monitoring
- **Adapters Module**: Connects with external tools and services

This architecture provides a flexible, extensible foundation for building
sophisticated agent workflows while maintaining simplicity for common use cases.
"""

from .agent import Agent
from .agent_internal.constants import RetryConfig
from .base_agent import BaseAgent
from .compiled_graph import CompiledGraph
from .edge import Edge
from .node import Node
from .state_graph import StateGraph
from .tool_node import ToolNode


__all__ = [
    "Agent",
    "BaseAgent",
    "CompiledGraph",
    "Edge",
    "Node",
    "RetryConfig",
    "StateGraph",
    "ToolNode",
]
