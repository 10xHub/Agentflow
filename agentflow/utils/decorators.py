"""Decorators for tool registration and metadata management.

This module provides decorators for registering and configuring tools within the
agentflow framework. The decorators allow developers to attach metadata such as
name, description, tags, and other attributes to functions, making them easier
to discover and use within agent workflows.

The main decorator is @tool, which can be used to mark functions as tools and
provide additional metadata for function calling APIs and agent execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


def tool[F: Callable[..., Any]](
    _func: F | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | set[str] | None = None,
    provider: str | None = None,
    capabilities: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[F], F] | F:
    """Decorator to mark a function as a tool with metadata.

    This decorator attaches metadata to a function, making it recognizable as a tool
    within the agentflow framework. The metadata includes the tool's name, description,
    tags for categorization, provider information, and capabilities.

    The decorator can be used in two ways:
    1. With arguments: @tool(name="my_tool", description="...")
    2. Without arguments: @tool

    When used without arguments, the function's name and docstring are used as
    the tool name and description respectively.

    Args:
        _func: The function to decorate (when used without parentheses).
        name: The name of the tool. If not provided, uses the function's __name__.
            This name is used when the tool is called by agents or listed in schemas.
        description: A clear description of what the tool does. If not provided,
            uses the function's docstring. This is crucial for agents to understand
            when and how to use the tool.
        tags: A list or set of tags for categorizing and filtering tools. Tags can
            be used to group related tools or enable/disable sets of tools based on
            context. Examples: ["search", "web"], ["database", "read"].
        provider: The provider or source of the tool (e.g., "local", "langchain",
            "mcp", "composio"). This helps identify where the tool comes from.
        capabilities: A list of capabilities or permissions required by the tool.
            Examples: ["read_files", "network_access", "database_write"].
        metadata: Additional arbitrary metadata to attach to the tool. This can
            include custom fields specific to your application or workflow.
        parameters: An explicit JSON Schema for the tool's parameters. When provided,
            automatic schema generation is skipped and this dict is forwarded to the
            provider verbatim. Use it for parameter types the generator rejects, or
            when a specific provider needs a hand-tuned schema. Argument coercion
            still runs off the real annotations, so keep the two in agreement.

    Returns:
        A decorator function that wraps the target function and attaches metadata
        as private attributes (_py_tool_*). The wrapped function behaves identically
        to the original but carries the metadata for discovery and schema generation.

    Raises:
        ValueError: If the decorated object is not a callable function.

    Examples:
        Basic usage with just a name:
        >>> @tool("calculator")
        ... def add(a: int, b: int) -> int:
        ...     '''Add two numbers together.'''
        ...     return a + b

        Full usage with all options:
        >>> @tool(
        ...     name="web_search",
        ...     description="Search the web for information",
        ...     tags=["search", "web", "external"],
        ...     provider="custom",
        ...     capabilities=["network_access"],
        ...     metadata={"rate_limit": 100, "timeout": 30},
        ... )
        ... async def search_web(query: str, max_results: int = 10) -> list[str]:
        ...     # Implementation here
        ...     return ["result1", "result2"]

        Usage without arguments (uses function name and docstring):
        >>> @tool
        ... def multiply(x: int, y: int) -> int:
        ...     '''Multiply two numbers.'''
        ...     return x * y

    Note:
        - The metadata is stored as private attributes on the function object:
          - _py_tool_name: The tool name
          - _py_tool_description: The tool description
          - _py_tool_tags: Set of tags
          - _py_tool_provider: The provider string
          - _py_tool_capabilities: List of capabilities
          - _py_tool_metadata: Additional metadata dict
          - _py_tool_parameters: Explicit parameters JSON Schema, if supplied
        - These attributes are used by ToolNode and schema generation to build
          OpenAI-compatible function calling schemas.
        - Tags are converted to a set for efficient membership testing.
    """

    def decorator(func: F) -> F:
        """Inner decorator that attaches metadata to the function.

        Args:
            func: The function to decorate.

        Returns:
            The same function with metadata attributes attached.

        Raises:
            ValueError: If func is not callable.
        """
        if not callable(func):
            msg = f"@tool decorator can only be applied to callable functions, got {type(func)}"
            raise ValueError(msg)

        # Set the tool name (use provided name or function name)
        tool_name = name if name is not None else func.__name__
        func._py_tool_name = tool_name  # type: ignore[attr-defined]

        # Set the tool description (use provided description or docstring)
        tool_description = description
        if tool_description is None:
            tool_description = func.__doc__ or "No description provided."
        func._py_tool_description = tool_description  # type: ignore[attr-defined]

        # Set tags (convert to set for efficient operations)
        if tags is not None:
            tool_tags = set(tags) if not isinstance(tags, set) else tags
            func._py_tool_tags = tool_tags  # type: ignore[attr-defined]
        else:
            func._py_tool_tags = set()  # type: ignore[attr-defined]

        # Set provider if provided
        if provider is not None:
            func._py_tool_provider = provider  # type: ignore[attr-defined]

        # Set capabilities if provided
        if capabilities is not None:
            func._py_tool_capabilities = capabilities  # type: ignore[attr-defined]

        # Set additional metadata if provided
        if metadata is not None:
            func._py_tool_metadata = metadata  # type: ignore[attr-defined]

        # Explicit parameters schema bypasses automatic schema generation entirely
        if parameters is not None:
            if not isinstance(parameters, dict):
                msg = f"@tool(parameters=...) must be a JSON Schema dict, got {type(parameters)}"
                raise ValueError(msg)
            func._py_tool_parameters = parameters  # type: ignore[attr-defined]

        return func

    # Handle being called without parentheses (@tool)
    if _func is not None:
        return decorator(_func)

    # Handle being called with parentheses (@tool(...))
    return decorator


def get_tool_metadata(func: Callable) -> dict[str, Any]:
    """Extract all tool metadata from a decorated function.

    This utility function retrieves all metadata that was attached to a function
    by the @tool decorator. It's useful for inspecting tool metadata programmatically.

    Args:
        func: A function that may have been decorated with @tool.

    Returns:
        A dictionary containing all tool metadata. Keys include:
        - name: The tool name (or None if not decorated)
        - description: The tool description (or None if not decorated)
        - tags: Set of tags (empty set if not decorated or no tags)
        - provider: The provider string (or None if not set)
        - capabilities: List of capabilities (or None if not set)
        - metadata: Additional metadata dict (or None if not set)
        - parameters: Explicit parameters JSON Schema (or None if not set)

    Example:
        >>> @tool(name="my_tool", tags=["test"])
        ... def example():
        ...     '''Example function.'''
        ...     pass
        >>> metadata = get_tool_metadata(example)
        >>> print(metadata["name"])
        my_tool
        >>> print(metadata["tags"])
        {"test"}
    """
    return {
        "name": getattr(func, "_py_tool_name", None),
        "description": getattr(func, "_py_tool_description", None),
        "tags": getattr(func, "_py_tool_tags", set()),
        "provider": getattr(func, "_py_tool_provider", None),
        "capabilities": getattr(func, "_py_tool_capabilities", None),
        "metadata": getattr(func, "_py_tool_metadata", None),
        "parameters": getattr(func, "_py_tool_parameters", None),
    }


def has_tool_decorator(func: Callable) -> bool:
    """Check if a function has been decorated with @tool.

    Args:
        func: The function to check.

    Returns:
        True if the function has been decorated with @tool, False otherwise.

    Example:
        >>> @tool
        ... def decorated():
        ...     pass
        >>> has_tool_decorator(decorated)
        True
        >>> def not_decorated():
        ...     pass
        >>> has_tool_decorator(not_decorated)
        False
    """
    return hasattr(func, "_py_tool_name")
