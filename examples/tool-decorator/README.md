# Tool Decorator Examples

This directory contains examples demonstrating the use of the `@tool` decorator in the agentflow framework.

## Overview

The `@tool` decorator allows you to easily register Python functions as tools with rich metadata, including:
- Custom tool names
- Descriptions for AI agents
- Tags for categorization and filtering
- Provider information
- Capability requirements
- Additional custom metadata

## Files

### `basic_decorator_usage.py`

Demonstrates the basic usage patterns of the `@tool` decorator:
- Simple tool decoration with just a name
- Using the decorator without arguments
- Full metadata specification
- Async tool support
- Tag-based filtering
- Metadata inspection utilities

**Run it:**
```bash
python examples/tool-decorator/basic_decorator_usage.py
```

### `react_agent_with_decorator.py`

Shows how to use decorated tools with a React agent to build a functional multi-tool agent:
- Defining tools with the decorator
- Integrating decorated tools with the React agent
- Multiple tool calls in a single conversation
- Complex multi-step tasks

**Requirements:**
- Set `OPENAI_API_KEY` environment variable

**Run it:**
```bash
export OPENAI_API_KEY='your-key-here'
python examples/tool-decorator/react_agent_with_decorator.py
```

## Key Concepts

### Basic Decoration

```python
from agentflow.utils import tool

@tool("my_tool_name")
def my_function(param: str) -> str:
    """This is the tool description."""
    return f"Result: {param}"
```

### Full Metadata

```python
@tool(
    name="advanced_tool",
    description="A more complex tool with metadata",
    tags=["category1", "category2"],
    provider="custom",
    capabilities=["capability1"],
    metadata={"key": "value"}
)
def advanced_function(x: int) -> int:
    return x * 2
```

### Tag Filtering

```python
from agentflow.core.graph import ToolNode

# Create tools with different tags
@tool(name="read_tool", tags=["database", "read"])
def read_data(): pass

@tool(name="write_tool", tags=["database", "write"])
def write_data(): pass

# Filter by tags
tool_node = ToolNode([read_tool, write_tool])
read_only_tools = tool_node.get_local_tool(tags={"read"})
```

### Injectable Parameters

Some parameters are automatically provided by the framework and excluded from the tool schema:
- `state: AgentState` - Current agent state
- `config: dict` - Configuration dictionary
- `tool_call_id: str` - The ID of the tool call

```python
@tool("stateful_tool")
def my_tool(user_input: str, state: AgentState | None = None) -> str:
    # 'state' won't appear in the tool schema
    # Only 'user_input' will be required from the AI
    return f"Processed: {user_input}"
```

## Benefits

1. **Clean API**: Decorator syntax is intuitive and Pythonic
2. **Metadata-rich**: Attach comprehensive information to tools
3. **Flexible Filtering**: Use tags to selectively enable/disable tool groups
4. **Framework Integration**: Seamlessly works with ToolNode and agent builders
5. **Type-safe**: Leverages Python type hints for automatic schema generation
6. **Introspection**: Helper functions to inspect tool metadata at runtime

## Advanced Usage

### Metadata Inspection

```python
from agentflow.utils import get_tool_metadata, has_tool_decorator

# Check if function is decorated
if has_tool_decorator(my_function):
    # Get all metadata
    metadata = get_tool_metadata(my_function)
    print(f"Name: {metadata['name']}")
    print(f"Tags: {metadata['tags']}")
```

### Dynamic Tool Registration

```python
# Tools can be registered dynamically
tools = []
if user_has_database_access:
    tools.extend([read_tool, write_tool])
else:
    tools.append(read_tool)

tool_node = ToolNode(tools)
```

## Related Documentation

- [Core decorators module](../../agentflow/utils/decorators.py)
- [ToolNode documentation](../../agentflow/graph/tool_node/)
- [React agent examples](../react/)
