# Xquik MCP Example

This example connects Agentflow's `ToolNode` to the remote Xquik MCP server and
loads its tool schemas without changing any framework defaults.

Install Agentflow's MCP extra, provide an API key through the environment, and
run the example:

```bash
uv sync --extra mcp
export XQUIK_API_KEY="your-api-key"
uv run python examples/xquik-mcp/client.py
```

The script prints the available tool names. Use the same `ToolNode` as an
agent's `tool_node` when building a graph. See the
[Xquik MCP overview](https://docs.xquik.com/mcp/overview) for authentication,
tool behavior, and safety requirements.
