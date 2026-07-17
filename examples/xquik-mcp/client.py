"""Discover Xquik tools through Agentflow's MCP adapter."""

import asyncio
import os

from fastmcp import Client

from agentflow.core import ToolNode


MCP_URL = "https://xquik.com/mcp"


def require_api_key() -> str:
    """Read the Xquik API key without embedding it in source."""
    api_key = os.getenv("XQUIK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set XQUIK_API_KEY before running this example")
    return api_key


def build_client(api_key: str) -> Client:
    """Build a Streamable HTTP client for the remote Xquik MCP server."""
    return Client(
        {
            "mcpServers": {
                "xquik": {
                    "url": MCP_URL,
                    "transport": "streamable-http",
                    "headers": {"x-api-key": api_key},
                }
            }
        }
    )


async def main() -> None:
    """Load the remote tool schemas through Agentflow."""
    tool_node = ToolNode([], client=build_client(require_api_key()))
    tools = await tool_node.all_tools()
    names = sorted(tool["function"]["name"] for tool in tools)
    print(f"Available Xquik tools: {', '.join(names)}")


if __name__ == "__main__":
    asyncio.run(main())
