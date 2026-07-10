"""Static checks for MCP examples that should not require live services."""

import ast
from pathlib import Path


MCP_AGENT_EXAMPLES = (
    "examples/github-mcp/git_mcp.py",
    "examples/github-mcp/mcp_file_download.py",
    "examples/react-mcp/react-mcp.py",
)
REPO_ROOT = Path(__file__).parents[2]


def test_mcp_examples_use_current_agent_tool_keyword() -> None:
    """Keep MCP examples aligned with Agent's current public constructor."""
    for relative_path in MCP_AGENT_EXAMPLES:
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        agent_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Agent"
        ]

        assert agent_calls, f"No Agent call found in {relative_path}"
        for call in agent_calls:
            keywords = {keyword.arg for keyword in call.keywords}
            assert "tools" not in keywords, relative_path
            assert "tool_node" in keywords, relative_path
