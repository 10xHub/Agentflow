"""Static checks for MCP examples that should not require live services."""

import ast
import runpy
from pathlib import Path

import pytest


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


def load_xquik_example() -> dict:
    """Load the Xquik example without running its network entry point."""
    return runpy.run_path(REPO_ROOT / "examples/xquik-mcp/client.py")


def test_xquik_example_requires_a_nonempty_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a missing or whitespace-only API key before creating a client."""
    example = load_xquik_example()
    monkeypatch.setenv("XQUIK_API_KEY", "   ")

    with pytest.raises(RuntimeError, match="Set XQUIK_API_KEY"):
        example["require_api_key"]()


def test_xquik_example_uses_the_api_key_header() -> None:
    """Send Xquik API keys through the current MCP authentication header."""
    example = load_xquik_example()
    captured = {}

    class FakeClient:
        def __init__(self, config: dict) -> None:
            captured.update(config)

    example["build_client"].__globals__["Client"] = FakeClient
    example["build_client"]("xq_example")

    server = captured["mcpServers"]["xquik"]
    assert server["headers"] == {"x-api-key": "xq_example"}
    assert "Authorization" not in server["headers"]
