"""Anthropic server-side tool definitions.

Server tools run on Anthropic's infrastructure: you declare them in ``tools``
and the results arrive as content blocks in the same response. There is no
client-side execution loop, so they never reach agentflow's ``ToolNode``.

Tool ``type`` strings are dated and model-gated. The builders here pick the
right variant for the model rather than making callers memorise which dated
string a given model accepts.
"""

from __future__ import annotations

import logging
from typing import Any

from .anthropic_request import strip_bedrock_prefix


logger = logging.getLogger("agentflow.agent.anthropic")

# Models supporting the dynamic-filtering web tool variants.
_DYNAMIC_FILTER_MODELS = frozenset(
    {
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    }
)

WEB_SEARCH_DYNAMIC = "web_search_20260209"
WEB_SEARCH_BASIC = "web_search_20250305"
WEB_FETCH_DYNAMIC = "web_fetch_20260209"
WEB_FETCH_BASIC = "web_fetch_20250910"
CODE_EXECUTION = "code_execution_20260521"

# The matching *result* block types live with the response converter
# (``runtime.adapters.llm.anthropic_converter.SERVER_TOOL_RESULT_TYPES``); that
# package is imported by this one, so the dependency only runs one way.


def _supports_dynamic_filtering(model: str) -> bool:
    return strip_bedrock_prefix(model) in _DYNAMIC_FILTER_MODELS


def web_search_tool(
    model: str,
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``web_search`` server tool definition for *model*.

    Newer models get the dynamic-filtering variant, which runs code execution
    under the hood. Do **not** also declare a separate ``code_execution`` tool
    alongside it: two execution environments confuse the model.

    ``allowed_domains`` and ``blocked_domains`` are mutually exclusive.
    """
    if allowed_domains and blocked_domains:
        raise ValueError("web_search accepts allowed_domains or blocked_domains, never both.")

    tool: dict[str, Any] = {
        "type": WEB_SEARCH_DYNAMIC if _supports_dynamic_filtering(model) else WEB_SEARCH_BASIC,
        "name": "web_search",
    }
    if max_uses is not None:
        tool["max_uses"] = max_uses
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains
    if blocked_domains:
        tool["blocked_domains"] = blocked_domains
    if user_location:
        tool["user_location"] = user_location
    return tool


def web_fetch_tool(
    model: str,
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    citations: dict[str, Any] | None = None,
    max_content_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a ``web_fetch`` server tool definition for *model*.

    Web fetch only retrieves URLs already present in the conversation.
    """
    if allowed_domains and blocked_domains:
        raise ValueError("web_fetch accepts allowed_domains or blocked_domains, never both.")

    tool: dict[str, Any] = {
        "type": WEB_FETCH_DYNAMIC if _supports_dynamic_filtering(model) else WEB_FETCH_BASIC,
        "name": "web_fetch",
    }
    if max_uses is not None:
        tool["max_uses"] = max_uses
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains
    if blocked_domains:
        tool["blocked_domains"] = blocked_domains
    if citations:
        tool["citations"] = citations
    if max_content_tokens is not None:
        tool["max_content_tokens"] = max_content_tokens
    return tool


def code_execution_tool() -> dict[str, Any]:
    """Build a ``code_execution`` server tool definition.

    Returns ``bash_code_execution_tool_result`` blocks, not the legacy bare
    ``code_execution_tool_result``.
    """
    return {"type": CODE_EXECUTION, "name": "code_execution"}


def is_server_tool(tool: Any) -> bool:
    """True when *tool* is already an Anthropic server-tool definition."""
    if not isinstance(tool, dict):
        return False
    ttype = tool.get("type", "")
    return isinstance(ttype, str) and ttype.startswith(
        ("web_search_", "web_fetch_", "code_execution_", "tool_search_tool_")
    )
