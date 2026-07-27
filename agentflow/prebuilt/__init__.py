"""Prebuilt tools and agent packages for Agentflow.

Import concrete agent implementations from ``agentflow.prebuilt.agent`` and
tool helpers from ``agentflow.prebuilt.tools``.
"""

from __future__ import annotations

# Context managers
from agentflow.core.state.message_context_manager import MessageContextManager
from agentflow.core.state.summary_context_manager import SummaryContextManager

# Agents
from .agent import (
    AudioAgent,
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    PlanActReflectAgent,
    RAGAgent,
    ReactAgent,
    StructuredOutputAgent,
    SupervisorTeamAgent,
    SwarmAgent,
    SwarmMemberConfig,
    WorkerConfig,
)

# Tools
from .tools import (
    create_handoff_tool,
    fetch_url,
    file_read,
    file_search,
    file_write,
    google_web_search,
    is_handoff_tool,
    make_agent_memory_tool,
    make_user_memory_tool,
    memory_tool,
    safe_calculator,
    vertex_ai_search,
)


__all__ = [
    # Agents
    "AudioAgent",
    "BaseReranker",
    "CohereReranker",
    "CrossEncoderReranker",
    # Context managers
    "MessageContextManager",
    "PlanActReflectAgent",
    "RAGAgent",
    "ReactAgent",
    "StructuredOutputAgent",
    "SummaryContextManager",
    "SupervisorTeamAgent",
    "SwarmAgent",
    "SwarmMemberConfig",
    "WorkerConfig",
    # Tools
    "create_handoff_tool",
    "fetch_url",
    "file_read",
    "file_search",
    "file_write",
    "google_web_search",
    "is_handoff_tool",
    "make_agent_memory_tool",
    "make_user_memory_tool",
    "memory_tool",
    "safe_calculator",
    "vertex_ai_search",
]
