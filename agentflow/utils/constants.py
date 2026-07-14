"""
Constants and enums for TAF agent graph execution and messaging.

This module defines special node names, message storage levels, execution states,
and response granularity options for agent workflows.
"""

from enum import Enum
from typing import Literal


# Special node names for graph execution flow
START: Literal["__start__"] = "__start__"
END: Literal["__end__"] = "__end__"

# Execution deadlines.
#
# Nothing else bounds a node or a tool: the LLM SDK's own timeout (600s) does not
# cover custom tools, MCP calls, or condition functions, so a hung call would
# block a run forever. These are deliberately generous -- they are a backstop
# against hangs, not a latency budget -- and both can be overridden per run via
# the `node_timeout` / `tool_timeout` config keys, or disabled with None/0.
#
# The node default sits above DEFAULT_LLM_TIMEOUT_SECONDS (600s) so that a slow
# LLM call fails on its own timeout, with its own error, rather than being masked
# by the node deadline.
DEFAULT_NODE_TIMEOUT_SECONDS: float = 900.0
DEFAULT_TOOL_TIMEOUT_SECONDS: float = 300.0

# Identity used when a run carries no user_id.
#
# This used to be "test-user-id", a placeholder that reads like a real account:
# with per-user isolation enabled, every un-authenticated run was silently filed
# under that one fake user. "anonymous" matches what the API layer already
# substitutes when no auth is configured, so both layers agree on who "nobody" is.
# It is deliberately not a valid-looking user id.
DEFAULT_ANONYMOUS_USER_ID: str = "anonymous"


class ExecutionState(str, Enum):
    """
    Graph execution states for agent workflows.

    Values:
        RUNNING: Execution is in progress.
        PAUSED: Execution is paused.
        COMPLETED: Execution completed successfully.
        ERROR: Execution encountered an error.
        INTERRUPTED: Execution was interrupted.
        ABORTED: Execution was aborted.
        IDLE: Execution is idle.
    """

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    ABORTED = "aborted"
    IDLE = "idle"


class ResponseGranularity(str, Enum):
    """
    Response granularity options for agent graph outputs.

    Values:
        FULL: State, latest messages.
        PARTIAL: Context, summary, latest messages.
        LOW: Only latest messages.
    """

    FULL = "full"
    PARTIAL = "partial"
    LOW = "low"
