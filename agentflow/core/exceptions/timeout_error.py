import logging
from typing import Any

from .node_error import NodeError


logger = logging.getLogger("agentflow.exceptions")


class NodeTimeoutError(NodeError):
    """
    Raised when a node or tool exceeds its allotted execution time.

    Without a bound, a node that hangs (a half-open socket in an MCP tool, a
    custom tool that never returns) blocks the graph forever: the loop never
    advances a step, so the recursion limit never trips and the cooperative stop
    check -- which only runs between nodes -- is never reached.

    Timing the call out converts that indefinite hang into a normal node error
    that the execution loop can persist, report, and recover from.

    Example:
        >>> raise NodeTimeoutError(
        ...     message="Node 'fetch' exceeded its 300s timeout",
        ...     error_code="NODE_TIMEOUT_001",
        ...     context={"node_name": "fetch", "timeout": 300.0},
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: str = "NODE_TIMEOUT_000",
        context: dict[str, Any] | None = None,
    ):
        """
        Initialize a NodeTimeoutError.

        Args:
            message (str): Description of the timeout.
            error_code (str): Unique error code (default: "NODE_TIMEOUT_000")
            context (dict): Additional contextual information (default: None)
        """
        super().__init__(message, error_code, context)


class GraphStopRequested(Exception):  # noqa: N818
    """
    Signals that a stop was requested while a node was still running.

    This is control flow, not a failure: the execution loop catches it, marks the
    run stopped, persists that, and returns normally. It exists because a stop
    observed *during* a node cannot be handled by the between-nodes stop check --
    the node has to be cancelled first, and the loop needs to tell that
    cancellation apart from a genuine error.
    """

    def __init__(self, node_name: str | None = None):
        self.node_name = node_name
        super().__init__(f"Stop requested during node '{node_name}'")
