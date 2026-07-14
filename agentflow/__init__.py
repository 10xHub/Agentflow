"""Agentflow: a graph-based orchestration framework for multi-agent LLM systems.

This module is intentionally light. Importing ``agentflow`` does NOT eagerly pull
in submodules -- import the subpackage you need:

    from agentflow.core.graph import StateGraph, Agent
    from agentflow.core.state import AgentState, Message
    from agentflow.storage.checkpointer import PgCheckpointer

Only ``__version__`` is exposed here. It is resolved from the installed
distribution metadata rather than hardcoded, so there is a single source of truth
(``pyproject.toml``) and the reported version cannot drift from what is actually
installed -- which is exactly how the previous 0.8.0-vs-0.7.5.1 mismatch arose.
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version


try:
    __version__ = _dist_version("10xscale-agentflow")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"


__all__ = ["__version__"]
