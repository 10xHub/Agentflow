"""
Checkpointer adapters for agent state persistence in agentflow.

This module exposes unified checkpointing interfaces for agent graphs, supporting
in-memory, Postgres-backed, and SQLite-backed persistence.

Exports:
    BaseCheckpointer: Abstract base class for checkpointing implementations.
    InMemoryCheckpointer: In-memory checkpointing for development/testing.
    PgCheckpointer: Postgres+Redis checkpointing (optional, requires extras).
    SqliteCheckpointer: Single-file SQLite checkpointing for client-side /
        single-user agents (optional, requires extras).

Usage:
    PgCheckpointer requires: pip install 10xscale-agentflow[pg_checkpoint]
    SqliteCheckpointer requires: pip install 10xscale-agentflow[sqlite_checkpoint]
"""

from .base_checkpointer import BaseCheckpointer
from .in_memory_checkpointer import InMemoryCheckpointer
from .pg_checkpointer import PgCheckpointer
from .sqlite_checkpointer import SqliteCheckpointer


__all__ = [
    "BaseCheckpointer",
    "InMemoryCheckpointer",
    "PgCheckpointer",
    "SqliteCheckpointer",
]
