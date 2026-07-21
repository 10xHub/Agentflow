"""
Custom exception classes for graph operations in agentflowntflow.

This package provides:
        - GraphError: Base exception for graph-related errors.
        - NodeError: Exception for node-specific errors.
        - GraphRecursionError: Exception for recursion limit errors in graphs.
        - StorageError: Base exception for storage-related errors.
        - TransientStorageError: Exception for retryable storage errors.
        - SerializationError: Exception for serialization/deserialization errors.
        - SchemaVersionError: Exception for schema version mismatch errors.
        - MetricsError: Exception for metrics emission errors.
"""

from .graph_error import GraphError
from .node_error import NodeError
from .recursion_error import GraphRecursionError
from .storage_exceptions import (
    MetricsError,
    ResourceNotFoundError,
    SchemaVersionError,
    SerializationError,
    StaleStateError,
    StorageError,
    TransientStorageError,
)
from .timeout_error import GraphStopRequested, NodeTimeoutError


__all__ = [
    "GraphError",
    "GraphRecursionError",
    "GraphStopRequested",
    "MetricsError",
    "NodeError",
    "NodeTimeoutError",
    "ResourceNotFoundError",
    "SchemaVersionError",
    "SerializationError",
    "StaleStateError",
    "StorageError",
    "TransientStorageError",
]
