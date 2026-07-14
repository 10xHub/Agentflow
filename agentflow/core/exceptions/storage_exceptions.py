"""Structured exception taxonomy for persistence & runtime layers.

These exceptions let higher-level orchestration decide retry / fail-fast logic
instead of relying on broad ``except Exception`` blocks.

All storage exceptions now support structured error responses with error codes
and contextual logging for better observability.
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger("agentflow.exceptions")


class StorageError(Exception):
    """Base class for non-retryable storage layer errors.

    Provides structured error handling with error codes and context.

    Attributes:
        message (str): Human-readable description of the error
        error_code (str): Unique error code for categorization
        context (dict): Additional contextual information about the error
    """

    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_000",
        context: dict[str, Any] | None = None,
    ):
        """Initialize a StorageError with structured error information.

        Args:
            message (str): Description of the error.
            error_code (str): Unique error code for categorization (default: "STORAGE_000")
            context (dict): Additional contextual information (default: None)
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}

        # Log the error with full context
        logger.error(
            "StorageError [%s]: %s | Context: %s",
            self.error_code,
            self.message,
            self.context,
            exc_info=True,
        )
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert the error to a structured dictionary format.

        Returns:
            dict: Structured error response with error_code, message, and context
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:
        """Return a string representation of the error."""
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"context={self.context})"
        )


class TransientStorageError(StorageError):
    """Retryable storage error (connection drops, timeouts).

    This exception indicates a temporary failure that may succeed on retry.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_TRANSIENT_000",
        context: dict[str, Any] | None = None,
    ):
        """Initialize a TransientStorageError.

        Args:
            message (str): Description of the transient error.
            error_code (str): Unique error code (default: "STORAGE_TRANSIENT_000")
            context (dict): Additional contextual information (default: None)
        """
        logger.warning(
            "TransientStorageError [%s]: %s | Context: %s",
            error_code,
            message,
            context or {},
            exc_info=True,
        )
        super().__init__(message, error_code, context)


class SerializationError(StorageError):
    """Raised when (de)serialization of state/messages fails deterministically.

    This exception indicates a permanent failure in data serialization/deserialization.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_SERIALIZATION_000",
        context: dict[str, Any] | None = None,
    ):
        """Initialize a SerializationError.

        Args:
            message (str): Description of the serialization error.
            error_code (str): Unique error code (default: "STORAGE_SERIALIZATION_000")
            context (dict): Additional contextual information (default: None)
        """
        logger.error(
            "SerializationError [%s]: %s | Context: %s",
            error_code,
            message,
            context or {},
            exc_info=True,
        )
        super().__init__(message, error_code, context)


class SchemaVersionError(StorageError):
    """Raised when schema version detection or migration application fails.

    This exception indicates issues with database schema versioning.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_SCHEMA_000",
        context: dict[str, Any] | None = None,
    ):
        """Initialize a SchemaVersionError.

        Args:
            message (str): Description of the schema version error.
            error_code (str): Unique error code (default: "STORAGE_SCHEMA_000")
            context (dict): Additional contextual information (default: None)
        """
        logger.error(
            "SchemaVersionError [%s]: %s | Context: %s",
            error_code,
            message,
            context or {},
            exc_info=True,
        )
        super().__init__(message, error_code, context)


class StaleStateError(StorageError):
    """Raised when a durable state write loses an optimistic-concurrency check.

    The writer based its update on a state version that is no longer current,
    which means another execution committed a newer state for the same thread
    in the meantime. Committing anyway would silently discard that other
    execution's work (a lost update), so the write is rejected instead.

    Callers (e.g. an API layer) should treat this as a conflict (HTTP 409):
    reload the latest state and retry, rather than overwrite blindly.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_CONFLICT_000",
        context: dict[str, Any] | None = None,
    ):
        """Initialize a StaleStateError.

        Args:
            message (str): Description of the concurrency conflict.
            error_code (str): Unique error code (default: "STORAGE_CONFLICT_000")
            context (dict): Additional contextual information (default: None)
        """
        logger.warning(
            "StaleStateError [%s]: %s | Context: %s",
            error_code,
            message,
            context or {},
        )
        super().__init__(message, error_code, context)


class ResourceNotFoundError(StorageError):
    """Raised when a requested resource is not found in storage.

    This exception indicates that the specified resource does not exist.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_NOT_FOUND_000",
        context: dict[str, Any] | None = None,
    ):
        """Initialize a ResourceNotFoundError.

        Args:
            message (str): Description of the not found error.
            error_code (str): Unique error code (default: "STORAGE_NOT_FOUND_000")
            context (dict): Additional contextual information (default: None)
        """
        logger.error(
            "ResourceNotFoundError [%s]: %s | Context: %s",
            error_code,
            message,
            context or {},
            exc_info=True,
        )
        super().__init__(message, error_code, context)


class MetricsError(Exception):
    """Raised when metrics emission fails (should normally be swallowed/logged).

    This exception is typically used for non-critical metrics reporting failures.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "METRICS_000",
        context: dict[str, Any] | None = None,
    ):
        """Initialize a MetricsError.

        Args:
            message (str): Description of the metrics error.
            error_code (str): Unique error code (default: "METRICS_000")
            context (dict): Additional contextual information (default: None)
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}

        # Log as warning since metrics errors are typically non-critical
        logger.warning(
            "MetricsError [%s]: %s | Context: %s",
            self.error_code,
            self.message,
            self.context,
        )
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert the error to a structured dictionary format.

        Returns:
            dict: Structured error response with error_code, message, and context
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:
        """Return a string representation of the error."""
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"context={self.context})"
        )
