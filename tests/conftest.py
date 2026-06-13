"""Pytest configuration and fixtures for all tests.

This module provides common fixtures and setup for the entire test suite.
"""

import os

import pytest

from agentflow.core.graph.node import Node


_ORIGINAL_NODE_INIT = Node.__init__


def _compat_node_init(self, name, func, publisher=None):
    """Test-only compatibility shim for legacy Node(name, func, publisher) calls."""
    _ORIGINAL_NODE_INIT(self, name, func)
    self.publisher = publisher


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up environment variables for testing.
    
    This fixture automatically runs for all test sessions and sets
    dummy API keys to prevent test failures due to missing credentials.
    This is test-only setup and does not affect production code.
    """
    # Set dummy OpenAI API key for tests
    # Using a valid-looking but fake key that won't make actual API calls
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-testing-only")
    
    # Set dummy Google API key for tests
    os.environ.setdefault("GEMINI_API_KEY", "dummy-gemini-key-for-testing-only")

    # Vertex AI selection must NOT be inherited from a developer's .env / shell.
    # Agent() reads GOOGLE_GENAI_USE_VERTEXAI as the default for use_vertex_ai, so
    # an ambient "true" would make provider auto-detection resolve to "google" for
    # every model and break deterministic unit tests. Force it off for the suite;
    # tests that exercise Vertex pass use_vertex_ai=True explicitly.
    os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

    # Keep tests compatible while core graph transitions from
    # Node(name, func, publisher) to Node(name, func).
    Node.__init__ = _compat_node_init
    
    yield

    Node.__init__ = _ORIGINAL_NODE_INIT
    
    # Note: We don't clean up the environment variables since they're test-only
    # and won't affect any other processes
