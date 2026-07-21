"""Pytest configuration and fixtures for all tests.

This module provides common fixtures and setup for the entire test suite.
"""

import os

import pytest
from injectq import InjectQ

from agentflow.core.graph.node import Node


_ORIGINAL_NODE_INIT = Node.__init__


def pytest_addoption(parser):
    """Register the --integration flag.

    Integration tests were previously gated on a `--integration` option that was
    never actually registered, so the gate could never be satisfied and every
    durable-storage test was permanently skipped. Registering it here is what
    makes those tests runnable at all.
    """
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require real services (Postgres, Redis).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip `integration`-marked tests unless --integration is passed.

    This replaces per-class `@pytest.mark.skipif(True, ...)` hardcodes, which
    could never be turned on. Now the gate is real: pass --integration (with the
    services running) and the durable-storage tests actually execute.
    """
    if config.getoption("--integration"):
        return

    skip_integration = pytest.mark.skip(
        reason="requires real services; run with --integration (see tests/integration/)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def isolate_injectq_container():
    """Give every test a clean dependency-injection container.

    `InjectQ.get_instance()` is a process-wide singleton, and tests bind real
    instances into it (publishers, checkpointers, task managers). Nothing reset
    it between tests, so bindings leaked forward: a test that bound a publisher
    left it installed for everything that ran afterwards. That is why the suite
    passed per-file but failed in full-suite order -- a test asserting on its own
    spy publisher was silently still publishing into a previous test's binding.

    Resetting the singleton around each test makes the suite order-independent,
    which is a precondition for the suite meaning anything at all.
    """
    InjectQ.reset_instance()
    yield
    InjectQ.reset_instance()


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
