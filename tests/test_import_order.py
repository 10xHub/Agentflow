"""Import-order regression guard.

``agentflow.core.graph`` imports back into ``agentflow.utils`` and
``agentflow.storage.checkpointer``. Historically that made those modules unimportable as the
*first* import in a fresh interpreter (``ImportError: ... partially initialized module``), because
``agentflow.core`` eagerly pulled in ``graph``. ``graph`` is now loaded lazily (PEP 562
``__getattr__`` in ``agentflow/core/__init__.py``) so every public entry point imports cleanly in
any order.

Each case runs in a *fresh* subprocess — importing in-process would not catch the bug once pytest
has already loaded ``agentflow.core``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


# Public entry points that must import cleanly as the very first import in a fresh interpreter.
FIRST_IMPORTS = [
    "import agentflow.utils",
    "from agentflow.utils import CallbackManager, convert_messages, tool",
    "import agentflow.storage",
    "import agentflow.storage.checkpointer",
    "from agentflow.storage.checkpointer import InMemoryCheckpointer, BaseCheckpointer",
    "import agentflow.core",
    "from agentflow.core import StateGraph, Agent, ToolNode, CompiledGraph, AgentState, Message",
    "from agentflow.core.graph import Agent, StateGraph, ToolNode, CompiledGraph",
    "from agentflow.core.state import AgentState, Message",
    "import agentflow.runtime.publisher",
    "import agentflow.qa.evaluation",
    "import agentflow.qa.testing",
]


@pytest.mark.parametrize("statement", FIRST_IMPORTS)
def test_importable_as_first_import(statement: str):
    """The statement succeeds when it is the only thing a fresh interpreter imports."""
    result = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"`{statement}` failed as a first import:\n{result.stderr}"
    )


def test_lazy_graph_symbol_identity():
    """The lazily-resolved aggregate symbol is the same object as the direct submodule symbol."""
    code = (
        "from agentflow.core import StateGraph as A\n"
        "from agentflow.core.graph import StateGraph as B\n"
        "assert A is B, 'aggregate symbol is not the submodule symbol'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
