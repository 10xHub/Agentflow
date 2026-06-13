"""Doc/example import guardrail.

Prevents regressions of the pre-refactor import drift: README and ``examples/`` must not
reference module paths that were removed in the ``core/`` / ``storage/`` / ``runtime/`` / ``qa/``
restructure, and must not use APIs that do not exist (``Message.from_text``,
``ToolNode(functions=...)``, ``Agent(tool_node_name=...)``).

Two layers:
  1. Static scan of every ``agentflow.*`` import in README + examples against a denylist of
     removed top-level shims, plus a scan for known-bad API call patterns.
  2. A live check that the canonical symbols the README now advertises are importable and real.

The static scan needs no optional dependencies and is the authoritative regression guard.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Top-level module prefixes removed in the package restructure. Any import that starts with one
# of these (followed by "." or end of token) is a dead path. Note that the canonical paths
# (agentflow.core.state, agentflow.core.graph, ...) do NOT start with any of these.
DEAD_PREFIXES = (
    "agentflow.graph",
    "agentflow.state",
    "agentflow.checkpointer",
    "agentflow.evaluation",
    "agentflow.skills",
    "agentflow.testing",
    "agentflow.adapters",
    "agentflow.publisher",
)

# API call patterns that reference symbols/keywords that do not exist.
BAD_API_PATTERNS = {
    r"\bMessage\.from_text\s*\(": "Message.from_text does not exist; use Message.text_message",
    r"\bToolNode\s*\(\s*functions\s*=": "ToolNode takes `tools` (positional), not `functions=`",
    r"\btool_node_name\s*=": "Agent uses `tool_node=`, not `tool_node_name=`",
}

_IMPORT_RE = re.compile(r"^\s*(?:from\s+(agentflow[\w.]*)\s+import|import\s+(agentflow[\w.]*))")
_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)


def _doc_files() -> list[Path]:
    files = [REPO_ROOT / "README.md"]
    examples = REPO_ROOT / "examples"
    if examples.is_dir():
        files += sorted(examples.rglob("*.md"))
        files += sorted(examples.rglob("*.py"))
    return [f for f in files if f.is_file()]


def _code_text(path: Path) -> str:
    """Return the Python source contained in a file (fenced blocks for .md, whole file for .py)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".md":
        return "\n".join(_PY_FENCE_RE.findall(text))
    return text


def _agentflow_imports(code: str) -> list[str]:
    """Yield the imported ``agentflow.*`` module path for each non-commented import line."""
    mods = []
    for line in code.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _IMPORT_RE.match(line)
        if m:
            mods.append(m.group(1) or m.group(2))
    return mods


def _is_dead(mod: str) -> bool:
    return any(mod == p or mod.startswith(p + ".") for p in DEAD_PREFIXES)


def test_no_dead_import_paths_in_docs():
    """No README/example code references a removed top-level module path."""
    violations = []
    for f in _doc_files():
        for mod in _agentflow_imports(_code_text(f)):
            if _is_dead(mod):
                violations.append(f"{f.relative_to(REPO_ROOT)}: {mod}")
    assert not violations, (
        "Dead import paths found (use agentflow.core.* / agentflow.storage.* / agentflow.qa.*):\n"
        + "\n".join(violations)
    )


def test_no_nonexistent_api_patterns_in_docs():
    """No README/example code uses an API symbol/keyword that does not exist."""
    violations = []
    for f in _doc_files():
        code = _code_text(f)
        for pattern, why in BAD_API_PATTERNS.items():
            if re.search(pattern, code):
                violations.append(f"{f.relative_to(REPO_ROOT)}: {why}")
    assert not violations, "Nonexistent API usage found:\n" + "\n".join(violations)


def test_canonical_readme_symbols_are_real():
    """The canonical symbols the README advertises import and exist."""
    from agentflow.core.graph import Agent, StateGraph, ToolNode  # noqa: F401
    from agentflow.core.state import AgentState, Message  # noqa: F401
    from agentflow.storage.checkpointer import InMemoryCheckpointer  # noqa: F401
    from agentflow.utils import ResponseGranularity, convert_messages  # noqa: F401
    from agentflow.utils.constants import END  # noqa: F401

    assert hasattr(Message, "text_message")
    assert not hasattr(Message, "from_text")
