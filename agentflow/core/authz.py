"""Authorization contract carried inside ``config["authz"]``.

This is the single source of truth for authorization in Agentflow. The **contract** (what
the block looks like) and the **enforcement primitives** (``should_isolate`` / ``has_scope``)
live here in the core library, so every consumer -- checkpointer, store, graph nodes, tools --
enforces identically whether the call came from the API or from a direct SDK call.

Producers of the block are trusted and layer-specific:

- **API**: builds it from a verified identity + authorization backend and overwrites any
  client-supplied value (non-hijackable).
- **SDK** (direct ``CompiledGraph.invoke``): the developer builds it (their process is the
  trust boundary), or passes nothing and gets the permissive default below.

Shape::

    config["authz"] = {
        "user_id": "u1",
        "scope": "owner" | "none",  # data isolation policy
        "scopes": ["checkpointer:read", "graph:invoke", ...],
    }

When ``config["authz"]`` is absent the defaults are permissive (isolation falls back to the
storage backend's own setting; every scope is allowed) -- so existing direct-SDK usage is
unchanged.
"""

from __future__ import annotations

from typing import Any


AUTHZ_KEY = "authz"

# --- Scopes (resource:action). The full, canonical set. ---------------------------------

# graph (execution) -- gated at the API layer; the checkpointer/store are the data backstop.
GRAPH_INVOKE = "graph:invoke"
GRAPH_STREAM = "graph:stream"
GRAPH_STOP = "graph:stop"
GRAPH_FIX = "graph:fix"
GRAPH_SETUP = "graph:setup"
GRAPH_READ = "graph:read"

# checkpointer (thread state / messages / threads)
CHECKPOINT_READ = "checkpointer:read"
CHECKPOINT_WRITE = "checkpointer:write"
CHECKPOINT_DELETE = "checkpointer:delete"

# memory / long-term store
MEMORY_READ = "memory:read"
MEMORY_WRITE = "memory:write"
MEMORY_DELETE = "memory:delete"

# files / media
FILES_READ = "files:read"
FILES_UPLOAD = "files:upload"

# misc
CONFIG_READ = "config:read"

ALL_SCOPES = frozenset(
    {
        GRAPH_INVOKE,
        GRAPH_STREAM,
        GRAPH_STOP,
        GRAPH_FIX,
        GRAPH_SETUP,
        GRAPH_READ,
        CHECKPOINT_READ,
        CHECKPOINT_WRITE,
        CHECKPOINT_DELETE,
        MEMORY_READ,
        MEMORY_WRITE,
        MEMORY_DELETE,
        FILES_READ,
        FILES_UPLOAD,
        CONFIG_READ,
    }
)

# Valid isolation scopes.
SCOPE_OWNER = "owner"
SCOPE_NONE = "none"


def get_authz(config: Any) -> dict[str, Any] | None:
    """Return the ``authz`` block from config, or None if absent/malformed.

    Looks in two places: a top-level ``config["authz"]`` (convenient for direct SDK calls)
    and ``config["user"]["authz"]`` -- where the API stamps it, because the trusted ``user``
    object is copied into ``config["user"]`` on every request, so a single stamp there
    reaches every downstream call.
    """
    if not isinstance(config, dict):
        return None
    block = config.get(AUTHZ_KEY)
    if isinstance(block, dict):
        return block
    user = config.get("user")
    if isinstance(user, dict):
        block = user.get(AUTHZ_KEY)
        if isinstance(block, dict):
            return block
    return None


def isolation_scope(config: Any) -> str | None:
    """Return the isolation scope (``"owner"``/``"none"``) or None when unset.

    None means "no policy present" -- callers apply their own default (a storage backend
    falls back to its ``enforce_user_isolation`` setting).
    """
    block = get_authz(config)
    if block is not None:
        scope = block.get("scope")
        if scope in (SCOPE_OWNER, SCOPE_NONE):
            return scope
    return None


def has_scope(config: Any, scope: str) -> bool:
    """Whether the request carries ``scope``. Permissive when no authz block is present."""
    block = get_authz(config)
    if block is None:
        return True  # no policy -> allow (single-user / direct SDK)
    scopes = block.get("scopes")
    return isinstance(scopes, list | tuple | set | frozenset) and scope in scopes


def build_authz(
    user_id: str | int | None,
    *,
    scope: str = SCOPE_NONE,
    scopes: Any = (),
) -> dict[str, Any]:
    """Build a trusted authz block to place at ``config["authz"]``."""
    if scope not in (SCOPE_OWNER, SCOPE_NONE):
        scope = SCOPE_NONE
    return {"user_id": user_id, "scope": scope, "scopes": list(scopes)}
