"""Tests for the core authz contract (config["authz"])."""

from __future__ import annotations

from agentflow.core.authz import (
    CHECKPOINT_READ,
    GRAPH_INVOKE,
    SCOPE_NONE,
    SCOPE_OWNER,
    build_authz,
    get_authz,
    has_scope,
    isolation_scope,
)


def test_build_authz_shape():
    a = build_authz("u1", scope=SCOPE_OWNER, scopes=[CHECKPOINT_READ])
    assert a == {"user_id": "u1", "scope": "owner", "scopes": ["checkpointer:read"]}


def test_build_authz_bad_scope_falls_back_to_none():
    assert build_authz("u1", scope="bogus")["scope"] == SCOPE_NONE


def test_isolation_scope_reads_block():
    cfg = {"authz": build_authz("u1", scope=SCOPE_OWNER)}
    assert isolation_scope(cfg) == "owner"
    cfg2 = {"authz": build_authz("u1", scope=SCOPE_NONE)}
    assert isolation_scope(cfg2) == "none"


def test_isolation_scope_absent_is_none():
    assert isolation_scope({}) is None
    assert isolation_scope({"authz": {"scope": "weird"}}) is None
    assert isolation_scope(None) is None


def test_has_scope():
    cfg = {"authz": build_authz("u1", scopes=[GRAPH_INVOKE])}
    assert has_scope(cfg, GRAPH_INVOKE) is True
    assert has_scope(cfg, CHECKPOINT_READ) is False


def test_has_scope_permissive_when_no_block():
    # No authz block -> allow (single-user / direct SDK usage).
    assert has_scope({}, GRAPH_INVOKE) is True


def test_get_authz_ignores_malformed():
    assert get_authz({"authz": "not-a-dict"}) is None
    assert get_authz("nope") is None


# ---------------------------------------------------------------------------
# Store isolation helper (BaseStore._scope_user_id honors the policy)
# ---------------------------------------------------------------------------


def _min_store():
    from agentflow.storage.store.base_store import BaseStore

    class _S(BaseStore):
        async def asetup(self):
            pass

        async def astore(self, *a, **k):
            pass

        async def asearch(self, *a, **k):
            pass

        async def aget(self, *a, **k):
            pass

        async def aget_all(self, *a, **k):
            pass

        async def aupdate(self, *a, **k):
            pass

        async def adelete(self, *a, **k):
            pass

        async def aforget_memory(self, *a, **k):
            pass

    return _S()


def test_store_scope_user_id_honors_policy():
    s = _min_store()
    owner = {"user": {"authz": build_authz("A", scope=SCOPE_OWNER)}}
    none = {"user": {"authz": build_authz("A", scope=SCOPE_NONE)}}
    assert s._scope_user_id(owner, "A") == "A"      # owner -> scope to A
    assert s._scope_user_id(none, "A") is None       # allow_all -> do not scope
    assert s._scope_user_id({}, "A") == "A"          # no policy -> safe default (scope)
