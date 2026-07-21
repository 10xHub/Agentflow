# 10xscale-agentflow 1.0.0

First stable release. `10xscale-agentflow` is now `Development Status :: 5 -
Production/Stable`, and the public API is covered by the deprecation policy in
`CHANGELOG.md`: from here on, breaking changes bump the **major** version and ship with
a migration path.

This release is the production-hardening round on top of `0.8.0` (which introduced the
realtime audio-to-audio subsystem). The focus is correctness and durability under
concurrency, real execution bounds, and tenant isolation.

## Highlights

### Durable state you can trust under concurrency
- **Optimistic concurrency control.** `states` carries a `version` column with
  `UNIQUE (thread_id, version)`; writes take a per-thread row lock and compare-and-swap.
  A write based on a stale version raises `StaleStateError` (HTTP 409 at the API)
  instead of silently discarding another run's update.
- **Idempotent tool-execution ledger** (`tool_executions`, schema v3). A node replayed
  after a crash no longer re-fires tool calls that already completed, keyed by
  `(thread_id, origin_message_id:tool_call_id)`.
- **Per-step durable checkpointing** (`durable_checkpoint_every_step`, default on), so a
  crash replays one node rather than the whole run.
- **Real schema migrations** with a stepwise, idempotent runner guarded by
  `pg_advisory_xact_lock`, so concurrent workers cannot race the DDL.

### Bounded execution
- **Node and tool timeouts** (`node_timeout`, `tool_timeout`) that actually cancel the
  work, plus **stop-cancels-a-running-node** — previously stop was only polled *between*
  nodes, so a hang inside one was unreachable.
- **Backpressure on background tasks** (`max_pending_tasks`, default 1000). A slow or
  dead publisher sink previously grew an unbounded task set until OOM.

### Isolation & security
- **Per-user isolation in the checkpointer** (`enforce_user_isolation`, default on)
  across state, messages, and threads, plus global thread-ownership resolution.
- **File ownership.** Uploads record an owner; reads by another user 404.
- **Production refuses to start with wildcard CORS *and* credentials enabled** (API
  package). Set explicit `ORIGINS`, or `CORS_ALLOW_CREDENTIALS=false`.

### Observability
- **OpenTelemetry metrics** via `metrics.setup_otel_metrics()` — counters and histograms
  on node/tool execution, with outcome dimensions.
- **Structured, correlated logging** via `logging.setup_structured_logging()`; every
  record carries `run_id` / `thread_id` / `node`.

## Upgrade notes (0.8.0 -> 1.0.0)

- **`Default user_id is now "anonymous"`** (was `"test-user-id"`). With per-user
  isolation on, unauthenticated runs previously pooled into one placeholder identity.
- **A conditional edge whose condition raises now fails the run** (`GraphError`,
  `GRAPH_ROUTING_001`) instead of silently falling through to a static edge or `END`.
- **`injectq` is pinned to `>=0.4.0,<0.5`.** It is pre-1.0; an unbounded `0.5` pickup
  could break fresh installs.
- Durable-storage schema advances to v3 and migrates in place on first connect.

See `CHANGELOG.md` for the full list of Added / Fixed / Breaking changes.
