# Changelog

All notable changes to `10xscale-agentflow` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/). From `1.0.0` on, the
public API is stable: breaking changes bump the **major** version and follow the
deprecation policy below.

## Deprecation policy

Starting from this release:

- A public API is never removed without first being deprecated for at least one
  minor release. Deprecated APIs emit a `DeprecationWarning` naming the
  replacement, and keep working until removal.
- Modules that move keep a back-compat shim for at least one minor release rather
  than being deleted outright. (The `agentflow.graph` / `agentflow.state` /
  `agentflow.checkpointer` moves in an earlier release shipped with no shims and
  simply started raising `ModuleNotFoundError`; that will not happen again.)
- Breaking changes are listed under a `### Breaking` heading, with the migration.

---

## [Unreleased]

## [1.0.0] - 2026-07-19

First stable release. The public API is now covered by the deprecation policy above,
and the package is classified `Development Status :: 5 - Production/Stable`. This
release is the production-hardening round on top of `0.8.0`: durable state gets
optimistic concurrency and an idempotent tool ledger, execution is bounded by real
timeouts and cancellation, and per-user isolation is enforced across the storage
layer.

### Breaking

- **`injectq` is pinned to `>=0.4.0,<0.5`.** It is pre-1.0, so a `0.5` release may
  break the API; without an upper bound it would have been picked up automatically
  and broken fresh installs.
- **Default `user_id` is now `"anonymous"`** (was `"test-user-id"`). A run with no
  `user_id` previously filed itself under a placeholder that looks like a real
  account, which, with per-user isolation enabled, silently pooled every
  unauthenticated run into one identity.
- **A conditional edge whose condition raises now fails the run** (`GraphError`,
  `GRAPH_ROUTING_001`). Previously the exception was swallowed and the graph fell
  through to the first static edge or `END` — silently taking a path nobody chose.
- **Production refuses to start with wildcard CORS *and* credentials enabled**
  (API package). Set explicit `ORIGINS`, or `CORS_ALLOW_CREDENTIALS=false`.

### Added

- **Optimistic concurrency control on durable state.** `states` now carries a
  `version` column with `UNIQUE (thread_id, version)`; writes take a per-thread row
  lock and compare-and-swap. A write based on a stale version raises the new
  `StaleStateError` (HTTP 409 at the API) instead of silently discarding another
  run's update.
- **Durable tool-execution ledger** (`tool_executions`, schema v3). A node replayed
  after a crash no longer re-fires tool calls that already completed — the
  double-charge scenario. Keyed by
  `(thread_id, origin_message_id:tool_call_id)`, because a `tool_call_id` alone is
  not unique across turns.
- **Per-step durable checkpointing** (`durable_checkpoint_every_step`, default on),
  so a crash replays one node rather than the whole run.
- **Node and tool timeouts** (`node_timeout`, `tool_timeout`) that actually cancel
  the work, plus **stop-cancels-a-running-node** — previously stop was only polled
  *between* nodes, so a hang inside one was unreachable.
- **Real schema migrations** with a stepwise, idempotent runner and a
  `pg_advisory_xact_lock`, so concurrent workers cannot race the DDL.
- **Per-user isolation in the checkpointer** (`enforce_user_isolation`, default on)
  across state, messages, and threads.
- **File ownership.** Uploads record an owner; reads by another user 404.
- **Backpressure on background tasks** (`max_pending_tasks`, default 1000). A slow
  or dead publisher sink previously grew an unbounded task set until OOM.
- **OpenTelemetry metrics** via `metrics.setup_otel_metrics()`; counters and
  histograms on node/tool execution, with outcome dimensions.
- **Structured, correlated logging** via `logging.setup_structured_logging()`; every
  record carries `run_id` / `thread_id` / `node`, so one run can be grepped out of a
  busy server.
- `agentflow build --k8s` generates a Kubernetes manifest whose termination grace
  period is long enough that a rolling deploy does not kill in-flight runs.

### Fixed

- **Lost updates on concurrent writes to one thread** (see CAS above). Reads were
  also non-deterministic: `ORDER BY created_at DESC` with no tiebreak.
- **The realtime cache could be moved backwards**, wedging a thread until its TTL
  expired. Cache writes are now an atomic version-guarded compare-and-set, and a
  lost version check invalidates the cache so the thread self-heals.
- **Parallel tools clobbering each other's state.** Each tool now runs on its own
  branch copy, merged back field-by-field against a baseline, using a field's
  reducer when it has one.
- **One failing tool orphaned its siblings** (`gather` without `return_exceptions`),
  and malformed tool arguments raised `JSONDecodeError` through the whole node.
- **Retries on non-retryable errors.** Status classification matched `"500"` as a
  substring, so `max_tokens must be <= 500` was treated as a server error.
- **Cross-tenant reads/deletes** of state, messages, threads, and files.
- **Rate limit bypass.** The bucket key came from the leftmost `X-Forwarded-For`
  entry, which the caller controls — a new value per request meant a new bucket and
  no limit at all. Proxy hops are now counted from the right.
- Blocking `urllib.urlopen` inside `async def` (cloud media store) stalled the event
  loop for every concurrent run in the process.
- Connection-pool and Qdrant-collection cold-start races (double creation).
- Schema-version failures were swallowed instead of raised.
