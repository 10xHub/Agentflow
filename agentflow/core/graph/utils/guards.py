"""Execution guards: bound how long work runs, and cancel it when stopped.

Without these, two failure modes are unreachable from the execution loop:

1. **Indefinite hang.** Nothing wrapped node or tool execution in a timeout. The
   only bound was the LLM SDK's own timeout, which does not cover custom tools,
   MCP calls, or condition functions. A node blocked on a half-open socket never
   returns, so the loop never advances a step -- the recursion limit never trips
   -- and the run blocks forever.

2. **Un-cancellable run.** Stop was cooperative and polled only *between* nodes.
   If the hang is inside a node, control never returns to the loop, so the stop
   flag is never observed. `stop()` appeared to succeed while the run kept going.

`execute_with_guards` fixes both by running the work as a task it can actually
cancel: it enforces a deadline, and while waiting it periodically asks whether a
stop has been requested, cancelling the task if so.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar


logger = logging.getLogger("agentflow.graph")

T = TypeVar("T")

# How often to check for an out-of-band stop request while work is in flight.
# Small enough that a stop feels responsive, large enough not to hammer the
# checkpointer's cache on every node.
DEFAULT_STOP_POLL_INTERVAL = 1.0


async def _cancel_and_wait(task: asyncio.Task) -> None:
    """Cancel a task and wait for it to actually finish unwinding.

    Awaiting after cancel matters: it lets the coroutine run its `finally`
    blocks (closing sessions, releasing connections) before we move on, and it
    prevents an orphaned task from being garbage-collected mid-flight.
    """
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def execute_with_guards(
    work: Awaitable[T],
    *,
    timeout: float | None = None,
    stop_check: Callable[[], Awaitable[bool]] | None = None,
    poll_interval: float = DEFAULT_STOP_POLL_INTERVAL,
    on_timeout: Callable[[], Exception] | None = None,
    on_stop: Callable[[], Exception] | None = None,
) -> T:
    """Run `work` under a deadline, cancelling it if a stop is requested.

    Args:
        work: The coroutine to run (node execution, tool call, ...).
        timeout: Seconds before the work is cancelled. None disables the deadline.
        stop_check: Async predicate polled while waiting; when it returns True the
            work is cancelled. None disables stop-cancellation.
        poll_interval: How often to run `stop_check`.
        on_timeout: Builds the exception raised when the deadline passes.
        on_stop: Builds the exception raised when a stop is observed.

    Returns:
        Whatever `work` returned.

    Raises:
        Whatever `work` raised, or the exception from `on_timeout` / `on_stop`.
    """
    task: asyncio.Task = asyncio.ensure_future(work)
    loop = asyncio.get_running_loop()
    deadline = (loop.time() + timeout) if timeout else None

    try:
        while True:
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    await _cancel_and_wait(task)
                    raise (on_timeout() if on_timeout else TimeoutError("Execution timed out"))

            # Wake up either at the next stop poll or at the deadline, whichever
            # comes first. With neither, block until the work completes.
            wait_for = remaining
            if stop_check is not None:
                wait_for = poll_interval if remaining is None else min(poll_interval, remaining)

            done, _ = await asyncio.wait({task}, timeout=wait_for)

            if task in done:
                # Surfaces the work's own exception, if it raised.
                return task.result()

            if stop_check is not None and await stop_check():
                logger.info("Stop requested while work was in flight; cancelling it")
                await _cancel_and_wait(task)
                raise (on_stop() if on_stop else asyncio.CancelledError())

    except asyncio.CancelledError:
        # The caller (or the whole run) is being cancelled: don't leak the task.
        if not task.done():
            await _cancel_and_wait(task)
        raise


def resolve_timeout(config: dict[str, Any], key: str, default: float | None) -> float | None:
    """Read a timeout from the run config, falling back to a default.

    An explicit ``None`` in the config disables the timeout, which is different
    from the key being absent (use the default). A non-positive value also
    disables it, so `0` is an easy way to opt out.
    """
    if key not in config:
        return default
    value = config[key]
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r in config; falling back to %s", key, config[key], default)
        return default
    return value if value > 0 else None
