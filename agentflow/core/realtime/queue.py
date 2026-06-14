"""Upstream input decoupler for realtime sessions.

``LiveInputQueue`` wraps an ``asyncio.Queue`` of ``LiveInput`` transport frames. ``put``
is synchronous and non-blocking (``put_nowait``) so the input side keeps accepting audio
while the model is still generating -- the precondition for barge-in. A fresh queue is
created per session; it is the object an SDK user (or the API bridge) feeds.

``LiveInput`` is a deliberately lightweight ``dataclass`` (not a ``Message`` and not a
pydantic model): these are ephemeral control frames headed for the provider socket, never
persisted and produced on the audio hot path (~50/sec). Conversation state and the
checkpointer are driven separately, from provider *transcripts* turned into ``Message``s
(see design section 7) -- not from this queue.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from agentflow.core.realtime.base import INPUT_SAMPLE_RATE


logger = logging.getLogger(__name__)

LiveInputKind = Literal["audio", "text", "activity_start", "activity_end", "close"]


@dataclass(slots=True)
class LiveInput:
    """A single upstream transport frame. Construct via ``LiveInputQueue.send_*``.

    ``kind`` discriminates the frame; only the fields relevant to that kind are set
    (``data``/``sample_rate`` for audio, ``text`` for text, neither for control frames).
    """

    kind: LiveInputKind
    data: bytes | None = None
    text: str | None = None
    sample_rate: int = INPUT_SAMPLE_RATE


class LiveInputQueue:
    """A non-blocking, single-session input queue feeding the realtime pump task.

    Producers call the synchronous ``send_*`` / ``close`` methods from any context
    (e.g. an audio callback). The pump task consumes via ``get`` / ``async for``.
    Once closed, further ``put``s are dropped (logged at debug), never raised.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[LiveInput] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _put(self, item: LiveInput) -> None:
        if self._closed and item.kind != "close":
            logger.debug("LiveInputQueue is closed; dropping %s frame", item.kind)
            return
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("LiveInputQueue full; dropping %s frame", item.kind)

    def send_audio(self, data: bytes, sample_rate: int = INPUT_SAMPLE_RATE) -> None:
        self._put(LiveInput(kind="audio", data=data, sample_rate=sample_rate))

    def send_text(self, text: str) -> None:
        self._put(LiveInput(kind="text", text=text))

    def send_activity_start(self) -> None:
        self._put(LiveInput(kind="activity_start"))

    def send_activity_end(self) -> None:
        self._put(LiveInput(kind="activity_end"))

    def close(self) -> None:
        """Signal end of input. Idempotent; enqueues a single ``close`` sentinel frame."""
        if self._closed:
            return
        self._put(LiveInput(kind="close"))
        self._closed = True

    async def get(self) -> LiveInput:
        return await self._queue.get()

    def get_nowait(self) -> LiveInput:
        return self._queue.get_nowait()

    async def __aiter__(self) -> AsyncIterator[LiveInput]:
        while True:
            item = await self._queue.get()
            if item.kind == "close":
                return
            yield item
