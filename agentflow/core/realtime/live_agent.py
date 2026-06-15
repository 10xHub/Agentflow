"""LiveAgent -- the realtime (audio-to-audio) node and root session controller.

``LiveAgent`` subclasses :class:`BaseAgent` and reuses the skills/memory builder mixins,
but it deliberately **excludes** the text turn loop (``AgentExecutionMixin``): realtime
inverts control (the provider owns the turn loop), so it writes its own duplex loop.

It is entered by ``CompiledGraph.arealtime`` (Phase 3) via :meth:`arun`, which:

1. opens one provider socket (the spine, held for the whole session),
2. runs a pump task (queue -> provider) concurrently with a receive loop,
3. dispatches tool calls through the existing :class:`ToolNode` (callbacks + publisher
   events fire inside ToolNode, so transparency is identical to text mode),
4. persists finished transcripts as ``Message``s (no audio at rest), and
5. transparently reconnects on ``go_away``/drop using the cached resumption handle.

Calling the agent through the turn-based engine (``execute``) raises -- the forcing rule.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Literal

from agentflow.core.graph.agent_internal.memory import AgentMemoryMixin
from agentflow.core.graph.agent_internal.skills import AgentSkillsMixin
from agentflow.core.graph.base_agent import BaseAgent
from agentflow.core.graph.tool_node import ToolNode
from agentflow.core.llm import detect_provider
from agentflow.core.realtime.base import (
    ErrorEvent,
    InputTranscriptEvent,
    OutputTranscriptEvent,
    RealtimeClient,
    RealtimeConfig,
    ToolCallEvent,
    ToolResultEvent,
)
from agentflow.core.realtime.providers.gemini_live import GeminiLiveClient
from agentflow.core.state import AgentState, Message, TextBlock, add_messages
from agentflow.runtime.publisher.events import ContentType, Event, EventModel, EventType
from agentflow.runtime.publisher.publish import publish_event
from agentflow.utils import CallbackManager


if TYPE_CHECKING:
    from agentflow.core.realtime.base import RealtimeEvent
    from agentflow.core.realtime.queue import LiveInputQueue
    from agentflow.core.state import BaseContextManager
    from agentflow.storage.checkpointer import BaseCheckpointer

logger = logging.getLogger(__name__)


class LiveAgent(AgentSkillsMixin, AgentMemoryMixin, BaseAgent):
    """Realtime audio agent node. Run via :meth:`arun` (or ``CompiledGraph.arealtime``)."""

    def __init__(
        self,
        model: str,
        *,
        realtime_config: RealtimeConfig | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        tool_node: str | ToolNode | None = None,
        skills: Any | None = None,
        memory: Any | None = None,
        realtime_client_factory: Callable[[], RealtimeClient] | None = None,
        **kwargs: Any,
    ) -> None:
        api_key: str | None = kwargs.pop("api_key", None)
        use_vertex_ai: bool = kwargs.pop("use_vertex_ai", False)
        project: str | None = kwargs.pop("project", None)
        location: str | None = kwargs.pop("location", None)

        provider = detect_provider(model, use_vertex_ai)
        if provider != "google":
            raise ValueError(
                "LiveAgent v1 supports only Gemini Live (google provider); "
                f"resolved provider '{provider}' for model '{model}'."
            )

        super().__init__(
            model=model,
            system_prompt=system_prompt or [],
            tool_node=tool_node,
            **kwargs,
        )

        self.provider = "google"
        self.use_vertex_ai = use_vertex_ai
        self.realtime_config = realtime_config or RealtimeConfig(model=model)

        # Tool wiring (we don't use AgentExecutionMixin._setup_tools).
        self._tool_node: ToolNode | None = tool_node if isinstance(tool_node, ToolNode) else None
        self.tool_node_name: str | None = tool_node if isinstance(tool_node, str) else None

        # One client per *connection*; the factory lets reconnects get a fresh socket
        # and lets tests inject a fake provider.
        self._client_factory: Callable[[], RealtimeClient] = realtime_client_factory or (
            lambda: GeminiLiveClient(
                api_key=api_key,
                use_vertex_ai=use_vertex_ai,
                project=project,
                location=location,
            )
        )
        self._active_client: RealtimeClient | None = None
        self._resume_handle: str | None = None
        # Serializes upstream sends against reconnect (close+connect) so the pump never
        # sends on a socket being torn down, and always picks up the reconnected client.
        self._send_lock = asyncio.Lock()

        # Per-session transcript accumulators (provider streams partial chunks; we flush
        # the concatenation on the finished marker). Reset at the start of each arun().
        self._input_transcript_buf = ""
        self._output_transcript_buf = ""

        # Error-driven reconnect backoff (go_away reconnects are immediate; only transient
        # drops back off). Instance attributes so tests can shrink them.
        self._reconnect_base_delay = 0.5
        self._reconnect_max_delay = 10.0
        self._reconnect_max_attempts = 5

        # Builder mixins (no-op when their config is None).
        self._setup_memory(memory)
        self._setup_skills(skills)

    # ------------------------------------------------------------------ #
    # Forcing rule: a live agent is not a turn-based node.
    # ------------------------------------------------------------------ #
    async def execute(self, state: AgentState, config: dict[str, Any]) -> Any:
        raise RuntimeError(
            "LiveAgent runs via CompiledGraph.arealtime(); it is not a turn-based node. "
            "Use .arealtime() (or the AudioAgent prebuilt), not invoke/stream."
        )

    async def _call_llm(
        self, messages: list[dict[str, Any]], tools: list | None = None, **kwargs: Any
    ) -> Any:
        # Realtime never makes a discrete turn-based LLM call; the provider owns the loop.
        raise RuntimeError(
            "LiveAgent has no discrete LLM call; audio turns are driven by the realtime socket."
        )

    def _resolve_tool_node(self) -> ToolNode | None:
        return self._tool_node

    # ------------------------------------------------------------------ #
    # The duplex realtime loop.
    # ------------------------------------------------------------------ #
    async def arun(
        self,
        input_queue: LiveInputQueue,
        config: dict[str, Any],
        state: AgentState | None = None,
        *,
        checkpointer: BaseCheckpointer | None = None,
        callback_manager: CallbackManager | None = None,
        context_manager: BaseContextManager | None = None,
    ) -> AsyncIterator[RealtimeEvent]:
        """Open the session and yield normalized events until the queue/session closes."""
        state = state if state is not None else AgentState()
        if callback_manager is None:
            callback_manager = CallbackManager()
        self._input_transcript_buf = ""
        self._output_transcript_buf = ""
        rt = self._session_realtime_config(config)
        rt = await self._resolve_session_tools(rt)

        handle = await self._load_resume_handle(config, checkpointer)
        client = self._client_factory()
        await client.connect(rt, resume_handle=handle)
        self._active_client = client
        # Only reseed when the provider did NOT restore context from a handle; otherwise the
        # model would receive the whole conversation twice (handle restore + reseed).
        await self._maybe_reseed(config, checkpointer, context_manager, resumed=handle is not None)

        # Closing the input queue ends the session: the pump sets this when it drains the
        # close sentinel, and the receive loop stops once the provider goes idle.
        stop_event = asyncio.Event()
        pump_task = asyncio.create_task(self._pump(input_queue, stop_event))
        attempts = 0  # consecutive error-driven reconnect attempts (reset on healthy receive)
        try:
            while True:
                reconnect = False
                forced = False  # go_away: reconnect even after input closed, to finish the turn
                received_any = False
                try:
                    async for event in self._receive_until_stop(self._active_client, stop_event):
                        received_any = True
                        for out in await self._handle_event(
                            event, config, state, checkpointer, callback_manager
                        ):
                            yield out
                        if event.type == "go_away":
                            reconnect = True
                            forced = True
                            break
                        if event.type == "error" and getattr(event, "fatal", False):
                            return
                except Exception:
                    # Transient drop: only resume if input is still open (avoid an
                    # infinite reconnect storm once the session is shutting down).
                    logger.warning("realtime receive loop error; attempting resume", exc_info=True)
                    reconnect = True

                # A turn that produced events means the connection is healthy again.
                if received_any:
                    attempts = 0

                resumable = reconnect and rt.session_resumption
                if not (resumable and (forced or not stop_event.is_set())):
                    break

                attempts, fatal = await self._attempt_reconnect(rt, forced, attempts)
                if fatal is not None:
                    yield fatal
                    break
        finally:
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task
            await self._active_client.close()

    def _session_realtime_config(self, config: dict[str, Any]) -> RealtimeConfig:
        """Merge per-session overrides (``config["realtime"]``) over the agent's base config.

        Lets a caller (e.g. the API init frame) pick model/voice/modalities/vad per session
        without rebuilding the agent. Unknown keys are ignored; the result is re-validated.
        """
        overrides = (config or {}).get("realtime") or {}
        if not overrides:
            return self.realtime_config
        base = self.realtime_config.model_dump()
        for key, value in overrides.items():
            if value is not None and key in base:
                base[key] = value
        return RealtimeConfig.model_validate(base)

    async def _resolve_session_tools(self, rt: RealtimeConfig) -> RealtimeConfig:
        """Advertise the agent's ToolNode tools to the provider so the model can call them.

        No-op when tools were set explicitly on the config (the caller wins) or when there
        is no ToolNode. Tools are emitted as provider-neutral OpenAI-style dicts (the same
        shape the turn-based path uses); the provider client converts them. ``rt.tools_tags``
        filters which tools are advertised.
        """
        if rt.tools is not None:
            return rt
        tool_node = self._resolve_tool_node()
        if tool_node is None:
            return rt
        tags = set(rt.tools_tags) if rt.tools_tags else None
        schemas = await tool_node.all_tools(tags=tags)
        if not schemas:
            return rt
        return rt.model_copy(update={"tools": schemas})

    async def _receive_until_stop(
        self, client: RealtimeClient, stop_event: asyncio.Event
    ) -> AsyncIterator[RealtimeEvent]:
        """Yield provider events, but return when ``stop_event`` fires *and* the provider
        is idle. Already-available events are always drained first (a closed input queue
        must not truncate the model's in-flight response)."""
        receiver = client.receive().__aiter__()
        stop_task = asyncio.ensure_future(stop_event.wait())
        try:
            while True:
                next_task = asyncio.ensure_future(receiver.__anext__())
                done, _ = await asyncio.wait(
                    {next_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if next_task in done:
                    try:
                        yield next_task.result()
                        continue
                    except StopAsyncIteration:
                        return
                # Provider idle and stop requested: abandon the pending receive and end.
                next_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await next_task
                return
        finally:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task

    # ------------------------------------------------------------------ #
    # Pump: upstream queue -> provider socket.
    # ------------------------------------------------------------------ #
    async def _pump(
        self, input_queue: LiveInputQueue, stop_event: asyncio.Event | None = None
    ) -> None:
        try:
            async for item in input_queue:
                # Hold the lock across the send so a concurrent reconnect can't swap the
                # socket mid-send; re-read the client *inside* the lock to use the live one.
                async with self._send_lock:
                    client = self._active_client
                    if client is None:
                        continue
                    try:
                        if item.kind == "audio" and item.data is not None:
                            await client.send_audio(item.data, item.sample_rate)
                        elif item.kind == "text" and item.text is not None:
                            await client.send_text(item.text)
                        elif item.kind == "activity_start":
                            await client.send_activity_start()
                        elif item.kind == "activity_end":
                            await client.send_activity_end()
                    except Exception:
                        logger.warning(
                            "realtime pump send failed for %s frame", item.kind, exc_info=True
                        )
        finally:
            # Input is exhausted (queue closed): let the receive loop end once idle.
            if stop_event is not None:
                stop_event.set()

    # ------------------------------------------------------------------ #
    # Per-event handling. Returns the caller-facing events to yield.
    # ------------------------------------------------------------------ #
    async def _handle_event(
        self,
        event: RealtimeEvent,
        config: dict[str, Any],
        state: AgentState,
        checkpointer: BaseCheckpointer | None,
        callback_manager: CallbackManager,
    ) -> list[RealtimeEvent]:
        if event.type == "tool_call":
            return await self._run_tool(event, config, state, callback_manager)

        if event.type == "input_transcript":
            return await self._accumulate_transcript(event, "user", config, state, checkpointer)
        if event.type == "output_transcript":
            return await self._accumulate_transcript(
                event, "assistant", config, state, checkpointer
            )

        if event.type == "interrupted":
            # Barge-in discards the model's in-flight turn; drop any partial transcript so
            # the restarted turn isn't concatenated onto the abandoned one.
            self._input_transcript_buf = ""
            self._output_transcript_buf = ""
            self._publish_realtime(EventType.INTERRUPTED, config, ContentType.AUDIO, "barge_in")
        elif event.type == "go_away":
            self._publish_realtime(EventType.UPDATE, config, ContentType.UPDATE, "go_away")
        elif event.type == "session_update":
            self._resume_handle = event.resumption_handle
            await self._persist_handle(config, checkpointer)
            self._publish_realtime(EventType.UPDATE, config, ContentType.UPDATE, "session_resumed")

        return [event]

    async def _accumulate_transcript(
        self,
        event: InputTranscriptEvent | OutputTranscriptEvent,
        role: Literal["user", "assistant"],
        config: dict[str, Any],
        state: AgentState,
        checkpointer: BaseCheckpointer | None,
    ) -> list[RealtimeEvent]:
        """Accumulate streamed transcript chunks and consolidate on the finished marker.

        Providers stream transcripts as partial chunks (``finished=False``) and end with a
        finished marker that usually carries no text. Partials pass through unchanged for live
        display; on ``finished`` we persist the full turn and emit a single consolidated
        event carrying the complete text (so consumers get the whole transcript without
        having to accumulate themselves). Turns with no transcribed text emit nothing.
        """
        is_user = role == "user"
        if is_user:
            self._input_transcript_buf += event.text
            buffered = self._input_transcript_buf
        else:
            self._output_transcript_buf += event.text
            buffered = self._output_transcript_buf

        if not event.finished:
            return [event]  # stream the partial for live UIs

        full = buffered.strip()
        if is_user:
            self._input_transcript_buf = ""
        else:
            self._output_transcript_buf = ""
        if not full:
            return []  # nothing transcribed this turn; drop the empty finish marker

        await self._persist_transcript(full, role, config, state, checkpointer)
        lifecycle = "input_transcript" if is_user else "output_transcript"
        self._publish_realtime(EventType.RESULT, config, ContentType.TRANSCRIPT, lifecycle)
        event_cls = InputTranscriptEvent if is_user else OutputTranscriptEvent
        return [event_cls(text=full, finished=True)]

    async def _run_tool(
        self,
        event: ToolCallEvent,
        config: dict[str, Any],
        state: AgentState,
        callback_manager: CallbackManager,
    ) -> list[RealtimeEvent]:
        tool_node = self._resolve_tool_node()
        if tool_node is None:
            result: Any = {"error": f"no tools registered for '{event.name}'"}
        else:
            invoked = await tool_node.invoke(
                event.name,
                event.args,
                event.id,
                config,
                state,
                callback_manager=callback_manager,
            )
            result = self._extract_tool_result(invoked)

        # Socket stays open; feed the result back to the model. Hold _send_lock (and re-read
        # the live client inside it) so this send is serialized against the pump and any
        # concurrent reconnect, exactly like the pump's own sends.
        async with self._send_lock:
            client = self._active_client
            if client is not None:
                await client.send_tool_response(event.id, event.name, result)
        return [event, ToolResultEvent(id=event.id, result=result)]

    @staticmethod
    def _extract_tool_result(invoked: Any) -> dict[str, Any]:
        if isinstance(invoked, Message):
            for block in invoked.content:
                if getattr(block, "type", None) == "tool_result":
                    return {"result": getattr(block, "output", None)}
            return {"result": None}
        if isinstance(invoked, dict):
            return invoked
        return {"result": invoked}

    # ------------------------------------------------------------------ #
    # Transcript persistence (Message only; audio is never stored at rest).
    # ------------------------------------------------------------------ #
    async def _persist_transcript(
        self,
        text: str,
        role: Literal["user", "assistant"],
        config: dict[str, Any],
        state: AgentState,
        checkpointer: BaseCheckpointer | None,
    ) -> None:
        msg = Message(role=role, content=[TextBlock(text=text)], metadata={"modality": "audio"})
        state.context = add_messages(state.context, [msg])
        if checkpointer is not None:
            await checkpointer.aput_messages(config, [msg])

    # ------------------------------------------------------------------ #
    # Resumption: within-session reconnect + cross-session reseed.
    # ------------------------------------------------------------------ #
    async def _load_resume_handle(
        self, config: dict[str, Any], checkpointer: BaseCheckpointer | None
    ) -> str | None:
        if checkpointer is None or not self.realtime_config.session_resumption:
            return None
        try:
            thread = await checkpointer.aget_thread(config)
        except Exception:
            return None
        if thread and thread.metadata:
            handle = thread.metadata.get("resumption_handle")
            self._resume_handle = handle
            return handle
        return None

    async def _persist_handle(
        self, config: dict[str, Any], checkpointer: BaseCheckpointer | None
    ) -> None:
        if checkpointer is None:
            return
        from agentflow.utils.thread_info import ThreadInfo

        try:
            thread = await checkpointer.aget_thread(config)
        except Exception:
            thread = None
        metadata = dict(thread.metadata or {}) if thread else {}
        metadata["resumption_handle"] = self._resume_handle
        info = ThreadInfo(
            thread_id=config.get("thread_id", ""),
            user_id=config.get("user_id"),
            metadata=metadata,
        )
        await checkpointer.aput_thread(config, info)

    async def _maybe_reseed(
        self,
        config: dict[str, Any],
        checkpointer: BaseCheckpointer | None,
        context_manager: BaseContextManager | None,
        *,
        resumed: bool = False,
    ) -> None:
        # When the provider restored context from a resumption handle, reseeding would
        # replay the whole conversation a second time.
        if checkpointer is None or resumed:
            return
        try:
            history = await checkpointer.alist_messages(config)
        except Exception:
            history = None
        if not history:
            return
        if context_manager is not None:
            try:
                trimmed = await context_manager.atrim_context(AgentState(context=list(history)))
                history = trimmed.context
            except Exception:
                logger.warning("context compression failed during reseed; using raw history")
        if history and self._active_client is not None:
            await self._active_client.reseed_history(list(history))

    async def _attempt_reconnect(
        self, rt: RealtimeConfig, forced: bool, attempts: int
    ) -> tuple[int, ErrorEvent | None]:
        """Reconnect after a drop. Returns ``(attempts, fatal_error_or_None)``.

        ``forced`` (go_away) is an expected provider rotation: reconnect promptly, no backoff.
        Error-driven drops back off exponentially with a hard attempt cap so a flapping or
        down provider can't spin a tight reconnect storm; once the cap is hit a fatal
        :class:`ErrorEvent` is returned for the caller to surface before ending the session.
        """
        if forced:
            with contextlib.suppress(Exception):
                await self._reconnect(rt)
            return 0, None

        attempts += 1
        if attempts > self._reconnect_max_attempts:
            logger.error(
                "realtime reconnect attempts exhausted (%d); giving up",
                self._reconnect_max_attempts,
            )
            return attempts, ErrorEvent(
                code="reconnect_failed",
                message=(
                    "realtime session lost and could not be resumed after "
                    f"{self._reconnect_max_attempts} attempts"
                ),
                fatal=True,
            )

        delay = min(self._reconnect_base_delay * (2 ** (attempts - 1)), self._reconnect_max_delay)
        await asyncio.sleep(delay)
        with contextlib.suppress(Exception):
            await self._reconnect(rt)
        return attempts, None

    async def _reconnect(self, rt: RealtimeConfig) -> None:
        async with self._send_lock:
            old = self._active_client
            with contextlib.suppress(Exception):
                if old is not None:
                    await old.close()
            client = self._client_factory()
            await client.connect(rt, resume_handle=self._resume_handle)
            self._active_client = client

    # ------------------------------------------------------------------ #
    # Observability for events ToolNode doesn't already publish.
    # ------------------------------------------------------------------ #
    def _publish_realtime(
        self,
        event_type: EventType,
        config: dict[str, Any],
        content_type: ContentType,
        lifecycle: str,
    ) -> None:
        publish_event(
            EventModel.default(
                config,
                data={},
                content_type=[content_type],
                event=Event.REALTIME,
                event_type=event_type,
                node_name=getattr(self, "_node_name", "LIVE"),
                extra={"lifecycle": lifecycle, "modality": "audio"},
            )
        )
