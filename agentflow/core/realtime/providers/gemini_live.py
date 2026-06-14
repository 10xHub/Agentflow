"""Gemini Live provider client.

Wraps ``client.aio.live.connect(...)`` (an async context manager yielding a live
session) behind the provider-neutral :class:`~agentflow.core.realtime.base.RealtimeClient`
protocol. ``normalize_message`` maps a google ``LiveServerMessage`` to the framework's
:data:`RealtimeEvent` union; it is duck-typed and imports no provider SDK, so it is unit
testable with lightweight stand-ins.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agentflow.core.realtime.base import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    AudioDeltaEvent,
    GoAwayEvent,
    InputTranscriptEvent,
    InterruptedEvent,
    OutputTranscriptEvent,
    RealtimeConfig,
    RealtimeEvent,
    SessionUpdateEvent,
    ToolCallEvent,
    TurnCompleteEvent,
)


logger = logging.getLogger(__name__)

_RATE_RE = re.compile(r"rate=(\d+)")


def _rate_from_mime(mime_type: str | None, default: int) -> int:
    """Extract the sample rate from a ``audio/pcm;rate=24000`` mime string."""
    if not mime_type:
        return default
    match = _RATE_RE.search(mime_type)
    return int(match.group(1)) if match else default


def normalize_message(message: Any) -> list[RealtimeEvent]:
    """Map a google ``LiveServerMessage`` to zero or more normalized events.

    Reads attributes defensively (``getattr``) so it tolerates both the real SDK objects
    and test stand-ins, and emits events in wire order within a single message.
    """
    events: list[RealtimeEvent] = []

    content = getattr(message, "server_content", None)
    if content is not None:
        model_turn = getattr(content, "model_turn", None)
        for part in getattr(model_turn, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                rate = _rate_from_mime(getattr(inline, "mime_type", None), OUTPUT_SAMPLE_RATE)
                events.append(AudioDeltaEvent(data=data, sample_rate=rate))

        in_tx = getattr(content, "input_transcription", None)
        if in_tx is not None and getattr(in_tx, "text", None) is not None:
            events.append(
                InputTranscriptEvent(
                    text=in_tx.text, finished=bool(getattr(in_tx, "finished", False))
                )
            )

        out_tx = getattr(content, "output_transcription", None)
        if out_tx is not None and getattr(out_tx, "text", None) is not None:
            events.append(
                OutputTranscriptEvent(
                    text=out_tx.text, finished=bool(getattr(out_tx, "finished", False))
                )
            )

        if getattr(content, "interrupted", None):
            events.append(InterruptedEvent())

        if getattr(content, "generation_complete", None) or getattr(content, "turn_complete", None):
            events.append(TurnCompleteEvent())

    tool_call = getattr(message, "tool_call", None)
    if tool_call is not None:
        for fc in getattr(tool_call, "function_calls", None) or []:
            events.append(
                ToolCallEvent(
                    id=getattr(fc, "id", None) or uuid4().hex,
                    name=getattr(fc, "name", "") or "",
                    args=getattr(fc, "args", None) or {},
                )
            )

    update = getattr(message, "session_resumption_update", None)
    if update is not None:
        events.append(SessionUpdateEvent(resumption_handle=getattr(update, "new_handle", None)))

    go_away = getattr(message, "go_away", None)
    if go_away is not None:
        events.append(GoAwayEvent(time_left=getattr(go_away, "time_left", None)))

    return events


class GeminiLiveClient:
    """``RealtimeClient`` implementation backed by the Gemini Live API.

    ``connector`` is the seam for testing/overrides: a callable
    ``(model=..., config=...) -> async context manager`` that yields a live session.
    In production it defaults to ``genai.Client(...).aio.live.connect``.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        connector: Any | None = None,
        api_key: str | None = None,
        use_vertex_ai: bool = False,
    ) -> None:
        self._client = client
        self._connector = connector
        self._api_key = api_key
        self._use_vertex_ai = use_vertex_ai
        self._config: RealtimeConfig | None = None
        self._cm: Any | None = None
        self._session: Any | None = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    # --- lazy provider construction (guarded optional dependency) --------- #
    @staticmethod
    def _genai():
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "google-genai SDK is required for Gemini realtime. "
                "Install it with: pip install 10xscale-agentflow[realtime]"
            ) from exc
        return genai, types

    def _ensure_client(self) -> Any:
        if self._client is None:
            genai, _ = self._genai()
            self._client = genai.Client(api_key=self._api_key, vertexai=self._use_vertex_ai)
        return self._client

    def _get_connector(self) -> Any:
        if self._connector is not None:
            return self._connector
        return self._ensure_client().aio.live.connect

    def _build_connect_config(
        self, config: RealtimeConfig, resume_handle: str | None = None
    ) -> Any:
        _, types = self._genai()
        kwargs: dict[str, Any] = {
            "response_modalities": [types.Modality(m) for m in config.response_modalities],
        }
        if config.system_instruction:
            kwargs["system_instruction"] = config.system_instruction
        if config.voice:
            kwargs["speech_config"] = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.voice)
                )
            )
        if config.input_audio_transcription:
            kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()
        if config.output_audio_transcription:
            kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
        if config.session_resumption or resume_handle:
            kwargs["session_resumption"] = types.SessionResumptionConfig(handle=resume_handle)
        if config.context_window_compression:
            kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            )
        if not config.vad.enabled:
            kwargs["realtime_input_config"] = types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
            )
        if config.tools:
            kwargs["tools"] = self._to_provider_tools(types, config.tools)
        return types.LiveConnectConfig(**kwargs)

    @staticmethod
    def _to_provider_tools(types: Any, tools: list[Any]) -> list[Any]:
        """Convert provider-neutral OpenAI-style tool dicts into Gemini tool objects.

        OpenAI-format dicts (``{"type":"function","function":{...}}``) are collected into a
        single ``types.Tool(function_declarations=[...])``. Anything else (raw callables,
        already-built ``types.Tool``/``FunctionDeclaration``) passes through untouched.
        """
        declarations: list[Any] = []
        passthrough: list[Any] = []
        for entry in tools:
            if isinstance(entry, dict) and "function" in entry:
                fn = entry["function"]
                decl_kwargs: dict[str, Any] = {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                }
                if fn.get("parameters") is not None:
                    decl_kwargs["parameters_json_schema"] = fn["parameters"]
                declarations.append(types.FunctionDeclaration(**decl_kwargs))
            else:
                passthrough.append(entry)
        result = list(passthrough)
        if declarations:
            result.append(types.Tool(function_declarations=declarations))
        return result

    # --- RealtimeClient protocol ----------------------------------------- #
    async def connect(self, config: RealtimeConfig, resume_handle: str | None = None) -> None:
        self._config = config
        connector = self._get_connector()
        live_config = self._build_connect_config(config, resume_handle=resume_handle)
        self._cm = connector(model=config.model, config=live_config)
        self._session = await self._cm.__aenter__()

    def _require_session(self) -> Any:
        if self._session is None:
            raise RuntimeError("GeminiLiveClient is not connected; call connect() first")
        return self._session

    async def send_audio(self, pcm: bytes, sample_rate: int = INPUT_SAMPLE_RATE) -> None:
        session = self._require_session()
        _, types = self._genai()
        await session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={sample_rate}")
        )

    async def send_text(self, text: str) -> None:
        session = self._require_session()
        await session.send_realtime_input(text=text)

    async def send_activity_start(self) -> None:
        session = self._require_session()
        _, types = self._genai()
        await session.send_realtime_input(activity_start=types.ActivityStart())

    async def send_activity_end(self) -> None:
        session = self._require_session()
        _, types = self._genai()
        await session.send_realtime_input(activity_end=types.ActivityEnd())

    async def send_tool_response(self, call_id: str, name: str, result: Any) -> None:
        session = self._require_session()
        _, types = self._genai()
        await session.send_tool_response(
            function_responses=[types.FunctionResponse(id=call_id, name=name, response=result)]
        )

    async def reseed_history(self, messages: list[Any]) -> None:
        session = self._require_session()
        _, types = self._genai()
        turns = []
        for message in messages:
            text = "".join(
                getattr(block, "text", "") or "" for block in getattr(message, "content", []) or []
            )
            if not text:
                continue
            role = "model" if getattr(message, "role", "user") == "assistant" else "user"
            turns.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
        if turns:
            await session.send_client_content(turns=turns, turn_complete=True)

    async def receive(self) -> AsyncIterator[RealtimeEvent]:
        session = self._require_session()
        async for message in session.receive():
            for event in normalize_message(message):
                yield event

    async def close(self) -> None:
        cm = self._cm
        if cm is None:
            return
        self._cm = None
        self._session = None
        try:
            await cm.__aexit__(None, None, None)
        except Exception:  # pragma: no cover - best-effort teardown
            logger.warning("Error while closing Gemini live session", exc_info=True)
