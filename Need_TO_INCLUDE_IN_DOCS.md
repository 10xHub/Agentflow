# Need to Include in Docs

Production-readiness work completed in this pass. Items below introduce or change
user-facing behavior and should be documented.

## Type checking (PEP 561)
- The package now ships a `py.typed` marker. Downstream `mypy`/`pyright` will
  type-check against Agentflow's annotations.

## Configurable LLM call timeout
- All LLM clients now apply a default request timeout (`600s`) so a stalled
  provider cannot hang a run indefinitely.
- Override globally via the `AGENTFLOW_LLM_TIMEOUT` environment variable
  (seconds), or programmatically:
  - `from agentflow.core.llm import set_default_llm_timeout, get_default_llm_timeout, DEFAULT_LLM_TIMEOUT_SECONDS`
  - `set_default_llm_timeout(120.0)` / `set_default_llm_timeout(None)` to reset.
- An explicit per-client `timeout=` still takes precedence.

## CompiledGraph async context manager
- `CompiledGraph` supports `async with`:
  ```python
  async with await build_and_compile_graph() as graph:
      await graph.ainvoke(input_data)
  # aclose() runs automatically on exit, even if the body raises
  ```
- `aclose()` is now idempotent (second call returns `{"status": "already_closed"}`).

## Circuit breaker for LLM calls (opt-in)
- Complements retry + `fallback_models`: once a `(provider, model)` fails
  `circuit_breaker_threshold` times in a row, its circuit opens and that target
  is skipped (straight to the next fallback) for `circuit_breaker_reset_timeout`
  seconds, instead of being retried on every call.
- Configure via `RetryConfig`:
  - `circuit_breaker_enabled: bool = False`
  - `circuit_breaker_threshold: int = 5`
  - `circuit_breaker_reset_timeout: float = 30.0`

## Secret redaction for logs
- New helpers in `agentflow.utils`:
  - `mask_secrets(text)` — redacts API keys, `Bearer` tokens, `key=value`
    secrets, and signed-URL credential query params.
  - `SecretRedactionFilter` — a `logging.Filter`; add it to a handler to cover
    all loggers that propagate to it.
  - `install_secret_redaction(logger_name="agentflow")` — convenience installer.

## ConsolePublisher logging option
- `ConsolePublisher` is a dev/debug, opt-in publisher (use a real transport in
  production). It writes to stdout by default; pass `{"use_logger": True}` to
  route events through the `agentflow.publisher` logger instead of stdout.

## Realtime audio-to-audio (Gemini Live)
- New subsystem for live, audio-to-audio sessions. Unlike `invoke`/`stream`
  (turn-based super-step traversal), a realtime graph is driven by a separate
  runtime because the provider owns the turn loop.
- **Install:** `pip install "10xscale-agentflow[realtime]"` (pulls `google-genai`).
  Set `GEMINI_API_KEY`; optionally `GEMINI_LIVE_MODEL`
  (default `gemini-live-2.5-flash-preview`). Provider SDK imports are lazy, so
  importing `agentflow.core.realtime` never requires the `realtime` extra.
- **Audio formats:** input PCM16 mono @ 16 kHz; output PCM16 @ 24 kHz. Transcripts
  are persisted as `Message`s (`metadata={"modality": "audio"}`); raw audio is
  never stored.
- **Image / video input:** still images and video frames are sent straight to the
  live model as JPEG frames via `LiveInputQueue.send_image(...)` (see below); like
  ADK, there is no media store/offload in the realtime path. Image frames are sent
  live to the model but are not persisted to history (a session reconnect reseeds
  text transcripts only).

### Prebuilt `AudioAgent`
- `from agentflow.prebuilt.agent import AudioAgent` — React-style builder mirroring
  `ReactAgent`'s construction surface, wrapping a `LiveAgent` as the graph root.
  ```python
  app = AudioAgent(
      "gemini-live-2.5-flash-preview",
      realtime_config=RealtimeConfig(model="gemini-live-2.5-flash-preview", voice="Puck"),
      tools=[my_tool],          # advertised to the model automatically; runs React-style
  ).compile()
  ```
- Tools work like a normal `ToolNode` (reason -> tool -> respond, including barge-in).
  No sub-agents/handoff in v1.
- `system_prompt`, `skills`, and `memory` work the same as `ReactAgent`: the agent's
  `system_prompt` (plus the skills trigger table / session-mode skill content and the
  memory system prompt) is flattened into the single Gemini Live `system_instruction`
  at connect, and `{field}` placeholders are interpolated from state exactly like the
  turn-based path. Skill/memory **tools** are advertised to the model normally.
  - Realtime caveat: `system_instruction` is fixed for the session, so state-dependent
    content (session-mode skill from a state field, memory preload) is a connect-time
    snapshot, not re-evaluated per turn. Mid-session dynamism goes through `set_skill`
    / memory tools, which work continuously.
- `compile()` takes `checkpointer`, `store`, `callback_manager`, `shutdown_timeout`.
  It does **not** take `media_store` or `interrupt_before`/`interrupt_after` — those
  belong to the turn-based super-step executor, which realtime bypasses.

### Driving a session: `CompiledGraph.arealtime` / `realtime`
- `arealtime(input_queue, config=None, state=None)` is an async generator yielding
  normalized `RealtimeEvent`s. `realtime(...)` is a sync wrapper (must run with no
  active event loop).
- Forcing rule: the graph must contain exactly one `LiveAgent`; ordinary graphs
  raise. Conversely a graph containing a `LiveAgent` must use `arealtime()` —
  `invoke`/`stream` raise.
  ```python
  queue = LiveInputQueue()
  queue.send_audio(pcm16_bytes)   # non-blocking; safe from an audio callback
  async for event in app.arealtime(queue, {"thread_id": "t1"}):
      ...                         # AudioDeltaEvent / transcripts / ToolCallEvent / ...
  queue.close()                   # ends the session once the provider goes idle
  ```

### Public API (`agentflow.core.realtime`)
- `LiveInputQueue` / `LiveInput` / `LiveInputKind` — non-blocking upstream input
  queue. `send_audio`, `send_text`, `send_image` (still image / video frame, default
  mime `image/jpeg`), `send_activity_start`, `send_activity_end`, `close` (all
  synchronous, callable from any context).
- `RealtimeConfig` — per-session config: `model`, `response_modalities`
  (exactly one per session; default `["AUDIO"]`), `voice`, `system_instruction`,
  `input_audio_transcription`, `output_audio_transcription`, `vad` (`VADConfig`),
  `reconnect` (`ReconnectConfig`), `context_window_compression`, `session_resumption`,
  `tools`, `tools_tags`.
- `VADConfig` — voice-activity detection; disable for push-to-talk (manual activity).
- `ReconnectConfig` — reconnect/backoff policy for a dropped socket: `base_delay`
  (default `0.5`), `max_delay` (default `10.0`), `max_attempts` (default `5`;
  set `0` to disable error-driven reconnect). See "Reconnection & resumption".
- `RealtimeEvent` — discriminated union (keyed on `type`) of: `AudioDeltaEvent`,
  `InputTranscriptEvent`, `OutputTranscriptEvent`, `ToolCallEvent`, `ToolResultEvent`,
  `TurnCompleteEvent`, `InterruptedEvent` (barge-in), `SessionUpdateEvent`,
  `GoAwayEvent`, `AgentChangedEvent`, `ErrorEvent`.
- `RealtimeClient` — provider Protocol (one implementation per provider).
- `GeminiLiveClient` / `normalize_message` — the Gemini Live provider client.
- Provider neutrality: contracts import no provider SDK; OpenAI Realtime can be
  added later behind the same `RealtimeClient` Protocol.

### Reconnection & resumption
- Reconnect is automatic inside the realtime runtime (the builder/`AudioAgent` wires
  nothing; you don't drive it).
  - Provider `go_away` (planned rotation): reconnect immediately, no backoff.
  - Transient drop / receive error: exponential backoff
    `min(base_delay * 2**(n-1), max_delay)`, up to `max_attempts`, then a fatal
    `ErrorEvent` (`code="reconnect_failed"`) ends the session.
- Tunable per session via `RealtimeConfig.reconnect` (`ReconnectConfig`):
  ```python
  from agentflow.core.realtime import RealtimeConfig, ReconnectConfig
  RealtimeConfig(model="...", reconnect=ReconnectConfig(base_delay=0.25, max_attempts=8))
  ```
- Context across reconnects: Gemini streams a resumption handle (`session_update`)
  that is stored and persisted to checkpointer thread metadata; reconnect resumes
  provider-side context (requires `session_resumption=True`, the default). With no
  handle (e.g. a fresh session on the same `thread_id`), persisted transcript history
  is reseeded instead. Cross-session resume therefore needs a checkpointer.

### Session & turn lifecycle hooks
- Realtime fires graph/turn hooks through the same `GraphLifecycleHook` used by
  turn-based graphs (register via `CallbackManager.register_lifecycle_hook` and pass
  the manager to `compile(callback_manager=...)`). These methods are no-ops for
  `invoke`/`stream` and only fire in realtime:
  - `on_graph_start(ctx, state)` / `on_graph_end(ctx, final_state, messages, total_steps)`
    — once per session (the `LIVE` node *is* the graph). `total_steps` = number of turns.
  - `on_turn_start(ctx, state, turn_index)` / `on_turn_end(ctx, state, turn_index)` —
    per model turn (1-based; a turn spans one model generation, bounded by
    `turn_complete` or a barge-in). A turn cut off by session end still gets a balanced
    `on_turn_end`.
- All hooks may return a modified state to replace the current one (same semantics as
  the graph hooks).
- Tool/MCP `before/after/error` callbacks fire as usual (tools run through `ToolNode`).
  There is no `AI`-invocation callback or input-validator pass in realtime (no discrete
  LLM call); `on_turn_start`/`on_turn_end` are the per-turn observability stand-in.

### API server WebSocket bridge (`/v1/graph/live`)
- `agentflow api` exposes a realtime WebSocket at `ws://<host>/v1/graph/live`
  when the configured graph is rooted at a `LiveAgent`.
- First frame: a JSON control frame (e.g. `{"model": "...", "thread_id": "abc",
  "voice": "Puck"}`); present fields override the agent's build-time config for
  that session.
- Upstream: binary frame = PCM16 input audio; JSON control frame =
  `{"type": "text" | "activity_start" | "activity_end" | "close", ...}`.
  (Image/video input is currently SDK-only via `LiveInputQueue.send_image`; the
  WebSocket bridge does not forward image frames yet.)
- Downstream: binary frame = PCM16 model audio; JSON text frame = every other
  event (transcripts, `turn_complete`, `interrupted`, `tool_call`, session/
  `go_away`, `error`).

### Event/publisher additions
- New `Event.REALTIME` event and `ContentType.TRANSCRIPT` content type in
  `agentflow.runtime.publisher.events`.

### Examples
- `examples/realtime/`: headless WAV-in/WAV-out (`audio_agent_file.py`), live
  full-duplex microphone with React-style tool calling (`audio_agent_mic.py`),
  and the API WebSocket setup (`agentflow.json` + `graph.py`). See
  `examples/realtime/README.md`.

## Project / repo
- Dependencies now have version bounds (e.g. `pydantic>=2,<3`).
- mypy runs in pre-commit/CI (phased adoption; see `CONTRIBUTING.md`).
- Test coverage gate raised to 80%.
- Added `SECURITY.md` and `CONTRIBUTING.md`.
- Added Dependabot config and a CodeQL workflow.
