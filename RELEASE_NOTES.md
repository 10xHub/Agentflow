# 10xscale-agentflow 0.8.0

This release adds a **realtime audio-to-audio subsystem** (Gemini Live) and a round of
production-hardening to the core engine. No breaking changes to the turn-based API.

## Highlights

### Realtime audio-to-audio (Gemini Live)
Live, full-duplex voice sessions with barge-in and React-style tool calling. Unlike
`invoke`/`stream` (turn-based super-step traversal), a realtime graph is driven by a
separate runtime because the provider owns the turn loop.

- **Prebuilt `AudioAgent`** (`agentflow.prebuilt.agent.AudioAgent`) — a React-style
  builder mirroring `ReactAgent`'s construction surface, with tools advertised to the
  model automatically and executed reason -> tool -> respond.
- **New runtime: `CompiledGraph.arealtime(input_queue, config, state)`** (async generator
  of normalized events) and `CompiledGraph.realtime(...)` (sync wrapper). A graph
  containing a `LiveAgent` must use these; ordinary graphs continue to use `invoke`/`stream`.
- **Provider-neutral contracts** in `agentflow.core.realtime`: `RealtimeConfig`,
  `VADConfig`, the `RealtimeEvent` discriminated union (`AudioDeltaEvent`,
  `InputTranscriptEvent`/`OutputTranscriptEvent`, `ToolCallEvent`/`ToolResultEvent`,
  `TurnCompleteEvent`, `InterruptedEvent`, `SessionUpdateEvent`, `GoAwayEvent`,
  `AgentChangedEvent`, `ErrorEvent`), the `RealtimeClient` Protocol, and `LiveInputQueue`.
  OpenAI Realtime can be added later behind the same Protocol.
- **Gemini Live provider client** (`GeminiLiveClient`). Provider SDK imports are lazy —
  importing `agentflow.core.realtime` never requires the optional dependency.
- **Audio formats:** input PCM16 mono @ 16 kHz, output PCM16 @ 24 kHz. Transcripts are
  persisted as `Message`s (`metadata={"modality": "audio"}`); raw audio is never stored.
- **API server WebSocket bridge** `/v1/graph/live` (`agentflow api`) when the graph is
  rooted at a `LiveAgent`.
- **Install:** `pip install "10xscale-agentflow[realtime]"`. Set `GEMINI_API_KEY`;
  optionally `GEMINI_LIVE_MODEL` (default `gemini-live-2.5-flash-preview`).
- **Examples:** `examples/realtime/` — headless WAV-in/WAV-out, live full-duplex
  microphone, and the API WebSocket setup.

### Reliability & operations
- **Configurable LLM call timeout.** All LLM clients apply a default request timeout
  (600s) so a stalled provider cannot hang a run. Override via `AGENTFLOW_LLM_TIMEOUT`
  (seconds) or `set_default_llm_timeout(...)`; an explicit per-client `timeout=` still wins.
- **Circuit breaker for LLM calls (opt-in).** Complements retry + `fallback_models`: after
  `circuit_breaker_threshold` consecutive failures a `(provider, model)` is skipped for
  `circuit_breaker_reset_timeout` seconds. Configure via `RetryConfig`
  (`circuit_breaker_enabled`, `circuit_breaker_threshold`, `circuit_breaker_reset_timeout`).
- **`CompiledGraph` async context manager.** `async with await build_and_compile_graph()
  as graph: ...` runs `aclose()` on exit even if the body raises; `aclose()` is now
  idempotent.

### Security & typing
- **Secret redaction for logs.** New `agentflow.utils` helpers: `mask_secrets(text)`,
  `SecretRedactionFilter` (a `logging.Filter`), and
  `install_secret_redaction(logger_name="agentflow")`.
- **PEP 561 typing.** The package now ships a `py.typed` marker, so downstream
  `mypy`/`pyright` type-check against Agentflow's annotations.

### Publishers
- New `Event.REALTIME` event and `ContentType.TRANSCRIPT` content type.
- `ConsolePublisher` is documented as dev/debug, opt-in; pass `{"use_logger": True}` to
  route events through the `agentflow.publisher` logger instead of stdout.

## Project / repository
- Dependencies now carry version bounds (e.g. `pydantic>=2,<3`).
- `mypy` runs in pre-commit/CI (phased adoption; see `CONTRIBUTING.md`).
- Test coverage gate raised to 80%.
- Added `SECURITY.md`, `CONTRIBUTING.md`, Dependabot config, and a CodeQL workflow.

## Upgrade notes
- No breaking changes. New realtime functionality is additive and gated behind the
  `realtime` extra.
- Runs may now fail faster on a stalled provider due to the default 600s LLM timeout;
  raise `AGENTFLOW_LLM_TIMEOUT` if you rely on longer single calls.
