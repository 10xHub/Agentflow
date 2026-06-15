# Realtime Audio-to-Audio Agent Support — Design

Date: 2026-06-14
Status: Approved design (pending spec review)

## Scope of v1

- Provider: **Gemini Live API only**. OpenAI Realtime plugs into the same `RealtimeClient` seam later.
- Agent: single prebuilt `AudioAgent` (React-style). **No sub-agents / multi-agent handoff in v1.**
- Surfaces: **Python SDK is the primary, standalone surface.** The HTTP/WebSocket API is **optional
  but recommended** — a thin convenience layer over the SDK, not required to use realtime.
- Auth (when the API is used): proxy-through-server.
- Persistence: transcripts only (no audio stored at rest).

## 1. Problem

Agentflow supports text and image through a turn-based graph engine (`CompiledGraph.invoke/astream`
yielding final results / `StreamChunk`). There is no realtime audio-to-audio (speech in, speech out).
We add bidirectional realtime voice, backed by Gemini Live, as a first-class part of the **same**
framework — same build/compile/tools/state/checkpointer/publisher — so developers learn one structure
and only the call differs.

## 2. Why realtime is a separate runtime, not the invoke/stream loop

The graph's `invoke`/`stream` engine traverses nodes super-step by super-step: **we** decide the next
node after each LLM call. Realtime inverts control — the **provider** owns the turn loop, decides turn
boundaries via VAD, can chain tool calls itself, and streams audio continuously with barge-in. There
is no per-turn control point for the executor to traverse edges.

Therefore realtime is a **separate runtime** on the compiled graph: `arealtime()`. In it, the live
(audio) agent is the **root controller** that holds one provider WebSocket open for the whole session.
Tool calls and any sub-node/sub-agent invocations happen **inside** that live block; the socket is
never torn down to run them. Other nodes are callable units the live session dispatches — they do
**not** run in the invoke/stream loop.

This keeps one framework (not two engines) while respecting who actually owns the turn loop.

## 3. Layering (SDK-first)

```
[Python SDK — primary, standalone]
   AudioAgent (prebuilt)  ->  LiveAgent (node)  ->  CompiledGraph.arealtime(queue, config)
   developer feeds an audio input queue, consumes RealtimeEvent async iterator
        |
        |  (uses) agentflow/core/realtime/
        |     - LiveInputQueue   (upstream decoupler)
        |     - RealtimeClient    (provider-neutral protocol)
        |     - GeminiLiveClient  (wraps google-genai client.aio.live)
        v
   Google Gemini Live API   (one WebSocket per session = the spine)

[HTTP/WS API — optional, recommended]
   agentflow-api: /v1/graph/live  bridges a client WebSocket <-> LiveInputQueue <-> arealtime()
   adds auth, transport, browser access. NOT required: the SDK runs realtime by itself.
```

Hard boundary: **core never imports FastAPI.** `LiveAgent` owns only the **provider** socket. The
**client** WebSocket (when the API is used) lives in `agentflow-api` and bridges to `arealtime()`
through the queue. A pure-Python user supplies their own audio source (mic, file, PThread) into the
queue and consumes events directly — no server involved.

## 4. Core components (`agentflow/core/realtime/`)

### 4.1 `base.py` — provider-neutral contracts

`RealtimeEvent` — normalized event everything downstream consumes (discriminated union):

| type | payload | meaning |
|---|---|---|
| `audio_delta` | `bytes` PCM16, `sample_rate` | model audio out |
| `input_transcript` | `text`, `finished` | user speech transcript |
| `output_transcript` | `text`, `finished` | model speech transcript |
| `tool_call` | `id`, `name`, `args` | provider requests a tool |
| `tool_result` | `id`, `result` | tool finished (observability) |
| `turn_complete` | — | model finished a turn |
| `interrupted` | — | barge-in; client flushes playback |
| `session_update` | `resumption_handle` | provider resume token |
| `go_away` | `time_left` | provider will close socket soon |
| `agent_changed` | `author` | active agent changed (future multi-agent) |
| `error` | `code`, `message` | provider error |

`RealtimeClient` — Protocol, one impl per provider:

```python
class RealtimeClient(Protocol):
    async def connect(self, config: RealtimeConfig) -> None: ...
    async def send_audio(self, pcm: bytes, sample_rate: int) -> None: ...
    async def send_text(self, text: str) -> None: ...
    async def send_activity_start(self) -> None: ...   # manual VAD / push-to-talk
    async def send_activity_end(self) -> None: ...
    async def send_tool_response(self, call_id: str, name: str, result: Any) -> None: ...
    def receive(self) -> AsyncIterator[RealtimeEvent]: ...
    async def close(self) -> None: ...
```

`RealtimeConfig` — per-session value object: `model`, `response_modalities` (single: `AUDIO`|`TEXT`),
`voice`, `system_instruction`, `input_audio_transcription`, `output_audio_transcription`, `vad`
(auto + sensitivity, or disabled for push-to-talk), `context_window_compression`,
`session_resumption`, `tools`, `tools_tags`.

### 4.2 `providers/gemini_live.py` — GeminiLiveClient

Wraps `client.aio.live.connect(model=..., config=types.LiveConnectConfig(...))` as an async context
manager. Mapping:

- `send_audio` -> `session.send_realtime_input(audio=types.Blob(data=..., mime_type="audio/pcm;rate=16000"))`
- `send_activity_start/end` -> `session.send_realtime_input(activity_start=ActivityStart())` / `activity_end`
- `send_tool_response` -> `session.send_tool_response(function_responses=[types.FunctionResponse(...)])`
- `receive()` maps `LiveServerMessage` -> `RealtimeEvent`:
  - `server_content.model_turn.parts[].inline_data` -> `audio_delta` (24kHz)
  - `server_content.input_transcription` / `output_transcription` -> transcripts
  - `tool_call.function_calls[]` -> `tool_call`
  - `server_content.interrupted` -> `interrupted`
  - `server_content.generation_complete` -> `turn_complete`
  - `session_resumption_update` -> `session_update`; `go_away` -> `go_away`

Audio facts: input PCM16 16kHz mono; output PCM16 24kHz. Client/SDK user resamples mic to 16kHz.

### 4.3 `queue.py` — LiveInputQueue

Upstream decoupler. Thin wrapper over `asyncio.Queue` of a `LiveInput` union
(`audio`|`text`|`activity_start`|`activity_end`|`close`). Synchronous non-blocking `put` (`put_nowait`).
Fresh queue per session. Lets the input side keep accepting audio while the model is still generating
— the precondition for barge-in. This is the object an SDK user (or the API bridge) feeds.

### 4.4 `LiveAgent` — the realtime node and root controller

`LiveAgent` subclasses the **base agent** (`BaseAgent`), reuses the **config/builder** mixins
(`AgentSkillsMixin`, `AgentMemoryMixin`, `AgentProviderMixin`, the tool-declaration/function-schema
builder, `convert_messages`), and **excludes** `AgentExecutionMixin` (the text turn loop) — it writes
its own duplex loop. It is a valid graph node (registerable via `add_node`) and the constructor
surface mirrors `Agent`, so tools, `container` (InjectQ), state, skills, memory, callbacks all pass
through identically to `ReactAgent`.

Behavior when entered (by `arealtime()`): it is the **root controller** for the session.

1. Opens the provider WebSocket (the spine; held for the whole session).
2. Runs two concurrent tasks over the `LiveInputQueue`:
   - **pump task**: drains the queue -> provider (`send_audio`/`send_text`/activity).
   - **receive loop** (the generator body): iterates `client.receive()` and per event:
     - `audio_delta`/transcripts/`turn_complete`/`interrupted` -> yield to caller; `interrupted`
       also fires a publisher/callback event.
     - `tool_call` -> `ToolNode.invoke(name, args, config, state)` **internally** (existing parallel
       exec, InjectQ deps, MCP); then `send_tool_response`. Socket stays open. Callbacks + publisher
       fire from inside `ToolNode` (see §5).
     - **route to another agent/node** (when present; future) -> invoke it as a callable unit, await
       result, feed it back into the socket as content/tool-response. **Socket not torn down.**
     - transcript `finished=True` -> append a `Message` to state via reducers; persist (§7).
     - `session_update` -> cache resumption handle, persist to thread metadata.
     - `go_away`/drop -> transparent reconnect (§8).
3. Joined with `asyncio.gather(..., return_exceptions=True)`; `close()` mandatory in `finally` to
   avoid leaking provider sessions against quota.

`on_node_start/end` callbacks fire once for the whole `LiveAgent` run. `on_llm_*` has no discrete
call; map to per-turn (`turn_complete`) or skip.

### 4.5 `AudioAgent` (prebuilt) — single agent, React-style

`prebuilt/agent/audio.py`. Mirrors `ReactAgent`'s builder signature (model, tools, container, state,
skills, memory, publisher, checkpointer, callbacks, system_prompt, `RealtimeConfig`), wraps
`LiveAgent` as the graph root. **No sub-agents/handoff wired in v1** (gated off; a handoff tool is
just a tool, so the door stays open). This is what users instantiate.

## 5. `CompiledGraph.arealtime` — new runtime + transparency

New methods alongside `invoke`/`ainvoke`/`stream`/`astream`:

```python
async def arealtime(self, input_queue: LiveInputQueue,
                    config: dict) -> AsyncIterator[RealtimeEvent]: ...
def realtime(self, input_queue, config): ...   # sync wrapper
```

`arealtime` is a **separate runtime**, not the super-step loop. The live agent is the root; ordinary
nodes (preprocess, memory-preload, post-summarize) run as bounded phases or as callable units the live
session dispatches — never as traversed loop nodes during the live phase.

**Forcing rule:** in a realtime graph the root/entry must be the live agent. `invoke`/`stream` on a
graph containing a `LiveAgent` node -> raise ("use `.arealtime()`"). `arealtime` on a graph with no
live agent -> raise. One live node per realtime run in v1 (multiple = mic/VAD/voice ownership
conflict; sequential transfer only, later).

**Transparency is inherited free.** `compile()` already binds `publisher`, `callback_manager`,
`container`, `id_generator` into the InjectQ container (`state_graph.py:166-180`). `ToolNode.invoke`
(`tool_node/base.py:280`) pulls `callback_manager = Inject[CallbackManager]` and fires `publish_event`
+ `execute_before/after_invoke` + `execute_on_error` **inside itself**; `publish_event`
(`publish.py:31`) pulls `publisher = Inject[BasePublisher]` from the same container. So when `LiveAgent`
calls `ToolNode.invoke`, callbacks + publisher events fire **identically to text mode**, with zero
extra wiring — because the binding lives in the container, not the traversal. The LLM/turn-level
events that the text `Agent` node publishes are emitted by `LiveAgent` itself through the same
`publish_event` (the new realtime types — see §6). A publisher (OTEL/Kafka/Redis) on a thread sees one
continuous event stream whether the turn was text or audio.

Pure-SDK usage (no API):

```python
from agentflow.core.realtime import LiveInputQueue
audio = AudioAgent(model="gemini-3.1-flash-live-preview", tools=[...], container=c, ...)
graph = audio.compile(checkpointer=cp, publisher=pub)   # same build/compile as ReactAgent
q = LiveInputQueue()
# feed mic frames: q.send_audio(pcm_16k);  consume:
async for event in graph.arealtime(q, config={"thread_id": tid, "user_id": uid}):
    if event.type == "audio_delta": play(event.bytes)
    ...
```

## 6. Publisher / callback contract change

Realtime emits events the current `Event`/`EventType` enums
(`agentflow/runtime/publisher/events.py`) do not cover: `interrupted`, `input_transcript`,
`output_transcript`, `go_away`, `session_resumed`, `agent_changed`. Extend the enums (add a `REALTIME`
event category + the new types) so all publisher backends inherit realtime telemetry without
per-backend changes. Existing `on_tool_start/end` callbacks fire normally. Small but real contract
change — call it out in the PR.

## 7. State persistence — transcripts only, no audio at rest

Audio is **not** persisted (~1.9MB/min, no retrieval value). Both-side transcripts come from Gemini
`input_audio_transcription` (user) + `output_audio_transcription` (model). On a finished transcript
turn, `LiveAgent` appends a `Message` to `AgentState` via the existing reducers (`add_messages`):

- role `user`/`assistant`, content = `TextBlock(text=transcript)`, metadata `{modality:"audio"}`.
- No change to `AudioBlock` (whose `media: MediaRef` is required and we have no file).

Persisted via the existing checkpointer (`aput_messages`/`aput_state`/`aput_thread`) at **VAD-turn
granularity** (not per frame). The Gemini resumption handle is stored in thread metadata. Because turns
persist as normal `Message` objects on the thread, the **same `thread_id` is continuable by the text
`ReactAgent` too** — one durable conversation across modalities.

## 8. Resumption — two tiers, sized by context not clock

- **Within session (socket reconnect):** on `session_update`, cache Gemini's resumption handle and
  persist to thread metadata. On `go_away`/drop, reconnect with `SessionResumptionConfig(handle=...)`
  under the receive loop; the caller-facing generator sees no gap. Logic lives in `LiveAgent`, not the
  API. Enabled by default.
- **Cross session (durable thread resume):** on a new session with an existing `thread_id`, load
  thread history and reseed it into the new live session via Gemini `send_client_content` initial
  history. History is **compressed by the existing context manager** (`SummaryContextManager` /
  `BaseContextManager.atrim_context`) — reseed `last-N turns + running summary` sized to the context
  window, **not** a time window. Reuse `context_window_compression` (sliding window) for live growth.
  Audio is never replayed (none stored).

## 9. API layer — `/v1/graph/live` (optional, recommended)

New router `agentflow-api/.../routers/realtime/`, separate from `/v1/graph/ws`. Thin bridge over the
SDK; not required to use realtime.

1. Client opens WS. Auth via existing `RequirePermission("graph","stream")` + `?token=` fallback.
2. Read init JSON control frame: `{model, thread_id?, voice?, modalities?, vad?, tools_tags?, system_prompt?}`.
3. Build `AudioAgent` + `ToolNode`, load thread history, create a `LiveInputQueue`, call `arealtime`.
4. Two concurrent tasks:
   - **upstream**: client WS frames -> queue. Binary frame = PCM16 audio; JSON control =
     `{type:"activity_start"|"activity_end"|"text"|"close"}`.
   - **downstream**: `async for event in graph.arealtime(queue, cfg)` -> client. Audio as a **binary**
     frame; metadata/control (transcripts, turn_complete, interrupted, tool_call, errors) as a **JSON
     text** frame (ADK bandwidth split, ~75% less than base64-in-JSON).
5. `asyncio.gather(...)`; `finally: graph.aclose()` + `queue.close()`.

Wire protocol is provider-neutral (client never sees Gemini vs OpenAI). CLI: no new command; served by
existing `agentflow api`.

## 10. Packaging

- New optional extra `realtime` in core `pyproject.toml` (depends on `google-genai>=1.56.0`, already
  present as the `google-genai` extra). OpenAI Realtime reuses the `openai` extra later.
- All realtime imports guarded; core import never pulls realtime deps (per CLAUDE.md optional-deps rule).

## 11. Error handling

- Provider `error` -> normalized `RealtimeEvent(error)`; fatal closes session, transient continues.
- Client/queue disconnect -> cancel tasks, `close()`, persist final state.
- Tool failure -> error result via `send_tool_response` so the model recovers, plus observability event.
- Reconnect failure after N attempts -> emit `error`, close.

## 12. Testing strategy (TDD, no live LLM)

- `FakeRealtimeClient` yields scripted `RealtimeEvent` sequences to drive `LiveAgent`/`arealtime`.
- Tool loop: fake emits `tool_call` -> assert `ToolNode.invoke` called -> `send_tool_response` with result.
- Transparency: assert publisher events + callbacks fire on the tool loop identically to text mode.
- Barge-in: `interrupted` mid-audio -> event propagated, pump task alive.
- Within-session resume: `go_away` -> reconnect with stored handle, stream continuity.
- Cross-session resume: existing `thread_id` -> history loaded, compressed via context manager, reseeded.
- Transcript persistence: finished transcripts -> `Message`(`TextBlock`+`{modality:"audio"}`) appended
  via reducer + `aput_messages`; assert no audio bytes persisted.
- Forcing rule: `invoke`/`stream` on a live-rooted graph raises; `arealtime` on a non-live graph raises.
- API endpoint (mock agent): binary/JSON split, auth rejection, thread persistence.
- Coverage stays >= 70%.

## 13. Future (explicit, out of v1; architecture must not block)

- **Case 1 — text sub-agent as agent-as-tool.** A self-contained `CompiledGraph` invoked as a tool by
  the live session (framework has no subgraph nesting; one top graph, many nodes). Socket held open;
  use Gemini `NON_BLOCKING` scheduling so the model can say "one moment" while it runs. Static routing
  = phases; dynamic mid-voice routing must surface as a tool/handoff (the only provider control point).
- **Case 2 — realtime sub-agent as a persona swap on ONE shared socket** (ADK model): transfer swaps
  system instruction + tool set, `agent_changed` author tag updates, same audio stream. Per-agent voice
  via agent-level `speech_config`.
- **Rejected — two concurrent provider sockets** (Case 3): two audio streams/VADs/voices fighting one
  mic; provider one-voice/one-modality per session. Not viable; use sequential transfer instead.
- OpenAI Realtime provider behind the same `RealtimeClient`.
- TypeScript client SDK / `useRealtime` hook / browser audio I/O.
- Ephemeral-token browser-direct (only ever a no-tools mode; agentic requires proxy).

## 14. Build phases

- **Phase 1**: `base.py` contracts + `GeminiLiveClient` + normalizer + `LiveInputQueue`. Unit tests vs fake socket.
- **Phase 2**: `LiveAgent` (subclass base, duplex loop, tool loop, transcript persist, resume) +
  `RealtimeConfig` + `AudioAgent` prebuilt. Tool/barge-in/resume/transparency/persistence tests.
- **Phase 3**: `CompiledGraph.arealtime`/`realtime` + forcing-rule guards + publisher enum extension.
  Pure-SDK end-to-end test with fake provider.
- **Phase 4** (optional surface): `/v1/graph/live` API endpoint, auth reuse, binary/JSON frame split.
- **Phase 5** (future): OpenAI provider; multi-agent (Case 1 + 2); TS client; ephemeral tokens.
