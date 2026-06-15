# Realtime audio-to-audio (Gemini Live)

These examples use `AudioAgent`, the prebuilt realtime agent. Unlike `invoke`/`stream`
(turn-based super-step traversal), a realtime graph is driven by a separate runtime,
`CompiledGraph.arealtime(input_queue, config)`, because the provider owns the turn loop.

- Input audio: PCM16, mono, 16 kHz.
- Output audio: PCM16, mono, 24 kHz.
- Transcripts are persisted as `Message`s (`metadata={"modality": "audio"}`); raw audio is
  never stored.

## Install

```bash
pip install "10xscale-agentflow[realtime]"     # pulls in google-genai
export GEMINI_API_KEY=...
# Optional: pick a Gemini Live model (defaults to gemini-live-2.5-flash-preview).
# Valid Live model names come from the google-genai SDK, e.g.:
#   gemini-live-2.5-flash-preview
#   gemini-2.0-flash-live-preview-04-09
# Check Google's current docs for availability in your region.
export GEMINI_LIVE_MODEL=gemini-live-2.5-flash-preview
```

## 1. Headless: WAV file in, WAV file out

No microphone or speaker needed. Good for a first run / CI.

```bash
python examples/realtime/audio_agent_file.py path/to/input.wav   # 16 kHz mono PCM16
# writes out.wav and prints transcripts + tool calls
```

## 2. Live microphone (full duplex, React-style tool calling)

Speak and the agent talks back out loud, with barge-in. Ask about the weather and it
calls the `get_weather` tool, then speaks the result (reason -> tool -> respond).

```bash
pip install sounddevice
python examples/realtime/audio_agent_mic.py
# then say: "What's the weather in Tokyo?"   (Ctrl+C to stop)
```

## 3. Through the API server (`/v1/graph/live` WebSocket)

```bash
cd examples/realtime
agentflow api
# connect a WebSocket client to ws://localhost:8000/v1/graph/live
```

Protocol:

- First frame: a JSON control frame, e.g. `{"model": "...", "thread_id": "abc", "voice": "Puck"}`.
  Present fields override the agent's build-time config for that session.
- Upstream: binary frame = PCM16 input audio; JSON control frame =
  `{"type": "text" | "activity_start" | "activity_end" | "close", ...}`.
- Downstream: binary frame = PCM16 model audio; JSON text frame = every other event
  (transcripts, `turn_complete`, `interrupted`, `tool_call`, session/`go_away`, `error`).

## Key APIs

```python
from agentflow.core.realtime.base import RealtimeConfig
from agentflow.core.realtime.queue import LiveInputQueue
from agentflow.prebuilt.agent import AudioAgent

app = AudioAgent(
    "gemini-live-2.5-flash-preview",
    realtime_config=RealtimeConfig(model="gemini-live-2.5-flash-preview", voice="Puck"),
    tools=[my_tool],            # advertised to the model automatically
).compile()

queue = LiveInputQueue()
queue.send_audio(pcm16_bytes)   # non-blocking; safe to call from an audio callback
async for event in app.arealtime(queue, {"thread_id": "t1"}):
    ...                         # AudioDeltaEvent / transcripts / ToolCallEvent / ...
queue.close()                   # ends the session once the provider goes idle
```
