"""Live voice weather assistant -- AudioAgent + Gemini Live, React-style tool calling.

Talk to it through your microphone and it talks back in real time. Ask "what's the
weather in Tokyo?" and the model calls the ``get_weather`` tool, then speaks the answer
it gets back -- the realtime analog of a ReactAgent's reason -> tool -> respond loop.

Features
    - Voice playback: the model's reply is played out loud on your speakers.
    - Tool calling: ``get_weather`` is advertised to the model and invoked on demand.
    - Echo-safe by default: the mic is muted while the agent is speaking, so it doesn't
      hear (and reply to) its own voice through your speakers.

Echo / feedback
    Without echo cancellation, your speaker audio leaks into the mic and the model
    transcribes its own replies as your input. This demo avoids that by muting the mic
    while the agent talks (half-duplex). Use headphones and set MIC_FULL_DUPLEX=1 for
    true full duplex with barge-in (speak over the agent to interrupt it).

Setup
    pip install "10xscale-agentflow[realtime]" sounddevice
    export GEMINI_API_KEY=...                                # or Vertex AI env (see README)
    export GEMINI_LIVE_MODEL=...                             # optional, see README

Run
    python examples/realtime/audio_agent_mic.py
    # then say e.g. "What's the weather in Paris?" -- press Ctrl+C to stop.
"""

import asyncio
import contextlib
import os
import sys

from dotenv import find_dotenv, load_dotenv

from agentflow.core.realtime.base import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, RealtimeConfig
from agentflow.core.realtime.queue import LiveInputQueue
from agentflow.prebuilt.agent import AudioAgent


# Load .env reliably no matter where you launch from: the one next to this script first,
# then the nearest .env walking up from the current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))
load_dotenv(find_dotenv(usecwd=True))

# Use Vertex AI (service account / ADC) when GOOGLE_GENAI_USE_VERTEXAI is set; otherwise
# fall back to a Gemini API key. Both are supported by the live client.
USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in ("1", "true", "yes")
# Live model names differ between the Gemini Developer API and Vertex AI. Override either
# with GEMINI_LIVE_MODEL; check Google's docs for what's enabled in your project/region.
_DEFAULT_MODEL = (
    "gemini-live-2.5-flash-preview-native-audio-09-2025"
    if USE_VERTEX
    else "gemini-live-2.5-flash-preview"
)
MODEL = os.getenv("GEMINI_LIVE_MODEL", _DEFAULT_MODEL)
MIC_BLOCK = INPUT_SAMPLE_RATE // 10  # 100 ms frames
# Full duplex (no mic muting) -- only sensible with headphones, enables barge-in.
FULL_DUPLEX = os.getenv("MIC_FULL_DUPLEX", "").strip().lower() in ("1", "true", "yes")
# How long to keep the mic muted after the agent's turn ends, to let the speaker drain.
MUTE_TAIL_SEC = 0.4

# A tiny canned forecast table so different cities give different answers. Swap the body
# for a real HTTP call to a weather API and nothing else here changes.
_FORECASTS = {
    "tokyo": "18 degrees Celsius, light rain",
    "paris": "24 degrees Celsius and sunny",
    "london": "15 degrees Celsius, overcast",
    "new york": "21 degrees Celsius, partly cloudy",
    "san francisco": "17 degrees Celsius, foggy",
}


def get_weather(location: str) -> str:
    """Get the current weather for a city. Call this whenever the user asks about weather.

    Args:
        location: The city name, e.g. "Tokyo" or "Paris".
    """
    forecast = _FORECASTS.get(location.strip().lower(), "22 degrees Celsius and clear")
    print(f"  [tool] get_weather(location={location!r}) -> {forecast}")
    return f"The weather in {location} is {forecast}."


def build_app():
    config = RealtimeConfig(
        model=MODEL,
        voice="Puck",
        system_instruction=(
            "You are a friendly, concise voice assistant. When the user asks about the "
            "weather, always call the get_weather tool and answer using its result. "
            "Keep replies to one or two sentences."
        ),
    )
    return AudioAgent(
        MODEL, realtime_config=config, tools=[get_weather], use_vertex_ai=USE_VERTEX
    ).compile()


async def main() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("This example needs sounddevice:  pip install sounddevice")

    app = build_app()
    queue = LiveInputQueue()
    loop = asyncio.get_running_loop()

    # Mic gate: while the agent is speaking we drop mic frames so its own voice (played on
    # the speaker and picked up by the mic) is never sent back as user input. Disabled in
    # full-duplex mode (headphones), where barge-in is wanted instead.
    agent_speaking = {"on": False}

    def on_mic(indata, _frames, _time, _status) -> None:
        # PortAudio calls this on its own thread; marshal onto the event loop so the
        # asyncio-backed queue is touched only from the loop thread.
        if agent_speaking["on"] and not FULL_DUPLEX:
            return  # muted while the agent talks (echo guard)
        loop.call_soon_threadsafe(queue.send_audio, bytes(indata))

    def unmute() -> None:
        agent_speaking["on"] = False

    speaker = sd.RawOutputStream(samplerate=OUTPUT_SAMPLE_RATE, channels=1, dtype="int16")
    mic = sd.RawInputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=MIC_BLOCK,
        callback=on_mic,
    )
    speaker.start()
    mic.start()
    mode = "full-duplex (barge-in)" if FULL_DUPLEX else "echo-safe (mic muted while agent talks)"
    print(f"Listening [{mode}]. Try: 'What's the weather in Tokyo?'   (Ctrl+C to stop)")

    try:
        async for event in app.arealtime(queue, {"thread_id": "audio-mic-demo"}):
            if event.type == "audio_delta":
                agent_speaking["on"] = True  # mute the mic for the duration of the reply
                speaker.write(event.data)  # play the model's voice
            elif event.type == "turn_complete":
                # Reopen the mic after the speaker has drained the buffered tail.
                loop.call_later(MUTE_TAIL_SEC, unmute)
            elif event.type == "interrupted":
                # Barge-in (full-duplex only): discard audio already queued for playback.
                speaker.stop()
                speaker.start()
                agent_speaking["on"] = False
            elif event.type == "input_transcript" and event.finished:
                print(f"you:   {event.text}")
            elif event.type == "output_transcript" and event.finished:
                print(f"agent: {event.text}")
            elif event.type == "tool_call":
                print(f"  [tool-call requested] {event.name}({event.args})")
            elif event.type == "error":
                print(f"  [error] {event.message}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopping...")
    finally:
        queue.close()
        mic.stop()
        mic.close()
        speaker.stop()
        speaker.close()
        await app.aclose()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
