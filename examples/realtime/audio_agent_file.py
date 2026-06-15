"""Realtime audio-to-audio from a WAV file, using AudioAgent + Gemini Live.

No microphone or speaker required. This streams a 16 kHz mono PCM16 WAV into the live
session, writes the model's 24 kHz audio reply to ``out.wav``, and prints the input and
output transcripts plus any tool calls. It is the headless counterpart to
``audio_agent_mic.py`` and is the easiest way to sanity-check your setup.

Setup
    pip install "10xscale-agentflow[realtime]"
    export GEMINI_API_KEY=...
    # optionally override the model (see README for valid Gemini Live models):
    export GEMINI_LIVE_MODEL=gemini-live-2.5-flash-preview

Run
    python examples/realtime/audio_agent_file.py path/to/input.wav

``input.wav`` must be mono, 16-bit PCM, 16 kHz (the format Gemini Live expects for input).
"""

import asyncio
import os
import sys
import wave

from dotenv import find_dotenv, load_dotenv

from agentflow.core.realtime.base import OUTPUT_SAMPLE_RATE, RealtimeConfig
from agentflow.core.realtime.queue import LiveInputQueue
from agentflow.prebuilt.agent import AudioAgent


# Load .env reliably regardless of the launch directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))
load_dotenv(find_dotenv(usecwd=True))

MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview")


def get_weather(location: str) -> str:
    """Return the current weather for a city. Called by the model during the conversation."""
    return f"It is 22 degrees Celsius and sunny in {location}."


def build_app():
    """Compile a single realtime audio agent with one tool and a voice."""
    config = RealtimeConfig(
        model=MODEL,
        voice="Puck",
        system_instruction="You are a concise voice assistant. Keep answers to one or two sentences.",
    )
    return AudioAgent(MODEL, realtime_config=config, tools=[get_weather]).compile()


def read_pcm16(path: str) -> tuple[int, bytes]:
    """Read a mono PCM16 WAV file, returning (sample_rate, raw_pcm_bytes)."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise ValueError("input must be mono, 16-bit PCM (sampwidth=2, channels=1)")
        return wf.getframerate(), wf.readframes(wf.getnframes())


async def feed_audio(queue: LiveInputQueue, pcm: bytes, sample_rate: int) -> None:
    """Stream the file into the session in ~100 ms chunks (the audio hot path)."""
    chunk = (sample_rate // 10) * 2  # 100 ms * 2 bytes/sample
    for offset in range(0, len(pcm), chunk):
        queue.send_audio(pcm[offset : offset + chunk], sample_rate=sample_rate)
        await asyncio.sleep(0.0)  # yield so the pump task can flush to the socket
    # Automatic VAD detects end-of-speech; we leave the queue open to receive the reply
    # and close it from the main loop once the model finishes its turn.


async def main() -> None:
    in_path = sys.argv[1] if len(sys.argv) > 1 else "input.wav"
    if not os.path.exists(in_path):
        sys.exit(f"Provide a 16 kHz mono PCM16 WAV path. Not found: {in_path}")

    sample_rate, pcm = read_pcm16(in_path)
    app = build_app()
    queue = LiveInputQueue()

    out = wave.open("out.wav", "wb")
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(OUTPUT_SAMPLE_RATE)

    feeder = asyncio.create_task(feed_audio(queue, pcm, sample_rate))
    try:
        async for event in app.arealtime(queue, {"thread_id": "audio-file-demo"}):
            if event.type == "audio_delta":
                out.writeframes(event.data)
            elif event.type == "input_transcript" and event.finished:
                print(f"you:   {event.text}")
            elif event.type == "output_transcript" and event.finished:
                print(f"agent: {event.text}")
            elif event.type == "tool_call":
                print(f"[tool] {event.name}({event.args})")
            elif event.type == "turn_complete":
                queue.close()  # one turn for this demo: end the session
    finally:
        feeder.cancel()
        out.close()
        await app.aclose()

    print("Wrote model audio reply to out.wav")


if __name__ == "__main__":
    asyncio.run(main())
