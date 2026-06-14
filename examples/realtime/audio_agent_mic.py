"""Live microphone audio-to-audio with AudioAgent + Gemini Live.

Speak into your microphone and the agent talks back in real time. It supports barge-in
(start talking while it is speaking and it stops to listen) and tool calls. This is the
full duplex demo; for a headless/no-hardware version see ``audio_agent_file.py``.

Setup
    pip install "10xscale-agentflow[realtime]" sounddevice
    export GEMINI_API_KEY=...
    export GEMINI_LIVE_MODEL=gemini-live-2.5-flash-preview   # optional, see README

Run
    python examples/realtime/audio_agent_mic.py
    # speak; press Ctrl+C to stop.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from agentflow.core.realtime.base import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, RealtimeConfig
from agentflow.core.realtime.queue import LiveInputQueue
from agentflow.prebuilt.agent import AudioAgent


load_dotenv()

MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview")
MIC_BLOCK = INPUT_SAMPLE_RATE // 10  # 100 ms frames


def get_weather(location: str) -> str:
    """Return the current weather for a city. Called by the model during the conversation."""
    return f"It is 22 degrees Celsius and sunny in {location}."


def build_app():
    config = RealtimeConfig(
        model=MODEL,
        voice="Puck",
        system_instruction="You are a friendly, concise voice assistant.",
    )
    return AudioAgent(MODEL, realtime_config=config, tools=[get_weather]).compile()


async def main() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("This example needs sounddevice:  pip install sounddevice")

    app = build_app()
    queue = LiveInputQueue()
    loop = asyncio.get_running_loop()

    def on_mic(indata, _frames, _time, _status) -> None:
        # PortAudio calls this on its own thread; marshal onto the event loop so the
        # asyncio-backed queue is touched only from the loop thread.
        loop.call_soon_threadsafe(queue.send_audio, bytes(indata))

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
    print("Listening. Speak into your mic; press Ctrl+C to stop.")

    try:
        async for event in app.arealtime(queue, {"thread_id": "audio-mic-demo"}):
            if event.type == "audio_delta":
                speaker.write(event.data)
            elif event.type == "interrupted":
                # Barge-in: discard audio already queued for playback.
                speaker.stop()
                speaker.start()
            elif event.type == "input_transcript" and event.finished:
                print(f"you:   {event.text}")
            elif event.type == "output_transcript" and event.finished:
                print(f"agent: {event.text}")
            elif event.type == "tool_call":
                print(f"[tool] {event.name}({event.args})")
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
