"""Realtime AudioAgent exposed through the Agentflow API server.

``agentflow.json`` points the server at ``app`` below. Once running, the server serves a
WebSocket at ``/v1/graph/live`` that bridges browser/client audio to this agent (binary
PCM16 frames upstream, model audio back as binary, transcripts/tool-calls/events as JSON).

Run
    cd examples/realtime
    export GEMINI_API_KEY=...
    agentflow api
    # then connect a WebSocket client to ws://localhost:8000/v1/graph/live
"""

import os

from dotenv import find_dotenv, load_dotenv

from agentflow.core.realtime.base import RealtimeConfig
from agentflow.prebuilt.agent import AudioAgent
from agentflow.storage.checkpointer import InMemoryCheckpointer


# Load .env reliably regardless of the launch directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))
load_dotenv(find_dotenv(usecwd=True))

MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview")


def get_weather(location: str) -> str:
    """Return the current weather for a city. Called by the model during the conversation."""
    return f"It is 22 degrees Celsius and sunny in {location}."


checkpointer = InMemoryCheckpointer()

app = AudioAgent(
    MODEL,
    realtime_config=RealtimeConfig(
        model=MODEL,
        voice="Puck",
        system_instruction="You are a concise, friendly voice assistant.",
    ),
    tools=[get_weather],
).compile(checkpointer=checkpointer)
