"""
Shared weather-agent graph builder for evaluation tests (test1–test5).

Uses the ``Agent`` class with the Google GenAI provider — no LiteLLM
dependency.  All graph wiring and compilation lives here so every test
folder only needs to call ``create_app_and_collector()``.

Example::

    from examples.evaluation.test_graph import create_app_and_collector

    app, collector = create_app_and_collector()
"""

from __future__ import annotations

from dotenv import load_dotenv


load_dotenv()


def build_weather_graph():
    """Build a weather-agent graph using the Agent class.

    Returns:
        StateGraph: The uncompiled graph.
    """
    from agentflow.core.graph import Agent, StateGraph, ToolNode
    from agentflow.core.state import AgentState
    from agentflow.utils.constants import END

    # ── Tools ────────────────────────────────────────────────────────
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny"

    def get_forecast(location: str, days: int = 3) -> str:
        """Get a multi-day weather forecast for a location."""
        return f"{days}-day forecast for {location}: sunny, cloudy, sunny"

    tool_node = ToolNode([get_weather, get_forecast])

    # ── Agent ────────────────────────────────────────────────────────
    agent = Agent(
        model="gemini-2.5-flash",
        provider="google",
        system_prompt=[
            {
                "role": "system",
                "content": (
                    "You are a helpful weather assistant. "
                    "Use get_weather for current conditions and "
                    "get_forecast for multi-day forecasts. "
                    "For all other questions (general knowledge, geography, etc.) "
                    "answer them directly from your own knowledge without using any tools."
                ),
            },
        ],
        tool_node="TOOL",
    )

    # ── Routing ──────────────────────────────────────────────────────
    def should_use_tools(state: AgentState) -> str:
        if not state.context or len(state.context) == 0:
            return "TOOL"
        last = state.context[-1]
        if (
            hasattr(last, "tools_calls")
            and last.tools_calls
            and len(last.tools_calls) > 0
            and last.role == "assistant"
        ):
            return "TOOL"
        if last.role == "tool":
            return "MAIN"
        return END

    # ── Graph wiring ─────────────────────────────────────────────────
    graph = StateGraph()
    graph.add_node("MAIN", agent)
    graph.add_node("TOOL", tool_node)
    graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")

    return graph


def create_app_and_collector():
    """Compile the weather graph with a fresh TrajectoryCollector.

    Returns:
        tuple[CompiledGraph, TrajectoryCollector]: Ready-to-invoke app and
        the wired collector.
    """
    from agentflow.qa.evaluation.testing import create_eval_app

    return create_eval_app(build_weather_graph())
