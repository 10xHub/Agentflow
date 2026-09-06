"""Anthropic provider example using a ReAct agent with tools.

Prerequisites:
    1. Install the extra:  pip install "10xscale-agentflow[anthropic]"
    2. An Anthropic API key from https://console.anthropic.com.
    3. Environment variables (set in .env or shell):
        ANTHROPIC_API_KEY=<your-api-key>

    An unset ANTHROPIC_API_KEY does not necessarily mean there are no
    credentials: the SDK also resolves ANTHROPIC_AUTH_TOKEN, an `ant auth login`
    profile, and Workload Identity Federation.

Other backends
--------------
Anthropic reaches three backends, selected with ``anthropic_backend``:

    Agent(model="claude-opus-5")                                # direct API
    Agent(model="claude-opus-5", anthropic_backend="vertex")     # Vertex AI
    Agent(model="anthropic.claude-opus-5",                       # Bedrock
          anthropic_backend="bedrock")

Vertex takes the bare model id; Bedrock ids keep their ``anthropic.`` prefix.
Install the matching extra (``anthropic-vertex`` / ``anthropic-bedrock``).
"""

from dotenv import load_dotenv

from agentflow.core import Agent, StateGraph, ToolNode
from agentflow.core.state import AgentState, Message
from agentflow.storage.checkpointer import InMemoryCheckpointer
from agentflow.utils.constants import END


load_dotenv()

checkpointer = InMemoryCheckpointer()


def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    return f"The weather in {location} is sunny"


tool_node = ToolNode([get_weather])

agent = Agent(
    model="claude-opus-5",
    system_prompt=[{"role": "system", "content": "You are a helpful assistant."}],
    trim_context=True,
    # Maps to thinking={"type": "adaptive"} + output_config={"effort": "high"}.
    # budget_tokens is never sent: it returns a 400 on current Claude models.
    reasoning_config={"effort": "high"},
    tool_node=tool_node,
)


def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    if last_message.role == "tool":
        return "MAIN"

    return END


graph = StateGraph()
graph.add_node("MAIN", agent)
graph.add_node("TOOL", tool_node)
graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    inp = {"messages": [Message.text_message("What is the weather in New York City?")]}
    config = {"thread_id": "12345", "recursion_limit": 10}

    res = app.invoke(inp, config=config)

    for msg in res["messages"]:
        print(f"[{msg.role}] {msg}")
