"""SqliteCheckpointer example — durable agent state in a single local file.

`SqliteCheckpointer` keeps everything (state, hot state cache, messages,
threads, generic cache) in one SQLite `.db` file — no Postgres or Redis. It is
ideal for client-side / single-user agents: a desktop app shipping a Python
sidecar (Tauri, Electron, PyInstaller), a local CLI agent, or any deployment
where each user has their own process and their own database file.

Run it twice with the same `thread_id`; the conversation is restored from disk
on the second run because the `.db` file persists between processes.

Requires: pip install 10xscale-agentflow[sqlite_checkpoint]
"""

from dotenv import load_dotenv

from agentflow.core import Agent, StateGraph, ToolNode
from agentflow.core.state import AgentState, Message
from agentflow.storage.checkpointer import SqliteCheckpointer
from agentflow.utils.constants import END


load_dotenv()

# One local file holds all state. Defaults to ~/.agentflow/checkpointer.db when
# no path is given; here we keep it beside the script.
checkpointer = SqliteCheckpointer("agent_state.db")


class CustomAgentState(AgentState):
    jd_name: str = "CustomAgentState"


def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    return f"The weather in {location} is sunny"


tool_node = ToolNode([get_weather])

agent = Agent(
    model="gemini-2.5-flash",
    provider="google",
    system_prompt="You are a helpful assistant. Use tools when appropriate.",
    tool_node=tool_node,
)


def should_use_tools(state: AgentState) -> str:
    if not state.context:
        return "TOOL"
    last_message = state.context[-1]
    if (
        getattr(last_message, "tools_calls", None)
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


def main() -> None:
    config = {"thread_id": "user-42", "recursion_limit": 10}

    # Prior turns for this thread are loaded from the SQLite file automatically.
    existing = checkpointer.list_messages(config)
    print(f"Restored {len(existing)} message(s) from disk for this thread.\n")

    inp = {"messages": [Message.text_message("What's the weather in New York City?")]}
    res = app.invoke(inp, config=config)

    for msg in res["messages"]:
        print(f"[{msg.role}] {msg}")

    print("\nState is now persisted in 'agent_state.db'. Run again to resume.")

    # Release the SQLite connection when done (safe to call once at shutdown).
    checkpointer.release()


if __name__ == "__main__":
    main()
