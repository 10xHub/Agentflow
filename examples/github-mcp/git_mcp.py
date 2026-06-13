import json
import os
from datetime import datetime

from dotenv import load_dotenv
from fastmcp import Client

from agentflow.core import Agent, StateGraph, ToolNode
from agentflow.core.state import AgentState, Message
from agentflow.storage.checkpointer import InMemoryCheckpointer
from agentflow.utils.constants import END


load_dotenv()

checkpointer = InMemoryCheckpointer()

config = {
    "mcpServers": {
        "github": {
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
            "transport": "streamable-http",
        },
    }
}


client_http = Client(config)

tool_node = ToolNode([], client=client_http)


main_agent = Agent(
    model="gemini-2.0-flash",
    provider="google",
    system_prompt=[
        {
            "role": "system",
            "content": """
                You are a helpful assistant.
                Your task is to assist the user in finding information and answering questions.
            """,
        },
    ],
    tools=tool_node,
    trim_context=True,
)


def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"  # No context, might need tools

    last_message = state.context[-1]

    # If the last message is from assistant and has tool calls, go to TOOL
    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    # If last message is a tool result, we should be done (AI will make final response)
    if last_message.role == "tool":
        return END

    # Default to END for other cases
    return END


graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)

# Add conditional edges from MAIN
graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", END: END},
)

# Always go back to MAIN after TOOL execution
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")


app = graph.compile(
    checkpointer=checkpointer,
)


# now run it

inp = {
    "messages": [
        Message.text_message(
            "Please call the list_commits function for the github repo 'https://github.com/suchith83/portfolio' of the 'suchith83' username, and give me the all commits in that repo."
        )
    ]
}
config = {"thread_id": "12345", "recursion_limit": 10}
res = app.invoke(inp, config=config)


def pretty_print_messages(messages):
    for i, m in enumerate(messages, 1):
        print("=" * 60)
        print(f"Message {i}:")
        print(f"  ID: {getattr(m, 'message_id', None)}")
        print(f"  Role: {m.role}")

        if hasattr(m, "timestamp") and m.timestamp:
            ts = m.timestamp
            if isinstance(ts, datetime):
                ts = ts.isoformat()
            print(f"  Timestamp: {ts}")

        # content
        if m.content:
            print("  Content:")
            print("    " + str(m.content).replace("\n", "\n    "))

        # tool calls
        if getattr(m, "tools_calls", None):
            print("  Tool Calls:")
            print(json.dumps(m.tools_calls, indent=4))

        # tool call id
        if getattr(m, "tool_call_id", None):
            print(f"  Tool Call ID: {m.tool_call_id}")

        # metadata
        if getattr(m, "metadata", None):
            print("  Metadata:")
            print(json.dumps(m.metadata, indent=4))


print("printing the response")
pretty_print_messages(res["messages"])
