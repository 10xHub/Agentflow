# """Optional A2A protocol bridge for Agentflow.

# This package exposes any agentflow ``CompiledGraph`` as a standard A2A
# agent using the official ``a2a-sdk`` package, and also provides client
# helpers to call remote A2A agents from within a graph.

# Install the extra:

#     pip install 10xscale-agentflow[a2a_sdk]

# Quick start - server:

#     from agentflow.runtime.protocols.a2a import (
#         AgentFlowExecutor,
#         create_a2a_server,
#         make_agent_card,
#     )

# Quick start - client:

#     from agentflow.runtime.protocols.a2a import delegate_to_a2a_agent
# """

# from .client import create_a2a_client_node, delegate_to_a2a_agent
# from .executor import AgentFlowExecutor
# from .server import build_a2a_app, create_a2a_server, make_agent_card


# __all__ = [
#     "AgentFlowExecutor",
#     "build_a2a_app",
#     "create_a2a_client_node",
#     "create_a2a_server",
#     "delegate_to_a2a_agent",
#     "make_agent_card",
# ]
