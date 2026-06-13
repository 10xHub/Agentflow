"""Agentflow Skills — dynamic skill injection for agents.

Two activation modes are supported:

**on-demand** (default) — the LLM sees a trigger table and calls ``set_skill()``
to load skill content when a user request matches a skill::

    from agentflow.core.skills import SkillConfig

    agent = Agent(
        model="gpt-4o",
        system_prompt=[{"role": "system", "content": "You are helpful."}],
        skills=SkillConfig(skills_dir="./skills/"),
    )

**session** — designed for multi-tenant agents where each session has a fixed
domain/persona.  The framework reads a state field to identify which skill to
preload, with no trigger table and no extra tool-call round-trip::

    from agentflow.core.skills import SkillConfig
    from agentflow.core.state import AgentState


    class TenantState(AgentState):
        SKILL_NAME: str = ""


    agent = Agent(
        model="gpt-4o",
        skills=SkillConfig(
            skills_dir="./skills/",
            mode="session",
            preload_from="SKILL_NAME",  # reads state.SKILL_NAME each call
        ),
    )
"""

from .models import SkillConfig, SkillMeta
from .registry import SkillsRegistry


__all__ = [
    "SkillConfig",
    "SkillMeta",
    "SkillsRegistry",
]
