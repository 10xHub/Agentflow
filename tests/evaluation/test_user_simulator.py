"""Tests for UserSimulator and BatchSimulator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any
import json
import uuid

from agentflow.qa.evaluation.simulators.user_simulator import (
    UserSimulator,
    BatchSimulator,
    ConversationScenario,
    SimulationResult
)
from agentflow.qa.evaluation.config.eval_config import UserSimulatorConfig
from agentflow.qa.evaluation.criteria.base import BaseCriterion, CriterionResult
from agentflow.core.graph.compiled_graph import CompiledGraph
from agentflow.core.state import Message

class MockCriterion(BaseCriterion):
    """Mock criterion for evaluation."""
    def __init__(self, name="mock_criterion"):
        super().__init__()
        self.name = name

    async def evaluate(self, execution_result: Any, eval_case: Any) -> CriterionResult:
        return CriterionResult(
            criterion=self.name,
            score=0.9,
            passed=True,
            details={"notes": "good response"}
        )

@pytest.mark.asyncio
async def test_user_simulator_init():
    # 1. Init with defaults
    sim = UserSimulator()
    assert sim.model == "gemini/gemini-2.5-flash"
    assert sim.temperature == 0.7
    assert sim.max_turns == 10
    assert sim.api_style == "responses"

    # 2. Init with config
    config = UserSimulatorConfig(
        model="gpt-4o",
        temperature=0.5,
        max_invocations=5
    )
    # Mocking config object to have api_style
    config.api_style = "chat"
    sim_config = UserSimulator(config=config)
    assert sim_config.model == "gpt-4o"
    assert sim_config.temperature == 0.5
    assert sim_config.max_turns == 5
    assert sim_config.api_style == "chat"

@pytest.mark.asyncio
async def test_user_simulator_run_success():
    # Prepare scenario
    scenario = ConversationScenario(
        scenario_id="test_scen",
        description="A test scenario",
        starting_prompt="Hello agent",
        goals=["Say hello back"],
        max_turns=3
    )

    # Mock CompiledGraph
    mock_graph = MagicMock(spec=CompiledGraph)
    msg = Message.text_message("Hello back!", role="assistant")
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

    # Mock call_llm response for goal check
    mock_result_json = json.dumps({"achieved": True, "reasoning": "Agent said hello back"})
    
    criterion = MockCriterion("test_criterion")
    sim = UserSimulator(criteria=[criterion])

    with patch("agentflow.qa.evaluation.simulators.user_simulator.call_llm") as mock_call:
        mock_call.return_value = (mock_result_json, 10, 20, 0)
        
        result = await sim.run(mock_graph, scenario)
        
        assert result.completed is True
        assert "Say hello back" in result.goals_achieved
        assert result.turns == 1
        assert result.criterion_scores["test_criterion"] == 0.9

@pytest.mark.asyncio
async def test_user_simulator_run_no_starting_prompt():
    scenario = ConversationScenario(
        scenario_id="test_scen_no_start",
        description="Test no start",
        starting_prompt="",
        goals=["Finish"],
        max_turns=2
    )

    mock_graph = MagicMock(spec=CompiledGraph)
    msg = Message.text_message("Got it", role="assistant")
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

    sim = UserSimulator()

    with patch("agentflow.qa.evaluation.simulators.user_simulator.call_llm") as mock_call:
        mock_call.side_effect = [
            ("Start prompt", 5, 5, 0),
            (json.dumps({"achieved": True, "reasoning": "Done"}), 10, 10, 0)
        ]
        
        result = await sim.run(mock_graph, scenario)
        assert result.completed is True
        assert result.conversation[0]["content"] == "Start prompt"

@pytest.mark.asyncio
async def test_user_simulator_run_max_turns():
    scenario = ConversationScenario(
        scenario_id="test_max_turns",
        description="Test max turns limit",
        starting_prompt="Hello",
        goals=["Goal never met"],
        max_turns=2
    )

    mock_graph = MagicMock(spec=CompiledGraph)
    msg = Message.text_message("Agent reply", role="assistant")
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

    sim = UserSimulator()

    with patch("agentflow.qa.evaluation.simulators.user_simulator.call_llm") as mock_call:
        mock_call.side_effect = [
            (json.dumps({"achieved": False, "reasoning": "Not yet"}), 5, 5, 0),
            ("User follow-up", 5, 5, 0),
            (json.dumps({"achieved": False, "reasoning": "Still not yet"}), 5, 5, 0),
            ("User second follow-up", 5, 5, 0),
        ]
        
        result = await sim.run(mock_graph, scenario)
        assert result.completed is False
        assert result.turns == 2
        assert len(result.conversation) == 4

@pytest.mark.asyncio
async def test_user_simulator_graph_exception():
    scenario = ConversationScenario(
        scenario_id="test_graph_err",
        starting_prompt="Hello",
        goals=["Done"]
    )
    mock_graph = MagicMock(spec=CompiledGraph)
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Graph crash"))

    sim = UserSimulator()
    result = await sim.run(mock_graph, scenario)
    assert result.completed is False
    assert "Graph crash" in result.error

@pytest.mark.asyncio
async def test_user_simulator_check_goals_fallback():
    sim = UserSimulator()
    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "the magic word is banana"}
    ]
    
    with patch("agentflow.qa.evaluation.simulators.user_simulator.call_llm", side_effect=Exception("LLM down")):
        achieved, usage = await sim._check_goals(
            all_goals=["banana", "apple"],
            achieved=[],
            conversation=conversation
        )
        assert "banana" in achieved
        assert "apple" not in achieved

@pytest.mark.asyncio
async def test_user_simulator_general_exception():
    scenario = ConversationScenario(scenario_id="crash")
    sim = UserSimulator()
    mock_graph = MagicMock(spec=CompiledGraph)
    mock_graph.ainvoke = AsyncMock(return_value={"messages": []})

    # Cause crash inside try block by raising TypeError in _generate_initial_message
    with patch.object(sim, "_generate_initial_message", side_effect=TypeError("Crashed inside")):
        result = await sim.run(mock_graph, scenario)
        assert result.completed is False
        assert "Crashed inside" in result.error

def test_extract_response_fallback_paths():
    sim = UserSimulator()
    # 1. empty result
    assert sim._extract_response({}) == ""
    # 2. messages is empty
    assert sim._extract_response({"messages": []}) == ""
    # 3. assistant content block list
    msg1 = type("Message", (), {
        "role": "assistant",
        "content": [
            type("Block", (), {"text": "Hello"}),
            type("Block", (), {})
        ]
    })
    assert sim._extract_response({"messages": [msg1]}) == "Hello"

    # 4. plain dict format
    msg2 = {"role": "assistant", "content": "Dict text"}
    assert sim._extract_response({"messages": [msg2]}) == "Dict text"

@pytest.mark.asyncio
async def test_batch_simulator():
    sim = UserSimulator()
    batch = BatchSimulator(simulator=sim, max_concurrency=2)
    
    mock_graph = MagicMock(spec=CompiledGraph)
    msg = Message.text_message("Done", role="assistant")
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

    scenarios = [
        ConversationScenario(scenario_id="s1", starting_prompt="p1", goals=["g1"]),
        ConversationScenario(scenario_id="s2", starting_prompt="p2", goals=["g2"])
    ]

    with patch("agentflow.qa.evaluation.simulators.user_simulator.call_llm") as mock_call:
        mock_call.return_value = (json.dumps({"achieved": True, "reasoning": "ok"}), 1, 1, 0)
        
        results = await batch.run_batch(mock_graph, scenarios)
        assert len(results) == 2
        assert results[0].scenario_id == "s1"
        assert results[1].scenario_id == "s2"
        
        summary = batch.summary(results)
        assert summary["total_scenarios"] == 2
        assert summary["completed"] == 2
        assert summary["completion_rate"] == 1.0
        assert summary["errors"] == 0
