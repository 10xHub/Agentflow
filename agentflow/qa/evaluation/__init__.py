"""
Agent Evaluation Module for Agentflow.

This module provides comprehensive evaluation capabilities for agent graphs,
including trajectory analysis, response quality assessment, and LLM-as-judge
evaluation patterns.

Main Components:
    - EvalSet, EvalCase:             Data models for test cases
    - EvalConfig, EvalPresets:       Configuration for evaluation criteria
    - AgentEvaluator:                Runs evaluation over EvalSet or single EvalCase
    - TrajectoryCollector:           Captures tool calls, node visits, and LLM outputs
    - ExecutionResult:               Holds tool calls, trajectory, final response
    - BaseCriterion + subclasses:    All evaluation criteria
    - EvalReport, EvalSummary:       Result and reporting models
    - Reporters:                     Console, JSON, HTML report generators

Example:
    ```python
    from agentflow.qa.evaluation import AgentEvaluator, EvalConfig, CriterionConfig
    from agentflow.qa.evaluation.dataset import EvalCase, ToolCall

    case = EvalCase.single_turn(
        eval_id="test_1",
        user_query="What is the weather in London?",
        expected_response="The weather in London is sunny",
        expected_tools=[ToolCall(name="get_weather")],
    )

    evaluator = AgentEvaluator(
        graph,
        collector=collector,
        config=EvalConfig(
            criteria={
                "tool_name_match_score": CriterionConfig(threshold=1.0),
            }
        ),
    )
    result = await evaluator.evaluate_case(case)
    assert result.passed
    ```
"""

# --- Dataset ---
# --- Collectors (event-based trajectory capture via callback_manager) ---
from agentflow.qa.evaluation.collectors.event_collector import EventCollector
from agentflow.qa.evaluation.collectors.trajectory_collector import (
    PublisherCallback,
    TrajectoryCollector,
    make_trajectory_callback,
)

# --- Config ---
from agentflow.qa.evaluation.config.eval_config import (
    CriteriaConfig,
    CriterionConfig,
    EvalConfig,
    MatchType,
    ReporterConfig,
    Rubric,
    UserSimulatorConfig,
)
from agentflow.qa.evaluation.config.presets import EvalPresets

# --- Criteria: base ---
from agentflow.qa.evaluation.criteria.base import (
    BaseCriterion,
    CompositeCriterion,
    SyncCriterion,
    WeightedCriterion,
)
from agentflow.qa.evaluation.criteria.factual_accuracy import FactualAccuracyCriterion

# --- Criteria: advanced ---
from agentflow.qa.evaluation.criteria.hallucination import HallucinationCriterion
from agentflow.qa.evaluation.criteria.llm_judge import LLMJudgeCriterion
from agentflow.qa.evaluation.criteria.llm_utils import LLMCallerMixin

# --- Criteria: response ---
from agentflow.qa.evaluation.criteria.response import (
    ContainsKeywordsCriterion,
    ExactMatchCriterion,
    ResponseMatchCriterion,
    RougeMatchCriterion,
)
from agentflow.qa.evaluation.criteria.rubric import RubricBasedCriterion
from agentflow.qa.evaluation.criteria.safety import SafetyCriterion

# --- Criteria: LLM-as-judge ---
from agentflow.qa.evaluation.criteria.simulation_goals import SimulationGoalsCriterion

# --- Criteria: trajectory ---
from agentflow.qa.evaluation.criteria.trajectory import (
    NodeOrderMatchCriterion,
    ToolNameMatchCriterion,
    TrajectoryMatchCriterion,
)
from agentflow.qa.evaluation.dataset.builder import EvalSetBuilder
from agentflow.qa.evaluation.dataset.eval_set import (
    EvalCase,
    EvalSet,
    Invocation,
    MessageContent,
    SessionInput,
    StepType,
    ToolCall,
    TrajectoryStep,
)

# --- Results ---
from agentflow.qa.evaluation.eval_result import (
    CriterionResult,
    EvalCaseResult,
    EvalReport,
    EvalSummary,
    NodeDetail,
)

# --- Evaluator ---
from agentflow.qa.evaluation.evaluator import AgentEvaluator, EvaluationRunner

# --- Execution ---
from agentflow.qa.evaluation.execution.result import ExecutionResult, NodeResponseData
from agentflow.qa.evaluation.quick_eval import QuickEval

# --- Reporters ---
from agentflow.qa.evaluation.reporters.base import BaseReporter
from agentflow.qa.evaluation.reporters.console import Colors, ConsoleReporter, print_report
from agentflow.qa.evaluation.reporters.html import HTMLReporter
from agentflow.qa.evaluation.reporters.json import JSONReporter, JUnitXMLReporter
from agentflow.qa.evaluation.reporters.manager import ReporterManager, ReporterOutput

# --- Simulators ---
from agentflow.qa.evaluation.simulators import (
    BatchSimulator,
    ConversationScenario,
    SimulationResult,
    UserSimulator,
)

# --- Testing helpers ---
from agentflow.qa.evaluation.testing import (
    EvalFixtures,
    EvalPlugin,
    EvalTestCase,
    assert_criterion_passed,
    assert_eval_passed,
    create_eval_app,
    create_simple_eval_set,
    eval_test,
    parametrize_eval_cases,
    run_eval,
)

# --- Token usage ---
from agentflow.qa.evaluation.token_usage import TokenUsage


__all__ = [
    # --- Evaluator ---
    "AgentEvaluator",
    # --- Criteria: base ---
    "BaseCriterion",
    # --- Reporters ---
    "BaseReporter",
    "BatchSimulator",
    "Colors",
    "CompositeCriterion",
    "ConsoleReporter",
    "ContainsKeywordsCriterion",
    "ConversationScenario",
    "CriteriaConfig",
    "CriterionConfig",
    # --- Results ---
    "CriterionResult",
    "EvalCase",
    "EvalCaseResult",
    # --- Config ---
    "EvalConfig",
    "EvalFixtures",
    "EvalPlugin",
    "EvalPresets",
    "EvalReport",
    # --- Dataset ---
    "EvalSet",
    "EvalSetBuilder",
    "EvalSummary",
    "EvalTestCase",
    "EvaluationRunner",
    "EventCollector",
    "ExactMatchCriterion",
    # --- Execution ---
    "ExecutionResult",
    "FactualAccuracyCriterion",
    "HTMLReporter",
    # --- Criteria: advanced ---
    "HallucinationCriterion",
    "Invocation",
    "JSONReporter",
    "JUnitXMLReporter",
    "LLMCallerMixin",
    "LLMJudgeCriterion",
    "MatchType",
    "MessageContent",
    "NodeDetail",
    "NodeOrderMatchCriterion",
    "NodeResponseData",
    "PublisherCallback",
    "QuickEval",
    "ReporterConfig",
    "ReporterManager",
    "ReporterOutput",
    # --- Criteria: response ---
    "ResponseMatchCriterion",
    "RougeMatchCriterion",
    "Rubric",
    "RubricBasedCriterion",
    "SafetyCriterion",
    "SessionInput",
    # --- Criteria: LLM-as-judge ---
    "SimulationGoalsCriterion",
    "SimulationResult",
    "StepType",
    "SyncCriterion",
    "TokenUsage",
    "ToolCall",
    "ToolNameMatchCriterion",
    # --- Collectors ---
    "TrajectoryCollector",
    # --- Criteria: trajectory ---
    "TrajectoryMatchCriterion",
    "TrajectoryStep",
    # --- Simulators ---
    "UserSimulator",
    "UserSimulatorConfig",
    "WeightedCriterion",
    "assert_criterion_passed",
    "assert_eval_passed",
    "create_eval_app",
    "create_simple_eval_set",
    "eval_test",
    "make_trajectory_callback",
    "parametrize_eval_cases",
    "print_report",
    # --- Testing ---
    "run_eval",
]
