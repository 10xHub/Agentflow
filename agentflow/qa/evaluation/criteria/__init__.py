"""
Evaluation criteria package.

CRITERIA_REGISTRY is the single authoritative mapping from criterion names
(both friendly and legacy) to their criterion classes.  AgentEvaluator uses
it to instantiate criteria from EvalConfig; no other mapping is needed.

Example:
    ```python
    from agentflow.qa.evaluation.criteria import (
        TrajectoryMatchCriterion,
        ResponseMatchCriterion,
        LLMJudgeCriterion,
        RubricBasedCriterion,
        HallucinationCriterion,
        SafetyCriterion,
        FactualAccuracyCriterion,
        CRITERIA_REGISTRY,
    )
    ```
"""

from .base import (
    BaseCriterion,
    CompositeCriterion,
    SyncCriterion,
    WeightedCriterion,
)
from .factual_accuracy import FactualAccuracyCriterion
from .hallucination import HallucinationCriterion
from .llm_base import TemplatedLLMCriterion
from .llm_judge import LLMJudgeCriterion
from .llm_utils import LLMCallerMixin
from .response import (
    ContainsKeywordsCriterion,
    ExactMatchCriterion,
    ResponseMatchCriterion,
    RougeMatchCriterion,
)
from .rubric import RubricBasedCriterion
from .safety import SafetyCriterion
from .simulation_goals import SimulationGoalsCriterion
from .trajectory import NodeOrderMatchCriterion, ToolNameMatchCriterion, TrajectoryMatchCriterion


# Single registry: all names (canonical + aliases) → criterion class.
# AgentEvaluator._create_criterion() and CriteriaConfig both use this;
# there is no other place where this mapping lives.
CRITERIA_REGISTRY: dict[str, type[BaseCriterion]] = {
    # Trajectory / tool matching
    "tool_trajectory_avg_score": TrajectoryMatchCriterion,  # legacy serialized name
    "trajectory_match": TrajectoryMatchCriterion,
    "trajectory": TrajectoryMatchCriterion,
    "tool_name_match_score": ToolNameMatchCriterion,  # legacy serialized name
    "tool_name_match": ToolNameMatchCriterion,
    # Node order
    "node_order_score": NodeOrderMatchCriterion,  # legacy serialized name
    "node_order": NodeOrderMatchCriterion,
    # Response matching
    "response_match_score": ResponseMatchCriterion,  # legacy serialized name
    "response_match": ResponseMatchCriterion,
    "rouge_match": RougeMatchCriterion,
    "exact_match": ExactMatchCriterion,
    "contains_keywords": ContainsKeywordsCriterion,
    # LLM-as-judge
    "final_response_match_v2": LLMJudgeCriterion,  # legacy serialized name
    "llm_judge": LLMJudgeCriterion,
    # Specialised LLM criteria
    "rubric_based_final_response_quality_v1": RubricBasedCriterion,  # legacy serialized name
    "rubric_based_score": RubricBasedCriterion,  # legacy serialized name
    "rubric_based": RubricBasedCriterion,
    "hallucinations_v1": HallucinationCriterion,  # legacy serialized name
    "hallucination": HallucinationCriterion,
    "safety_v1": SafetyCriterion,  # legacy serialized name
    "safety_score": SafetyCriterion,  # legacy serialized name
    "safety": SafetyCriterion,
    "factual_accuracy_v1": FactualAccuracyCriterion,  # legacy serialized name
    "factual_accuracy_score": FactualAccuracyCriterion,  # legacy serialized name
    "factual_accuracy": FactualAccuracyCriterion,
    # Simulation / multi-turn (UserSimulator only)
    "simulation_goals": SimulationGoalsCriterion,
    "simulation_goals_match": SimulationGoalsCriterion,  # legacy serialized name
    "conversation_goals": SimulationGoalsCriterion,
}


__all__ = [
    "BaseCriterion",
    "CompositeCriterion",
    "ContainsKeywordsCriterion",
    "CRITERIA_REGISTRY",
    "ExactMatchCriterion",
    "FactualAccuracyCriterion",
    "HallucinationCriterion",
    "LLMCallerMixin",
    "LLMJudgeCriterion",
    "NodeOrderMatchCriterion",
    "ResponseMatchCriterion",
    "RougeMatchCriterion",
    "RubricBasedCriterion",
    "SafetyCriterion",
    "SimulationGoalsCriterion",
    "SyncCriterion",
    "TemplatedLLMCriterion",
    "ToolNameMatchCriterion",
    "TrajectoryMatchCriterion",
    "WeightedCriterion",
]
