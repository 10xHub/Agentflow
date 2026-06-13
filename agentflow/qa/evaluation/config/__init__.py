"""
Evaluation configuration package.

Provides configuration models and presets for agent evaluation.

Example:
    ```python
    from agentflow.qa.evaluation.config import EvalConfig, CriterionConfig
    from agentflow.qa.evaluation.config import EvalPresets, MatchType, Rubric

    # Use a preset
    config = EvalPresets.tool_usage(strict=True)

    # Or build custom config
    config = EvalConfig(
        criteria={
            "trajectory_match": CriterionConfig.trajectory(threshold=1.0),
        }
    )
    ```
"""

from .eval_config import (
    DEFAULT_JUDGE_MODEL,
    CriteriaConfig,
    CriterionConfig,
    EvalConfig,
    MatchType,
    ReporterConfig,
    Rubric,
    UserSimulatorConfig,
)
from .presets import EvalPresets


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "CriteriaConfig",
    "CriterionConfig",
    # Core config models
    "EvalConfig",
    # Presets
    "EvalPresets",
    "MatchType",
    "ReporterConfig",
    "Rubric",
    "UserSimulatorConfig",
]
