"""
Evaluation dataset package.

Provides data models and builders for defining evaluation test cases.

Example:
    ```python
    from agentflow.qa.evaluation.dataset import (
        EvalSet,
        EvalCase,
        EvalSetBuilder,
        ToolCall,
    )

    # Quick build from tuples
    eval_set = EvalSetBuilder.quick(
        ("Hello", "Hi!"),
        ("Weather in London?", "It is sunny."),
    )

    # Build EvalCases directly
    cases = [
        EvalCase.single_turn(
            eval_id="nyc",
            user_query="Weather in NYC?",
            expected_response="It is sunny.",
            expected_tools=[ToolCall(name="get_weather")],
        ),
    ]
    eval_set = EvalSet(eval_set_id="weather", name="Weather Tests", eval_cases=cases)
    ```
"""

from .builder import EvalSetBuilder
from .eval_set import (
    EvalCase,
    EvalSet,
    Invocation,
    MessageContent,
    SessionInput,
    StepType,
    ToolCall,
    TrajectoryStep,
)


__all__ = [
    "EvalCase",
    # Core models
    "EvalSet",
    # Builder
    "EvalSetBuilder",
    "Invocation",
    "MessageContent",
    "SessionInput",
    "StepType",
    "ToolCall",
    "TrajectoryStep",
]
