"""
Pytest integration utilities for agent evaluation.

This module provides decorators and fixtures for integrating
agent evaluations into pytest test suites.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from agentflow.core.graph.compiled_graph import CompiledGraph
    from agentflow.qa.evaluation.config.eval_config import EvalConfig
    from agentflow.qa.evaluation.eval_result import EvalReport


class EvalTestCase:
    """Represents a single evaluation test case for pytest.

    This class wraps EvalCase and provides pytest-friendly methods
    for assertion and reporting.
    """

    def __init__(
        self,
        eval_id: str,
        name: str = "",
        description: str = "",
    ):
        """Initialize eval test case.

        Args:
            eval_id: Unique identifier for the case.
            name: Human-readable name.
            description: Description of what the test validates.
        """
        self.eval_id = eval_id
        self.name = name
        self.description = description

    def __repr__(self) -> str:
        return f"EvalTestCase({self.name or self.eval_id})"


def eval_test(
    eval_file: str | None = None,
    config: EvalConfig | None = None,
    threshold: float = 1.0,
) -> Callable:
    """Decorator for agent evaluation tests.

    This decorator wraps a test function that should run agent
    evaluation against an eval set file.

    Args:
        eval_file: Path to the eval set JSON file.
        config: Optional evaluation configuration.
        threshold: Required pass rate (0.0 to 1.0).

    Returns:
        Decorated test function.

    Example:
        ```python
        @eval_test("tests/fixtures/weather_agent.evalset.json")
        async def test_weather_agent():
            collector = TrajectoryCollector()
            _, mgr = make_trajectory_callback(collector)
            graph = build_weather_graph().compile(callback_manager=mgr)
            return graph, collector
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            import pytest

            from agentflow.qa.evaluation import AgentEvaluator
            from agentflow.qa.evaluation.config.eval_config import EvalConfig

            # Call the test function to get the graph and collector
            result = await func(*args, **kwargs)

            if result is None:
                pytest.skip("No graph returned from test function")
                return

            _expected_tuple_len = 2
            if isinstance(result, tuple) and len(result) == _expected_tuple_len:
                graph, collector = result
            else:
                pytest.fail(
                    "eval_test decorated function must return (graph, collector) tuple. "
                    "See eval_test docstring for usage."
                )
                return

            # Create evaluator
            eval_config = config or EvalConfig.default()
            evaluator = AgentEvaluator(graph, collector, config=eval_config)

            # Determine eval file path
            file_path = eval_file
            if file_path is None:
                # Try to find eval file based on test name
                test_name = func.__name__.replace("test_", "")
                possible_paths = [
                    f"tests/fixtures/{test_name}.evalset.json",
                    f"tests/eval/{test_name}.evalset.json",
                    f"eval/{test_name}.evalset.json",
                ]
                for path in possible_paths:
                    if Path(path).exists():
                        file_path = path
                        break

            if file_path is None or not Path(file_path).exists():
                pytest.fail(f"Eval file not found: {eval_file or 'auto-detected'}")
                return

            # Run evaluation
            report = await evaluator.evaluate(file_path, verbose=True)

            # Assert pass rate meets threshold
            if report.summary.pass_rate < threshold:
                failure_details = "\n".join(
                    f"  - {r.name or r.eval_id}: {r.error or ', '.join(c.criterion for c in r.failed_criteria)}"  # noqa: E501
                    for r in report.failed_cases
                )
                pytest.fail(
                    f"Evaluation failed: {report.summary.pass_rate:.1%} pass rate "
                    f"(threshold: {threshold:.1%})\n"
                    f"Failed cases:\n{failure_details}"
                )

        return wrapper

    return decorator


def assert_eval_passed(report: EvalReport, min_pass_rate: float = 1.0) -> None:
    """Assert that an evaluation report meets the pass rate threshold.

    Args:
        report: The evaluation report to check.
        min_pass_rate: Minimum required pass rate (0.0 to 1.0).

    Raises:
        AssertionError: If pass rate is below threshold.
    """
    if report.summary.pass_rate < min_pass_rate:
        failed_cases = [f"{r.name or r.eval_id}" for r in report.failed_cases]
        raise AssertionError(
            f"Evaluation pass rate {report.summary.pass_rate:.1%} "
            f"below threshold {min_pass_rate:.1%}. "
            f"Failed cases: {', '.join(failed_cases)}"
        )


def assert_criterion_passed(
    report: EvalReport,
    criterion: str,
    min_score: float = 0.0,
) -> None:
    """Assert that a specific criterion passed across all cases.

    Args:
        report: The evaluation report to check.
        criterion: Name of the criterion to check.
        min_score: Minimum required average score.

    Raises:
        AssertionError: If criterion doesn't meet requirements.
    """
    stats = report.summary.criterion_stats.get(criterion)
    if stats is None:
        raise AssertionError(f"Criterion '{criterion}' not found in report")

    avg_score = stats.get("avg_score", 0.0)
    if avg_score < min_score:
        raise AssertionError(
            f"Criterion '{criterion}' average score {avg_score:.2f} below minimum {min_score:.2f}"
        )


def parametrize_eval_cases(eval_file: str) -> Callable:
    """Parametrize a test with cases from an eval file.

    This decorator loads eval cases from a file and creates
    a parametrized test for each case.

    Args:
        eval_file: Path to the eval set JSON file.

    Returns:
        pytest.mark.parametrize decorator.

    Example:
        ```python
        @parametrize_eval_cases("tests/fixtures/weather_agent.evalset.json")
        async def test_single_case(eval_case):
            collector = TrajectoryCollector()
            _, mgr = make_trajectory_callback(collector)
            graph = build_agent().compile(callback_manager=mgr)
            evaluator = AgentEvaluator(graph, collector)
            result = await evaluator._evaluate_case(eval_case)
            assert result.passed
        ```
    """
    import pytest

    from agentflow.qa.evaluation.dataset.eval_set import EvalSet

    eval_set = EvalSet.from_file(eval_file)
    cases = [(case.eval_id, case) for case in eval_set.eval_cases]

    return pytest.mark.parametrize(
        "eval_case",
        [c[1] for c in cases],
        ids=[c[0] for c in cases],
    )


class EvalFixtures:
    """Collection of pytest fixtures for evaluation testing.

    Use this class to register evaluation fixtures in conftest.py.

    Example:
        ```python
        # conftest.py
        from agentflow.qa.evaluation.testing import EvalFixtures

        fixtures = EvalFixtures()
        fixtures.register()
        ```
    """

    def __init__(self, default_config: EvalConfig | None = None):
        """Initialize fixtures.

        Args:
            default_config: Default evaluation configuration.
        """
        self.default_config = default_config

    def evaluator_factory(self) -> Callable:
        """Create an evaluator factory fixture.

        Returns:
            Factory function that creates AgentEvaluator instances.
        """
        from agentflow.qa.evaluation import AgentEvaluator
        from agentflow.qa.evaluation.config.eval_config import EvalConfig

        default = self.default_config

        def factory(
            graph: CompiledGraph,
            collector: Any,
            config: EvalConfig | None = None,
        ) -> AgentEvaluator:
            return AgentEvaluator(
                graph, collector, config=config or default or EvalConfig.default()
            )

        return factory


class EvalPlugin:
    """Pytest plugin for agent evaluation.

    Provides pytest integration for running agent evaluations as part
    of the test suite.
    """

    def __init__(self):
        pass

    def pytest_configure(self, config):
        """Configure pytest with evaluation plugin."""

    def pytest_collection_modifyitems(self, config, items):
        """Modify collected test items."""


# Fixture-style helpers that can be used directly


async def run_eval(
    graph: CompiledGraph,
    collector: Any,
    eval_set_path: str,
    config: EvalConfig | None = None,
    verbose: bool = False,
) -> EvalReport:
    """Run evaluation and return report.

    Convenience function for running evaluations in tests.

    Args:
        graph: The compiled graph to evaluate. Must be compiled with
               the collector's callback_manager.
        collector: TrajectoryCollector wired into the graph at compile time.
        eval_set_path: Path to eval set JSON file.
        config: Optional evaluation configuration.
        verbose: Whether to log progress.

    Returns:
        Evaluation report.

    Example:
        ```python
        async def test_agent():
            collector = TrajectoryCollector(capture_all_events=True)
            _, callback_mgr = make_trajectory_callback(collector)
            graph = my_agent.compile(callback_manager=callback_mgr)
            report = await run_eval(graph, collector, "tests/fixtures/my_agent.evalset.json")
            assert report.summary.pass_rate == 1.0
        ```
    """
    from agentflow.qa.evaluation import AgentEvaluator
    from agentflow.qa.evaluation.config.eval_config import EvalConfig as _EvalConfig

    evaluator = AgentEvaluator(graph, collector, config or _EvalConfig.default())
    return await evaluator.evaluate(eval_set_path, verbose=verbose)


def create_eval_app(
    graph: Any,
    *,
    capture_all_events: bool = True,
) -> tuple[Any, Any]:
    """Compile a StateGraph for evaluation with a TrajectoryCollector.

    Handles all evaluation plumbing so the user only needs to provide
    their graph.  Returns the compiled app and the collector — ready
    for ``AgentEvaluator``.

    Args:
        graph: An *uncompiled* ``StateGraph`` instance.
        capture_all_events: Forward to ``TrajectoryCollector``.

    Returns:
        ``(compiled_app, collector)`` tuple.

    Example:
        ```python
        # conftest.py
        from agentflow.qa.evaluation.testing import create_eval_app


        @pytest.fixture(scope="session")
        def trajectory_app():
            graph = build_my_graph()  # <-- only thing the user writes
            return create_eval_app(graph)
        ```
    """
    from agentflow.qa.evaluation.collectors import TrajectoryCollector, make_trajectory_callback
    from agentflow.storage.checkpointer import InMemoryCheckpointer

    collector = TrajectoryCollector(capture_all_events=capture_all_events)
    _, callback_mgr = make_trajectory_callback(collector)
    app = graph.compile(
        checkpointer=InMemoryCheckpointer(),
        callback_manager=callback_mgr,
    )
    return app, collector


def create_simple_eval_set(
    eval_set_id: str,
    cases: list[tuple[str, str, str | None]],
) -> Any:
    """Create a simple eval set for testing.

    Args:
        eval_set_id: ID for the eval set.
        cases: List of (user_query, expected_response, name) tuples.

    Returns:
        EvalSet object.

    Example:
        ```python
        eval_set = create_simple_eval_set(
            "basic_tests",
            [
                ("Hello", "Hi there!", "greeting"),
                ("What is 2+2?", "4", "math"),
            ],
        )
        ```
    """
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase, EvalSet

    eval_cases = []
    for i, (query, expected, name) in enumerate(cases):
        case = EvalCase.single_turn(
            eval_id=f"case_{i}",
            user_query=query,
            expected_response=expected,
            name=name or f"Case {i}",
        )
        eval_cases.append(case)

    return EvalSet(
        eval_set_id=eval_set_id,
        name=eval_set_id,
        eval_cases=eval_cases,
    )
