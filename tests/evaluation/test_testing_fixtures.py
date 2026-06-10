"""Tests for Pytest integration utilities in testing.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any
from pathlib import Path

from agentflow.qa.evaluation.testing import (
    EvalTestCase,
    eval_test,
    assert_eval_passed,
    assert_criterion_passed,
    parametrize_eval_cases,
    EvalFixtures,
    EvalPlugin,
    run_eval,
    create_eval_app,
    create_simple_eval_set,
)
from agentflow.qa.evaluation.eval_result import EvalReport
from agentflow.core.graph.compiled_graph import CompiledGraph

def test_eval_test_case_repr():
    case = EvalTestCase(eval_id="id1", name="name1", description="desc1")
    assert case.eval_id == "id1"
    assert case.name == "name1"
    assert case.description == "desc1"
    assert repr(case) == "EvalTestCase(name1)"

    case2 = EvalTestCase(eval_id="id2")
    assert repr(case2) == "EvalTestCase(id2)"

@pytest.mark.asyncio
async def test_eval_test_decorator_success():
    mock_report = MagicMock()
    mock_report.summary.pass_rate = 1.0

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=mock_report)

    with patch("agentflow.qa.evaluation.AgentEvaluator", return_value=mock_evaluator) as mock_class:
        with patch("agentflow.qa.evaluation.testing.Path.exists", return_value=True):
            @eval_test(eval_file="dummy.json", threshold=0.8)
            async def my_test():
                return "graph", "collector"

            await my_test()
            mock_class.assert_called_once()
            mock_evaluator.evaluate.assert_called_once_with("dummy.json", verbose=True)

@pytest.mark.asyncio
async def test_eval_test_decorator_skips():
    import _pytest.outcomes

    @eval_test(eval_file="dummy.json")
    async def my_skip_test():
        return None

    with pytest.raises(_pytest.outcomes.Skipped):
        await my_skip_test()

@pytest.mark.asyncio
async def test_eval_test_decorator_fails_invalid_return():
    import _pytest.outcomes

    @eval_test(eval_file="dummy.json")
    async def my_fail_test():
        return "not-a-tuple"

    with pytest.raises(_pytest.outcomes.Failed):
        await my_fail_test()

@pytest.mark.asyncio
async def test_eval_test_decorator_fails_threshold_not_met():
    import _pytest.outcomes

    mock_report = MagicMock()
    mock_report.summary.pass_rate = 0.5
    mock_report.failed_cases = [
        MagicMock(eval_id="case_1", name="Case One", error="failed", failed_criteria=[])
    ]

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=mock_report)

    with patch("agentflow.qa.evaluation.AgentEvaluator", return_value=mock_evaluator):
        with patch("agentflow.qa.evaluation.testing.Path.exists", return_value=True):
            @eval_test(eval_file="dummy.json", threshold=0.9)
            async def my_fail_threshold_test():
                return "graph", "collector"

            with pytest.raises(_pytest.outcomes.Failed):
                await my_fail_threshold_test()

@pytest.mark.asyncio
async def test_eval_test_decorator_auto_detect_path():
    mock_report = MagicMock()
    mock_report.summary.pass_rate = 1.0
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=mock_report)

    with patch("agentflow.qa.evaluation.AgentEvaluator", return_value=mock_evaluator):
        with patch("agentflow.qa.evaluation.testing.Path.exists", return_value=True):
            @eval_test()
            async def test_my_custom_scenario():
                return "graph", "collector"

            await test_my_custom_scenario()

def test_assert_eval_passed():
    report = MagicMock()
    report.summary.pass_rate = 0.9
    report.failed_cases = [MagicMock(eval_id="c1", name="", failed_criteria=[])]
    
    assert_eval_passed(report, min_pass_rate=0.8)

    with pytest.raises(AssertionError, match="Evaluation pass rate 90.0% below threshold 95.0%"):
        assert_eval_passed(report, min_pass_rate=0.95)

def test_assert_criterion_passed():
    report = MagicMock()
    report.summary.criterion_stats = {
        "accuracy": {"avg_score": 0.85}
    }

    with pytest.raises(AssertionError, match="Criterion 'safety' not found"):
        assert_criterion_passed(report, "safety")

    assert_criterion_passed(report, "accuracy", min_score=0.8)

    with pytest.raises(AssertionError, match="Criterion 'accuracy' average score 0.85 below minimum 0.90"):
        assert_criterion_passed(report, "accuracy", min_score=0.9)

def test_parametrize_eval_cases():
    mock_set = MagicMock()
    mock_case = MagicMock(eval_id="case1")
    mock_set.eval_cases = [mock_case]

    with patch("agentflow.qa.evaluation.dataset.eval_set.EvalSet.from_file", return_value=mock_set) as mock_from_file:
        decorator = parametrize_eval_cases("dummy_path.json")
        assert decorator is not None
        mock_from_file.assert_called_once_with("dummy_path.json")

def test_eval_fixtures():
    fixtures = EvalFixtures(default_config="my_config")
    assert fixtures.default_config == "my_config"

    with patch("agentflow.qa.evaluation.AgentEvaluator") as mock_eval:
        factory = fixtures.evaluator_factory()
        factory("graph", "collector")
        mock_eval.assert_called_once_with("graph", "collector", config="my_config")

def test_eval_plugin_noop():
    plugin = EvalPlugin()
    plugin.pytest_configure(None)
    plugin.pytest_collection_modifyitems(None, None)

@pytest.mark.asyncio
async def test_run_eval_success():
    with patch("agentflow.qa.evaluation.AgentEvaluator") as mock_eval_class:
        mock_eval = mock_eval_class.return_value
        mock_eval.evaluate = AsyncMock(return_value="eval_report")

        res = await run_eval("graph", "collector", "path.json")
        assert res == "eval_report"
        mock_eval.evaluate.assert_called_once_with("path.json", verbose=False)

def test_create_eval_app():
    mock_graph = MagicMock()
    mock_graph.compile.return_value = "compiled_app"

    app, collector = create_eval_app(mock_graph)
    assert app == "compiled_app"
    assert collector is not None
    mock_graph.compile.assert_called_once()

def test_create_simple_eval_set():
    eval_set = create_simple_eval_set("my_set_id", [("query", "expected", "test_name")])
    assert eval_set.eval_set_id == "my_set_id"
    assert eval_set.name == "my_set_id"
    assert len(eval_set.eval_cases) == 1
    assert eval_set.eval_cases[0].name == "test_name"
