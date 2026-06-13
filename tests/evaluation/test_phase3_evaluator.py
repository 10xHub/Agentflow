"""
Tests for Phase 3: AgentEvaluator and reporters.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentflow.qa.evaluation import (
    AgentEvaluator,
    CriteriaConfig,
    CriterionConfig,
    CriterionResult,
    EvalCase,
    EvalCaseResult,
    EvalConfig,
    EvalReport,
    EvalSet,
    EvalSummary,
    EvaluationRunner,
    Invocation,
    MatchType,
    MessageContent,
    TrajectoryCollector,
)
from agentflow.qa.evaluation.reporters.console import ConsoleReporter, Colors, print_report
from agentflow.qa.evaluation.reporters.json import JSONReporter, JUnitXMLReporter
from agentflow.qa.evaluation.reporters.html import HTMLReporter


# ============================================================================
# AgentEvaluator Tests
# ============================================================================


class TestAgentEvaluator:
    """Tests for the AgentEvaluator class."""

    def test_init_with_default_config(self):
        """Test evaluator initializes with default config."""
        mock_graph = MagicMock()
        mock_collector = MagicMock(spec=TrajectoryCollector)
        evaluator = AgentEvaluator(mock_graph, mock_collector)

        assert evaluator.graph is mock_graph
        assert evaluator.config is not None
        assert isinstance(evaluator.criteria, list)

    def test_init_with_custom_config(self):
        """Test evaluator initializes with custom config."""
        mock_graph = MagicMock()
        mock_collector = MagicMock(spec=TrajectoryCollector)
        config = EvalConfig(
            criteria=CriteriaConfig(
                trajectory=CriterionConfig(threshold=0.9),
                response_match=CriterionConfig(threshold=0.7),
            )
        )
        evaluator = AgentEvaluator(mock_graph, mock_collector, config=config)

        assert evaluator.config == config
        # Should have 2 enabled criteria
        assert len(evaluator.criteria) == 2

    def test_build_criteria(self):
        """Test criteria are built from config."""
        mock_graph = MagicMock()
        mock_collector = MagicMock(spec=TrajectoryCollector)
        config = EvalConfig(
            criteria=CriteriaConfig(
                trajectory=CriterionConfig(threshold=0.8),
                response_match=CriterionConfig(threshold=0.6),
            )
        )
        evaluator = AgentEvaluator(mock_graph, mock_collector, config=config)

        assert len(evaluator.criteria) == 2
        criterion_names = [c.name for c in evaluator.criteria]
        assert "tool_trajectory_avg_score" in criterion_names
        assert "response_match_score" in criterion_names

    def test_create_unknown_criterion(self):
        """Test unknown criterion returns None."""
        mock_graph = MagicMock()
        mock_collector = MagicMock(spec=TrajectoryCollector)
        config = EvalConfig(
            criteria=CriteriaConfig()  # all fields None — no criteria enabled
        )
        evaluator = AgentEvaluator(mock_graph, mock_collector, config=config)

        # No criteria set — evaluator should have empty list
        assert len(evaluator.criteria) == 0

    def test_load_eval_set_file_not_found(self):
        """Test loading non-existent eval set raises error."""
        mock_graph = MagicMock()
        mock_collector = MagicMock(spec=TrajectoryCollector)
        evaluator = AgentEvaluator(mock_graph, mock_collector)

        with pytest.raises(FileNotFoundError):
            evaluator._load_eval_set("/nonexistent/path.json")

    def test_load_eval_set_success(self):
        """Test loading eval set from file."""
        mock_graph = MagicMock()
        mock_collector = MagicMock(spec=TrajectoryCollector)
        evaluator = AgentEvaluator(mock_graph, mock_collector)

        eval_set = EvalSet(
            eval_set_id="test_set",
            name="Test Set",
            eval_cases=[
                EvalCase.single_turn(
                    eval_id="case1",
                    user_query="Hello",
                    expected_response="Hi there",
                )
            ],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(eval_set.model_dump_json())
            temp_path = f.name

        try:
            loaded = evaluator._load_eval_set(temp_path)
            assert loaded.eval_set_id == "test_set"
            assert len(loaded.eval_cases) == 1
        finally:
            Path(temp_path).unlink()

    def test_execution_from_collector(self):
        """Test building ExecutionResult from collector fields."""
        from agentflow.qa.evaluation import ToolCall as EvalToolCall

        collector = TrajectoryCollector()
        collector.tool_calls = [
            EvalToolCall(name="get_weather", args={"city": "NYC"}),
        ]
        collector.final_response = "Hi there!"
        collector.node_visits = ["agent"]

        result = AgentEvaluator._build_execution_result(
            node_responses=collector.node_responses,
            tool_calls=collector.tool_calls,
            trajectory=collector.trajectory,
            node_visits=collector.node_visits,
            actual_response=collector.final_response,
            duration_seconds=collector.duration,
        )
        assert result.actual_response == "Hi there!"
        assert len(result.tool_calls) == 1
        assert result.node_visits == ["agent"]

    def test_execution_from_collector_empty(self):
        """Test building ExecutionResult from empty collector fields."""
        collector = TrajectoryCollector()

        result = AgentEvaluator._build_execution_result(
            node_responses=collector.node_responses,
            tool_calls=collector.tool_calls,
            trajectory=collector.trajectory,
            node_visits=collector.node_visits,
            actual_response=collector.final_response,
            duration_seconds=collector.duration,
        )
        assert result.actual_response == ""
        assert len(result.tool_calls) == 0


class TestEvaluationRunner:
    """Tests for the EvaluationRunner class."""

    def test_init(self):
        """Test runner initialization."""
        runner = EvaluationRunner()
        assert runner.default_config is not None
        assert runner.results == {}

    def test_summary_empty(self):
        """Test summary with no results."""
        runner = EvaluationRunner()
        summary = runner.summary

        assert summary["total_evaluations"] == 0


# ============================================================================
# ConsoleReporter Tests
# ============================================================================


class TestConsoleReporter:
    """Tests for the ConsoleReporter class."""

    def test_init_with_color(self):
        """Test reporter initializes with color enabled."""
        reporter = ConsoleReporter(use_color=True)
        assert reporter.use_color is True

    def test_init_without_color(self):
        """Test reporter initializes with color disabled."""
        reporter = ConsoleReporter(use_color=False)
        assert reporter.use_color is False

    def test_report_prints_output(self, capsys):
        """Test report prints to stdout."""
        reporter = ConsoleReporter(use_color=False)

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[
                        CriterionResult.success(
                            criterion="test",
                            score=1.0,
                            threshold=0.8,
                        )
                    ],
                )
            ],
            eval_set_name="Test Set",
        )

        reporter.report(report)
        captured = capsys.readouterr()

        assert "Test Set" in captured.out
        assert "1" in captured.out  # Total cases

    def test_verbose_mode(self, capsys):
        """Test verbose mode shows more details."""
        reporter = ConsoleReporter(use_color=False, verbose=True)

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[
                        CriterionResult.success(
                            criterion="trajectory_match",
                            score=0.9,
                            threshold=0.8,
                        )
                    ],
                    name="Test Case 1",
                )
            ],
        )

        reporter.report(report)
        captured = capsys.readouterr()

        assert "Test Case 1" in captured.out


class TestPrintReport:
    """Test the print_report convenience function."""

    def test_print_report(self, capsys):
        """Test print_report function."""
        report = EvalReport.create(
            eval_set_id="test",
            results=[],
        )

        print_report(report, use_color=False)
        captured = capsys.readouterr()

        assert "test" in captured.out


# ============================================================================
# JSONReporter Tests
# ============================================================================


class TestJSONReporter:
    """Tests for the JSONReporter class."""

    def test_to_dict(self):
        """Test converting report to dict."""
        reporter = JSONReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[],
                )
            ],
        )

        data = reporter.to_dict(report)
        assert data["eval_set_id"] == "test_set"
        assert len(data["results"]) == 1

    def test_to_json(self):
        """Test converting report to JSON string."""
        reporter = JSONReporter(indent=2)

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[],
        )

        json_str = reporter.to_json(report)
        data = json.loads(json_str)
        assert data["eval_set_id"] == "test_set"

    def test_save(self):
        """Test saving report to file."""
        reporter = JSONReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            reporter.save(report, str(path))

            assert path.exists()
            with path.open() as f:
                data = json.load(f)
            assert data["eval_set_id"] == "test_set"

    def test_exclude_details(self):
        """Test excluding details from output."""
        reporter = JSONReporter(include_details=False)

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[
                        CriterionResult.success(
                            criterion="test",
                            score=1.0,
                            threshold=0.8,
                            details={"key": "value"},
                        )
                    ],
                )
            ],
        )

        data = reporter.to_dict(report)
        # Details should be removed
        cr = data["results"][0]["criterion_results"][0]
        assert "details" not in cr


class TestJUnitXMLReporter:
    """Tests for the JUnitXMLReporter class."""

    def test_to_xml(self):
        """Test converting report to JUnit XML."""
        reporter = JUnitXMLReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[],
                    name="Test Case 1",
                )
            ],
        )

        xml = reporter.to_xml(report)
        assert "<?xml" in xml
        assert "testsuite" in xml
        assert "testcase" in xml

    def test_save(self):
        """Test saving to XML file."""
        reporter = JUnitXMLReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "junit.xml"
            reporter.save(report, str(path))

            assert path.exists()
            content = path.read_text()
            assert "testsuite" in content


# ============================================================================
# HTMLReporter Tests
# ============================================================================


class TestHTMLReporter:
    """Tests for the HTMLReporter class."""

    def test_to_html(self):
        """Test converting report to HTML."""
        reporter = HTMLReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[],
                    name="Test Case 1",
                )
            ],
            eval_set_name="Test Set",
        )

        html = reporter.to_html(report)
        assert "<!DOCTYPE html>" in html
        assert "Test Set" in html
        assert "Test Case 1" in html

    def test_save(self):
        """Test saving to HTML file."""
        reporter = HTMLReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            reporter.save(report, str(path))

            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_render_failed_case(self):
        """Test rendering a failed case."""
        reporter = HTMLReporter()

        result = EvalCaseResult(
            eval_id="failed_case",
            name="Failed Case",
            passed=False,
            criterion_results=[
                CriterionResult(
                    criterion="test",
                    score=0.5,
                    passed=False,
                    threshold=0.8,
                )
            ],
        )

        html = reporter._render_case(result)
        assert "fail" in html
        assert "Failed Case" in html

    def test_render_error_case(self):
        """Test rendering an error case."""
        reporter = HTMLReporter()

        result = EvalCaseResult.failure(
            eval_id="error_case",
            error="Something went wrong",
            name="Error Case",
        )

        html = reporter._render_case(result)
        assert "error" in html
        assert "Something went wrong" in html


# ============================================================================
# Additional Reporter Tests for Coverage
# ============================================================================

from agentflow.qa.evaluation.dataset.eval_set import ToolCall, TrajectoryStep, StepType
from agentflow.qa.evaluation.eval_result import NodeDetail
from agentflow.qa.evaluation.token_usage import TokenUsage

class NodeResponseObj:
    node_name = "object_node"
    response_text = "hello from object node"
    tool_call_names = ["other_tool"]
    is_final = False
    has_tool_calls = False
    timestamp = 140.0
    input_messages = [{"role": "system", "content": "system_prompt"}]

def build_comprehensive_report():
    # 1. Tool Call
    tc = ToolCall(
        name="test_tool",
        args={"arg1": "val1"},
        call_id="call_123",
        result="success_result"
    )

    # 2. Trajectory Steps
    step1 = TrajectoryStep(
        step_type=StepType.TOOL,
        name="test_tool",
        args={"arg1": "val1"},
        timestamp=100.0,
        metadata={"meta1": "val1"}
    )

    # 3. Node Response (dict format)
    nr_dict = {
        "node_name": "agent_node",
        "response_text": "hello from node",
        "tool_call_names": ["test_tool"],
        "is_final": True,
        "has_tool_calls": True,
        "timestamp": 120.0,
        "input_messages": [{"role": "user", "content": "hello"}]
    }

    # 4. Node Detail (object format for node_details)
    node_detail = NodeDetail(
        node_name="other_node",
        input_messages=[{"role": "user", "content": "hi"}],
        response_text="hi response",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        timestamp=130.0
    )

    # 5. Criterion Results
    cr1 = CriterionResult(
        criterion="traj_crit",
        score=0.4,
        passed=False,
        threshold=0.8,
        details={"reason": "traj failed reason", "extra": "extra_detail"},
        error="eval error message"
    )
    cr2 = CriterionResult.success(
        criterion="resp_crit",
        score=0.9,
        threshold=0.8,
        details={"reason": "resp passed reason"}
    )

    # 6. Case Results
    # Result 1: Failed
    r1 = EvalCaseResult.success(
        eval_id="case1",
        name="Case One",
        criterion_results=[cr1, cr2],
        actual_trajectory=[step1],
        actual_tool_calls=[tc],
        actual_response="Hello world",
        messages=[{"role": "user", "content": "query"}, {"role": "assistant", "content": "Hello world"}],
        node_responses=[],
        node_visits=["start_node", "agent_node"],
        duration_seconds=2.5,
        metadata={"case_meta": "meta_val"},
        turn_results=[{
            "turn_index": 0,
            "user_input": "query",
            "agent_response": "Hello world",
            "tool_calls": [{"name": "test_tool"}],
            "node_visits": ["start_node", "agent_node"]
        }],
        node_details=[node_detail]
    )
    
    # Assign attributes that bypass basic pydantic constructor validation
    r1.actual_trajectory = [step1, "simple_trajectory_step"]
    r1.node_responses = [nr_dict, NodeResponseObj()]

    # Result 2: Error
    r2 = EvalCaseResult.failure(
        eval_id="case2",
        error="Case execution crash",
        name="Case Two",
        duration_seconds=1.2
    )

    # 7. Create report
    report = EvalReport.create(
        eval_set_id="test_set_123",
        eval_set_name="Comprehensive Test Set",
        results=[r1, r2],
        config_used={"eval_param": "value"}
    )
    report.metadata = {"report_meta": "val"}
    return report

def test_colors_disable():
    # Save all original attributes from Colors
    orig_attrs = {k: getattr(Colors, k) for k in ["RED", "RESET", "BOLD", "DIM", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN", "WHITE", "BG_RED", "BG_GREEN"]}
    try:
        Colors.disable()
        assert Colors.RED == ""
    finally:
        for k, v in orig_attrs.items():
            setattr(Colors, k, v)

def test_json_reporter_quick_save():
    report = build_comprehensive_report()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "quick.json"
        JSONReporter.quick_save(report, str(path))
        assert path.exists()

def test_json_reporter_generate():
    report = build_comprehensive_report()
    reporter = JSONReporter()
    
    # 1. output_dir = None -> returns JSON string
    json_str = reporter.generate(report)
    assert "Comprehensive Test Set" in json_str
    
    # 2. output_dir is provided -> saves to file and returns path
    with tempfile.TemporaryDirectory() as tmpdir:
        res_path = reporter.generate(report, output_dir=tmpdir)
        assert Path(res_path).exists()
        assert res_path.endswith("report.json")

def test_json_reporter_exclusions():
    report = build_comprehensive_report()
    
    # Disable everything
    reporter = JSONReporter(
        include_details=False,
        include_trajectory=False,
        include_node_responses=False,
        include_actual_response=False,
        include_tool_call_details=False,
    )
    data = reporter.to_dict(report)
    
    for result in data["results"]:
        assert "actual_trajectory" not in result
        assert "actual_tool_calls" not in result
        assert "node_responses" not in result
        assert "node_details" not in result
        assert "actual_response" not in result
        for cr in result.get("criterion_results", []):
            assert "details" not in cr

def test_junit_reporter_generate():
    report = build_comprehensive_report()
    reporter = JUnitXMLReporter()
    
    # 1. output_dir = None -> returns XML string
    xml_str = reporter.generate(report)
    assert "Comprehensive Test Set" in xml_str
    
    # 2. output_dir is provided -> saves to file and returns path
    with tempfile.TemporaryDirectory() as tmpdir:
        res_path = reporter.generate(report, output_dir=tmpdir)
        assert Path(res_path).exists()
        assert res_path.endswith("junit.xml")

def test_junit_reporter_details():
    report = build_comprehensive_report()
    reporter = JUnitXMLReporter()
    xml_str = reporter.to_xml(report)
    
    assert "config_used" in xml_str
    assert "report_meta" in xml_str
    assert '<error message="Case execution crash">' in xml_str
    assert '<failure type="traj_crit"' in xml_str
    assert "=== Agent Response ===" in xml_str
    assert "=== Tool Calls" in xml_str
    assert "=== Trajectory" in xml_str
    assert "=== Node Visits ===" in xml_str
    assert "=== Node Responses" in xml_str
    assert "=== Messages" in xml_str
    assert "=== Metadata ===" in xml_str
    assert "=== Turn Results" in xml_str
    assert "=== Criteria Results ===" in xml_str
    assert "traj failed reason" in xml_str
    assert "eval error message" in xml_str

def test_console_reporter_generate():
    report = build_comprehensive_report()
    reporter = ConsoleReporter(use_color=False)
    res = reporter.generate(report)
    assert res is None

def test_console_reporter_unicode_error():
    report = build_comprehensive_report()
    mock_output = MagicMock()
    mock_output.encoding = "ascii"
    
    def write_side_effect(text):
        if any(ord(c) > 127 for c in text):
            raise UnicodeEncodeError("ascii", text, 0, len(text), "non-ascii")
        return len(text)
        
    mock_output.write.side_effect = write_side_effect
    
    reporter = ConsoleReporter(use_color=False, output=mock_output)
    reporter.report(report)

def test_console_reporter_partial_stats(capsys):
    cr_passed = CriterionResult.success(
        criterion="yellow_crit",
        score=0.9,
        threshold=0.8,
    )
    cr_failed = CriterionResult.success(
        criterion="yellow_crit",
        score=0.5,
        threshold=0.8,
    )
    r1 = EvalCaseResult.success(
        eval_id="case1",
        criterion_results=[cr_passed],
    )
    r2 = EvalCaseResult.success(
        eval_id="case2",
        criterion_results=[cr_failed],
    )
    
    report = EvalReport.create(
        eval_set_id="partial_set",
        results=[r1, r2],
    )
    report.summary.pass_rate = 0.5
    report.summary.criterion_stats = {
        "yellow_crit": {
            "pass_rate": 0.7,
            "avg_score": 0.75,
            "passed": 7,
            "total": 10,
        },
        "red_crit": {
            "pass_rate": 0.3,
            "avg_score": 0.35,
            "passed": 3,
            "total": 10,
        }
    }
    
    reporter = ConsoleReporter(use_color=False, verbose=True)
    reporter.report(report)
    
    captured = capsys.readouterr()
    assert "PARTIAL" in captured.out
    assert "yellow_crit" in captured.out
    assert "red_crit" in captured.out

def test_console_reporter_comprehensive_printing(capsys):
    report = build_comprehensive_report()
    
    reporter = ConsoleReporter(
        use_color=True,
        verbose=True,
        include_trajectory=True,
        include_actual_response=True,
    )
    reporter.report(report)
    
    captured = capsys.readouterr()
    assert "Comprehensive Test Set" in captured.out
    assert "Case One" in captured.out
    assert "Case Two" in captured.out
    assert "ERROR" in captured.out
    assert "Case execution crash" in captured.out
    assert "case_meta" in captured.out
    assert "test_tool" in captured.out
    assert "call_id" in captured.out
    assert "simple_trajectory_step" in captured.out
    assert "object_node" in captured.out
    assert "hello from object node" in captured.out
    assert "Turn 0:" in captured.out
    assert "traj failed reason" in captured.out
    assert "extra_detail" in captured.out
