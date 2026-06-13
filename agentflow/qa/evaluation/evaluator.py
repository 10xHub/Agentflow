"""
Main AgentEvaluator class for running agent evaluations.

This module provides the central entry point for evaluating agent graphs
against test cases defined in EvalSet files.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.collectors.trajectory_collector import (
    TrajectoryCollector,
    make_trajectory_callback,
)
from agentflow.qa.evaluation.config.eval_config import CriterionConfig, EvalConfig
from agentflow.qa.evaluation.criteria import CRITERIA_REGISTRY
from agentflow.qa.evaluation.criteria.base import BaseCriterion
from agentflow.qa.evaluation.dataset.eval_set import EvalCase, EvalSet
from agentflow.qa.evaluation.eval_result import (
    CriterionResult,
    EvalCaseResult,
    EvalReport,
)
from agentflow.qa.evaluation.execution.result import ExecutionResult, NodeResponseData
from agentflow.utils.callbacks import CallbackManager


if TYPE_CHECKING:
    from agentflow.core.graph.compiled_graph import CompiledGraph

logger = logging.getLogger("agentflow.evaluation")


class AgentEvaluator:
    """Main class for running agent evaluations.

    The AgentEvaluator orchestrates the evaluation process by:
    1. Loading evaluation sets from files or accepting EvalSet objects
    2. Running the agent graph for each test case
    3. Collecting tool calls, trajectory, and final response via TrajectoryCollector
    4. Evaluating against configured criteria built from EvalConfig
    5. Generating comprehensive EvalReport objects

    Attributes:
        graph: The compiled agent graph to evaluate.
        config: Evaluation configuration with criteria settings.
        collector: TrajectoryCollector wired into the graph at compile time.
        criteria: List of active BaseCriterion instances built from config.

    Example:
        ```python
        from agentflow.qa.evaluation import AgentEvaluator, EvalConfig
        from agentflow.qa.evaluation.collectors import TrajectoryCollector, make_trajectory_callback

        collector = TrajectoryCollector(capture_all_events=True)
        _, callback_mgr = make_trajectory_callback(collector)
        compiled_graph = my_state_graph.compile(callback_manager=callback_mgr)

        evaluator = AgentEvaluator(compiled_graph, collector, config=EvalConfig.default())

        # Run evaluation from a JSON file
        report = await evaluator.evaluate("tests/fixtures/my_tests.evalset.json")

        # Or from an EvalSet object
        report = await evaluator.evaluate(eval_set)

        print(report.format_summary())
        ```
    """

    def __init__(
        self,
        graph: CompiledGraph,
        collector: TrajectoryCollector,
        config: EvalConfig | None = None,
    ):
        """Initialize the evaluator.

        Args:
            graph: The compiled agent graph to evaluate. Must be compiled with
                   the collector's callback_manager (via make_trajectory_callback).
            collector: TrajectoryCollector wired into the graph at compile time.
                       Reset automatically before each case.
            config: Optional evaluation configuration. Uses defaults if not provided.
        """
        self.graph = graph
        self.config = config or EvalConfig.default()
        self.collector = collector
        self.criteria = self._build_criteria()
        self._run_id = uuid.uuid4().hex[:8]

    # ------------------------------------------------------------------
    # Criteria construction
    # ------------------------------------------------------------------

    def _build_criteria(self) -> list[BaseCriterion]:
        """Build enabled criterion instances from ``self.config.criteria``.

        Iterates over all entries in the config's criteria mapping, skips
        disabled ones, and delegates construction to ``_create_criterion``.

        Returns:
            Ordered list of ``BaseCriterion`` instances ready for evaluation.
        """
        criteria = []

        for name in self.config.criteria.model_fields:
            criterion_config = getattr(self.config.criteria, name)
            if criterion_config is None:
                continue
            if not criterion_config.enabled:
                continue

            criterion = self._create_criterion(name, criterion_config)
            if criterion:
                criteria.append(criterion)

        return criteria

    def _create_criterion(
        self,
        name: str,
        criterion_config: CriterionConfig,
    ) -> BaseCriterion | None:
        """Create a criterion instance by name using the shared registry."""
        criterion_class = CRITERIA_REGISTRY.get(name)
        if criterion_class:
            return criterion_class(config=criterion_config)

        logger.warning(
            "Unknown criterion '%s'. Available: %s",
            name,
            ", ".join(sorted(CRITERIA_REGISTRY)),
        )
        return None

    # ------------------------------------------------------------------
    # Public evaluation entry points
    # ------------------------------------------------------------------

    async def evaluate_case(self, case: EvalCase) -> EvalCaseResult:
        """Run evaluation for a single EvalCase.

        Runs the graph, extracts execution data, and evaluates all
        configured criteria.  Useful for per-case pytest assertions:

            result = await evaluator.evaluate_case(case)
            assert result.passed, result.criterion_results

        Args:
            case: The evaluation case to run.

        Returns:
            EvalCaseResult with scores and pass/fail status.
        """
        return await self._evaluate_case(case)

    async def evaluate(
        self,
        eval_set: EvalSet | str,
        parallel: bool = False,
        max_concurrency: int = 4,
        verbose: bool = False,
        output_dir: str | None = None,
    ) -> EvalReport:
        """Run evaluation on an eval set.

        Args:
            eval_set: EvalSet object or path to JSON file.
            parallel: Whether to run cases in parallel.
            max_concurrency: Maximum concurrent case evaluations.
            verbose: Whether to log detailed progress.
            output_dir: Override the report output directory. If ``None``,
                the value from ``ReporterConfig.output_dir`` is used.
                Relative paths are resolved against the current working
                directory.

        Returns:
            Complete evaluation report with results and summary.
        """
        if isinstance(eval_set, str):
            eval_set = self._load_eval_set(eval_set)

        if verbose:
            logger.info(
                "Starting evaluation of '%s' with %d cases",
                eval_set.name or eval_set.eval_set_id,
                len(eval_set.eval_cases),
            )

        if parallel and len(eval_set.eval_cases) > 1:
            results = await self._evaluate_parallel(
                eval_set.eval_cases,
                max_concurrency,
                verbose,
            )
        else:
            results = await self._evaluate_sequential(
                eval_set.eval_cases,
                verbose,
            )

        report = EvalReport.create(
            eval_set_id=eval_set.eval_set_id,
            results=results,
            eval_set_name=eval_set.name,
            config_used=self.config.model_dump(),
        )

        if verbose:
            logger.info(
                "Evaluation complete: %d/%d passed (%.1f%%)",
                report.summary.passed_cases,
                report.summary.total_cases,
                report.summary.pass_rate * 100,
            )

        # --- Auto-invoke reporters if configured ---
        self._run_reporters(report, output_dir=output_dir)

        return report

    # ------------------------------------------------------------------
    # Reporter integration
    # ------------------------------------------------------------------

    def _run_reporters(
        self,
        report: EvalReport,
        output_dir: str | None = None,
    ) -> None:
        """Invoke reporters according to ``self.config.reporter``.

        Failures are logged but never propagated — the evaluation result
        is always returned to the caller regardless of reporter errors.

        Args:
            report: The evaluation report to render.
            output_dir: Optional override for the report output directory.
        """
        try:
            from agentflow.qa.evaluation.reporters.manager import ReporterManager

            reporter_cfg = self.config.reporter
            if not reporter_cfg.enabled:
                return

            manager = ReporterManager(reporter_cfg)
            output = manager.run_all(report, output_dir=output_dir)

            if output.has_errors:
                for name, err in output.errors:
                    logger.warning("Reporter '%s' error: %s", name, err)
        except Exception as exc:
            logger.error("Reporter pipeline failed: %s", exc)

    # ------------------------------------------------------------------
    # Sequential / parallel runners
    # ------------------------------------------------------------------

    async def _evaluate_sequential(
        self,
        cases: list[EvalCase],
        verbose: bool = False,
    ) -> list[EvalCaseResult]:
        """Evaluate cases one at a time, in order.

        Args:
            cases: Ordered list of EvalCase objects to evaluate.
            verbose: When True, logs per-case progress and pass/fail status.

        Returns:
            List of EvalCaseResult objects in the same order as *cases*.
        """
        results = []
        for i, case in enumerate(cases):
            if verbose:
                logger.info(
                    "Evaluating case %d/%d: %s",
                    i + 1,
                    len(cases),
                    case.name or case.eval_id,
                )
            result = await self._evaluate_case(case)
            results.append(result)
            if verbose:
                logger.info(
                    "  -> %s (%.2fs)",
                    "PASSED" if result.passed else "FAILED",
                    result.duration_seconds,
                )
        return results

    async def _evaluate_parallel(
        self,
        cases: list[EvalCase],
        max_concurrency: int,
        verbose: bool = False,
    ) -> list[EvalCaseResult]:
        """Evaluate cases concurrently using an asyncio Semaphore.

        Creates a **fresh TrajectoryCollector per case** to avoid the race
        condition that would occur if cases shared ``self.collector``.

        Args:
            cases: List of EvalCase objects to evaluate.
            max_concurrency: Maximum number of cases allowed to run at once.
            verbose: Unused in this path; kept for API symmetry with
                     ``_evaluate_sequential``.

        Returns:
            List of EvalCaseResult objects in the same order as *cases*.
            Failed-with-exception cases are represented as ``EvalCaseResult.failure``
            entries rather than raising.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run(case: EvalCase) -> EvalCaseResult:
            async with semaphore:
                # Per-case collector + callback manager to avoid cross-case bleed
                local_collector = TrajectoryCollector(
                    capture_all_events=self.collector.capture_all_events,
                )
                return await self._evaluate_case(case, collector_override=local_collector)

        raw = await asyncio.gather(*[_run(c) for c in cases], return_exceptions=True)

        results = []
        for i, item in enumerate(raw):
            if isinstance(item, Exception):
                results.append(
                    EvalCaseResult.failure(
                        eval_id=cases[i].eval_id,
                        error=str(item),
                        name=cases[i].name,
                    )
                )
            else:
                results.append(item)
        return results

    # ------------------------------------------------------------------
    # Core: run one case
    # ------------------------------------------------------------------

    @staticmethod
    def _build_execution_result(
        node_responses: list[Any],
        tool_calls: list[Any],
        trajectory: list[Any],
        node_visits: list[str],
        actual_response: str,
        duration_seconds: float,
    ) -> ExecutionResult:
        """Build an ExecutionResult from raw accumulated lists.

        De-duplicates messages by role+content key, serialises NodeResponse
        dataclasses to NodeResponseData Pydantic models.  Used for both
        single-turn (pass collector fields directly) and multi-turn
        (pass the accumulated lists from all turns).
        """
        messages: list[dict[str, Any]] = []
        seen_contents: set[str] = set()
        for nr in node_responses:
            for msg in nr.input_messages:
                key = f"{msg.get('role', '')}:{msg.get('content', '')}"
                if key not in seen_contents:
                    seen_contents.add(key)
                    messages.append(msg)

        node_resp_data = [
            NodeResponseData(
                node_name=nr.node_name,
                input_messages=nr.input_messages,
                response_text=nr.response_text,
                has_tool_calls=nr.has_tool_calls,
                tool_call_names=nr.tool_call_names,
                is_final=nr.is_final,
                timestamp=nr.timestamp,
                token_usage=nr.token_usage,
            )
            for nr in node_responses
        ]

        return ExecutionResult(
            actual_response=actual_response,
            tool_calls=list(tool_calls),
            trajectory=list(trajectory),
            messages=messages,
            node_responses=node_resp_data,
            node_visits=list(node_visits),
            duration_seconds=duration_seconds,
        )

    async def _run_conversation_turns(
        self,
        case: EvalCase,
        collector: TrajectoryCollector,
        graph: Any,
        config: dict[str, Any],
        case_start_time: float,
    ) -> tuple[ExecutionResult, list[dict[str, Any]]] | EvalCaseResult:
        """Run all conversation turns for a case, accumulating execution data.

        Returns either a tuple of (ExecutionResult, turn_results) on success,
        or an EvalCaseResult failure if a turn raises an exception.
        """
        from agentflow.core.state import Message

        cumulative_messages: list[Any] = []
        turn_results: list[dict[str, Any]] = []
        all_tool_calls: list[Any] = []
        all_trajectory: list[Any] = []
        all_node_visits: list[str] = []
        all_node_responses: list[Any] = []
        last_response: str = ""
        cumulative_start: float | None = None
        cumulative_end: float | None = None

        for turn_idx, invocation in enumerate(case.conversation):
            collector.reset()
            user_text = invocation.user_content.get_text()
            cumulative_messages.append(Message.text_message(user_text, role="user"))

            try:
                await graph.ainvoke({"messages": list(cumulative_messages)}, config=config)
            except Exception as exc:
                logger.warning(
                    "Graph execution failed for case %s turn %d: %s",
                    case.eval_id,
                    turn_idx,
                    exc,
                )
                return EvalCaseResult.failure(
                    eval_id=case.eval_id,
                    error=f"Graph execution error on turn {turn_idx}: {exc}",
                    name=case.name,
                    duration_seconds=time.time() - case_start_time,
                )

            turn_response = collector.final_response or ""
            turn_results.append(
                {
                    "turn_index": turn_idx,
                    "user_input": user_text,
                    "agent_response": turn_response,
                    "tool_calls": [tc.model_dump() for tc in collector.tool_calls],
                    "node_visits": list(collector.node_visits),
                    "trajectory_steps": len(collector.trajectory),
                }
            )

            all_tool_calls.extend(collector.tool_calls)
            all_trajectory.extend(collector.trajectory)
            all_node_visits.extend(collector.node_visits)
            all_node_responses.extend(collector.node_responses)
            last_response = turn_response

            if collector.start_time is not None:
                if cumulative_start is None:
                    cumulative_start = collector.start_time
                cumulative_end = collector.end_time

            if collector.final_response:
                cumulative_messages.append(
                    Message.text_message(collector.final_response, role="assistant")
                )

        total_duration = (
            cumulative_end - cumulative_start
            if cumulative_start is not None and cumulative_end is not None
            else 0.0
        )

        execution = self._build_execution_result(
            node_responses=all_node_responses,
            tool_calls=all_tool_calls,
            trajectory=all_trajectory,
            node_visits=all_node_visits,
            actual_response=last_response,
            duration_seconds=total_duration,
        )
        return execution, turn_results

    async def _evaluate_criteria(
        self,
        execution: ExecutionResult,
        case: EvalCase,
    ) -> list[CriterionResult]:
        """Run all configured criteria against an ExecutionResult."""
        criterion_results: list[CriterionResult] = []
        for criterion in self.criteria:
            try:
                criterion_results.append(await criterion.evaluate(execution, case))
            except Exception as exc:
                logger.error(
                    "Criterion '%s' failed for case %s: %s",
                    criterion.name,
                    case.eval_id,
                    exc,
                )
                criterion_results.append(
                    CriterionResult.failure(criterion=criterion.name, error=str(exc))
                )
        return criterion_results

    async def _evaluate_case(
        self,
        case: EvalCase,
        collector_override: TrajectoryCollector | None = None,
    ) -> EvalCaseResult:
        """Evaluate a single test case.

        Supports multi-turn conversations: iterates over every Invocation in
        case.conversation, feeding the agent one user message per turn and
        accumulating execution data across turns.

        Args:
            case: The evaluation case to run.
            collector_override: Optional per-case collector (used in parallel
                mode to avoid cross-case data bleed).
        """
        start_time = time.time()
        collector = collector_override or self.collector

        try:
            config: dict[str, Any] = {
                "thread_id": f"eval_{self._run_id}_{case.eval_id}",
                **case.session_input.config,
            }
            if case.session_input.user_id:
                config["user_id"] = case.session_input.user_id

            collector.reset()

            graph = self.graph
            if collector_override is not None:
                _, local_mgr = make_trajectory_callback(collector_override, config=config)
                graph._state_graph._container.bind_instance(CallbackManager, local_mgr)

            # Run all conversation turns and accumulate results
            result = await self._run_conversation_turns(case, collector, graph, config, start_time)
            if isinstance(result, EvalCaseResult):
                return result  # turn raised an exception — failure already built
            execution, turn_results = result

            # Evaluate all criteria against the full conversation ExecutionResult
            criterion_results = await self._evaluate_criteria(execution, case)

            # Aggregate agent + judge token usage
            from agentflow.qa.evaluation.token_usage import TokenUsage

            agent_token_usage = execution.token_usage
            criteria_token_usage = sum((cr.token_usage for cr in criterion_results), TokenUsage())

            return EvalCaseResult.success(
                eval_id=case.eval_id,
                criterion_results=criterion_results,
                actual_trajectory=execution.trajectory,
                actual_tool_calls=execution.tool_calls,
                actual_response=execution.actual_response,
                messages=execution.messages,
                node_responses=[nr.model_dump() for nr in execution.node_responses],
                node_visits=execution.node_visits,
                duration_seconds=time.time() - start_time,
                name=case.name,
                metadata=case.metadata if hasattr(case, "metadata") else {},
                turn_results=turn_results,
                token_usage=agent_token_usage + criteria_token_usage,
                agent_token_usage=agent_token_usage,
            )

        except Exception as exc:
            logger.error("Case evaluation failed unexpectedly: %s", exc)
            return EvalCaseResult.failure(
                eval_id=case.eval_id,
                error=str(exc),
                name=case.name,
                duration_seconds=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # File / module helpers
    # ------------------------------------------------------------------

    def _load_eval_set(self, path: str) -> EvalSet:
        """Load an EvalSet from a JSON file on disk.

        Args:
            path: Filesystem path to a ``.evalset.json`` file.

        Returns:
            Parsed ``EvalSet`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Eval set file not found: {path}")
        return EvalSet.from_file(str(file_path))

    @classmethod
    def evaluate_sync(
        cls,
        graph: CompiledGraph,
        collector: TrajectoryCollector,
        eval_set: EvalSet | str,
        config: EvalConfig | None = None,
        verbose: bool = False,
    ) -> EvalReport:
        """Synchronous wrapper for evaluate().

        Args:
            graph: The compiled graph to evaluate.
            collector: TrajectoryCollector wired into the graph at compile time.
            eval_set: EvalSet object or path to JSON file.
            config: Optional evaluation configuration.
            verbose: Whether to log progress.

        Returns:
            Complete evaluation report.
        """
        evaluator = cls(graph, collector, config)
        return asyncio.run(evaluator.evaluate(eval_set, verbose=verbose))

    @classmethod
    async def evaluate_file(
        cls,
        agent_module: str,
        eval_file: str,
        config_file: str | None = None,
    ) -> EvalReport:
        """Convenience method to evaluate from file paths.

        Loads the agent graph from a Python module and runs evaluation
        against the specified eval file.

        Args:
            agent_module: Python module path containing a 'graph' variable.
            eval_file: Path to the eval set JSON file.
            config_file: Optional path to eval config JSON file.

        Returns:
            Complete evaluation report.

        Example:
            ```python
            report = await AgentEvaluator.evaluate_file(
                agent_module="examples.react.react_weather_agent",
                eval_file="tests/fixtures/weather_agent.evalset.json",
            )
            ```
        """
        graph = cls._load_graph(agent_module)
        config = cls._load_config(config_file) if config_file else None
        from agentflow.qa.evaluation.collectors import TrajectoryCollector, make_trajectory_callback

        collector = TrajectoryCollector(capture_all_events=True)
        _, callback_mgr = make_trajectory_callback(collector)
        # Re-compile with the collector's callback_manager
        if hasattr(graph, "compile"):
            graph = graph.compile(callback_manager=callback_mgr)
        evaluator = cls(graph, collector, config)
        return await evaluator.evaluate(eval_file)

    @classmethod
    def _load_graph(cls, agent_module: str) -> CompiledGraph:
        """Load a compiled graph from a Python module.

        Looks for 'graph', 'compiled_graph', or 'agent_graph' attributes.
        If the attribute is an uncompiled StateGraph, compiles it first.

        Args:
            agent_module: Dotted module path, e.g. "examples.react.agent".

        Returns:
            The compiled graph.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the module has no recognised graph attribute.
        """
        import importlib

        module = importlib.import_module(agent_module)

        for attr_name in ("graph", "compiled_graph", "agent_graph", "app"):
            if hasattr(module, attr_name):
                obj = getattr(module, attr_name)
                # Compile if it's still a StateGraph
                if hasattr(obj, "compile"):
                    return obj.compile()
                return obj

        raise AttributeError(
            f"Module '{agent_module}' has no recognised graph attribute "
            f"('graph', 'compiled_graph', 'agent_graph', or 'app')"
        )

    @classmethod
    def _load_config(cls, config_file: str) -> EvalConfig:
        """Load an EvalConfig from a JSON file.

        Args:
            config_file: Filesystem path to a JSON config file whose
                         structure matches the ``EvalConfig`` schema.

        Returns:
            Validated ``EvalConfig`` instance.
        """
        import json

        data = json.loads(Path(config_file).read_text(encoding="utf-8"))
        return EvalConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Batch evaluation runner for multiple eval sets.

    Useful for running comprehensive evaluation suites across
    multiple agents and test sets.

    Example:
        ```python
        runner = EvaluationRunner()
        reports = await runner.run(
            [
                (graph_a, eval_set_a),
                (graph_b, eval_set_b),
            ]
        )
        print(runner.summary)
        ```
    """

    def __init__(self, default_config: EvalConfig | None = None):
        """Initialize the runner.

        Args:
            default_config: Default configuration applied to all evaluations.
        """
        self.default_config = default_config or EvalConfig.default()
        self.results: dict[str, EvalReport] = {}

    async def run(
        self,
        evaluations: list[tuple[CompiledGraph, TrajectoryCollector, EvalSet | str]],
        config: EvalConfig | None = None,
        verbose: bool = False,
    ) -> dict[str, EvalReport]:
        """Run multiple evaluations sequentially.

        Args:
            evaluations: List of (graph, collector, eval_set) tuples.
                         Each graph must be compiled with the matching collector's
                         callback_manager via make_trajectory_callback.
            config: Override configuration for all evaluations.
            verbose: Whether to log progress.

        Returns:
            Dictionary mapping eval_set_id to EvalReport.
        """
        cfg = config or self.default_config

        for graph, collector, eval_set in evaluations:
            evaluator = AgentEvaluator(graph, collector, cfg)
            report = await evaluator.evaluate(eval_set, verbose=verbose)
            self.results[report.eval_set_id] = report

        # --- Auto-invoke reporters for consolidated results ---
        self._run_reporters(cfg)

        return self.results

    def _run_reporters(self, config: EvalConfig) -> None:
        """Run reporters for each collected report."""
        try:
            from agentflow.qa.evaluation.reporters.manager import ReporterManager

            reporter_cfg = config.reporter
            if not reporter_cfg.enabled:
                return

            manager = ReporterManager(reporter_cfg)
            for report in self.results.values():
                manager.run_all(report)
        except Exception as exc:
            logger.error("Reporter pipeline failed in EvaluationRunner: %s", exc)

    @property
    def summary(self) -> dict[str, Any]:
        """Aggregate summary across all evaluations."""
        if not self.results:
            return {"total_evaluations": 0}

        total_cases = sum(r.summary.total_cases for r in self.results.values())
        passed_cases = sum(r.summary.passed_cases for r in self.results.values())

        return {
            "total_evaluations": len(self.results),
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "overall_pass_rate": passed_cases / total_cases if total_cases > 0 else 0.0,
        }
