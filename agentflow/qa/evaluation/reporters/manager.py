"""
Reporter manager / orchestrator.

Coordinates all enabled reporters so that a single call after evaluation
produces console output **and** persisted JSON / HTML / JUnit-XML files
in the correct output directory with timestamped filenames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agentflow.qa.evaluation.reporters._utils import format_timestamp


if TYPE_CHECKING:
    from agentflow.qa.evaluation.config.eval_config import ReporterConfig
    from agentflow.qa.evaluation.eval_result import EvalReport

logger = logging.getLogger("agentflow.evaluation.reporters")


@dataclass
class ReporterOutput:
    """Result of running all reporters.

    Attributes:
        json_path: Path to the generated JSON file (or None).
        html_path: Path to the generated HTML file (or None).
        junit_path: Path to the generated JUnit XML file (or None).
        console_output: Whether console reporter ran successfully.
        errors: List of (reporter_name, error_message) pairs.
    """

    json_path: str | None = None
    html_path: str | None = None
    junit_path: str | None = None
    console_output: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def generated_files(self) -> list[str]:
        """Return list of all successfully generated file paths."""
        return [p for p in (self.json_path, self.html_path, self.junit_path) if p]


class ReporterManager:
    """Orchestrates all enabled reporters for an evaluation report.

    Given a ``ReporterConfig``, creates the output directory and invokes
    each enabled reporter in sequence.  Failures in one reporter do **not**
    prevent others from running — errors are collected and returned in
    ``ReporterOutput``.

    Example:
        ```python
        from agentflow.qa.evaluation.config.eval_config import ReporterConfig
        from agentflow.qa.evaluation.reporters.manager import ReporterManager

        manager = ReporterManager(ReporterConfig())
        output = manager.run_all(report)
        print(output.generated_files)
        ```
    """

    def __init__(self, config: ReporterConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self,
        report: EvalReport,
        output_dir: str | None = None,
    ) -> ReporterOutput:
        """Run every enabled reporter and return a summary.

        Args:
            report: The evaluation report to render.
            output_dir: Override the output directory from config.

        Returns:
            ``ReporterOutput`` with paths and error info.
        """
        if not self.config.enabled:
            logger.debug("Reporters disabled via config; skipping.")
            return ReporterOutput()

        out_dir = output_dir or self.config.output_dir

        # Resolve relative output_dir relative to the current working
        # directory so that reports land in the user's project tree
        # regardless of how agentflow was installed.
        if not Path(out_dir).is_absolute():
            out_dir = str(Path.cwd() / out_dir)

        result = ReporterOutput()

        # Ensure output directory exists (even if only console is enabled,
        # so the directory is ready if the user calls save() manually later).
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create output directory '%s': %s", out_dir, exc)
            result.errors.append(("directory", str(exc)))
            # Fall through — console can still work.

        # Build the filename stem: {eval_set_id}_{timestamp} or just {eval_set_id}
        stem = self._build_stem(report)

        # 1. Console
        if self.config.console:
            result.console_output = self._run_console(report, result)

        # 2. JSON
        if self.config.json_report:
            result.json_path = self._run_json(report, out_dir, stem, result)

        # 3. HTML
        if self.config.html:
            result.html_path = self._run_html(report, out_dir, stem, result)

        # 4. JUnit XML
        if self.config.junit_xml:
            result.junit_path = self._run_junit(report, out_dir, stem, result)

        # Log summary
        self._log_summary(result)

        return result

    # ------------------------------------------------------------------
    # Individual reporter runners (each catches its own errors)
    # ------------------------------------------------------------------

    def _run_console(self, report: EvalReport, output: ReporterOutput) -> bool:
        try:
            from agentflow.qa.evaluation.reporters.console import ConsoleReporter

            reporter = ConsoleReporter(
                use_color=True,
                verbose=self.config.verbose,
                include_trajectory=self.config.include_trajectory,
                include_actual_response=getattr(self.config, "include_actual_response", True),
            )
            reporter.report(report)
            return True
        except Exception as exc:
            logger.error("ConsoleReporter failed: %s", exc)
            output.errors.append(("ConsoleReporter", str(exc)))
            return False

    def _run_json(
        self,
        report: EvalReport,
        out_dir: str,
        stem: str,
        output: ReporterOutput,
    ) -> str | None:
        try:
            from agentflow.qa.evaluation.reporters.json import JSONReporter

            reporter = JSONReporter(
                indent=2,
                include_details=self.config.include_details,
                include_trajectory=self.config.include_trajectory,
                include_node_responses=getattr(self.config, "include_node_responses", True),
                include_actual_response=getattr(self.config, "include_actual_response", True),
                include_tool_call_details=getattr(self.config, "include_tool_call_details", True),
            )
            path = str(Path(out_dir) / f"{stem}.json")
            reporter.save(report, path)
            logger.info("JSON report saved: %s", path)
            return path
        except Exception as exc:
            logger.error("JSONReporter failed: %s", exc)
            output.errors.append(("JSONReporter", str(exc)))
            return None

    def _run_html(
        self,
        report: EvalReport,
        out_dir: str,
        stem: str,
        output: ReporterOutput,
    ) -> str | None:
        try:
            from agentflow.qa.evaluation.reporters.html import HTMLReporter

            reporter = HTMLReporter(
                include_details=self.config.include_details,
                include_actual_response=getattr(self.config, "include_actual_response", True),
                include_tool_call_details=getattr(self.config, "include_tool_call_details", True),
                include_node_responses=getattr(self.config, "include_node_responses", True),
                include_trajectory=self.config.include_trajectory,
            )
            path = str(Path(out_dir) / f"{stem}.html")
            reporter.save(report, path)
            logger.info("HTML report saved: %s", path)
            return path
        except Exception as exc:
            logger.error("HTMLReporter failed: %s", exc)
            output.errors.append(("HTMLReporter", str(exc)))
            return None

    def _run_junit(
        self,
        report: EvalReport,
        out_dir: str,
        stem: str,
        output: ReporterOutput,
    ) -> str | None:
        try:
            from agentflow.qa.evaluation.reporters.json import JUnitXMLReporter

            reporter = JUnitXMLReporter()
            path = str(Path(out_dir) / f"{stem}_junit.xml")
            reporter.save(report, path)
            logger.info("JUnit XML report saved: %s", path)
            return path
        except Exception as exc:
            logger.error("JUnitXMLReporter failed: %s", exc)
            output.errors.append(("JUnitXMLReporter", str(exc)))
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_stem(self, report: EvalReport) -> str:
        """Build a filename stem from the report metadata."""
        raw_id = report.eval_set_id or "eval"
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_id)

        if self.config.timestamp_files:
            ts = format_timestamp(report.timestamp, fmt="%Y%m%d_%H%M%S")
            return f"{safe_id}_{ts}"
        return safe_id

    def _log_summary(self, output: ReporterOutput) -> None:
        files = output.generated_files
        if files:
            logger.info(
                "Reporter output — %d file(s) generated: %s",
                len(files),
                ", ".join(files),
            )
        if output.has_errors:
            for name, err in output.errors:
                logger.warning("Reporter error [%s]: %s", name, err)
