"""
Structured Logging Configuration using structlog.

This module configures structured logging to show the Agent's "Thought" process
in the console.

"""

import logging
import sys
from typing import Any, Dict, Literal, Optional

import structlog
from structlog.typing import Processor

from src.config.settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[Literal["json", "text"]] = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Override log format (json for production, text for development)
    """
    settings = get_settings()
    level = log_level or settings.app.log_level
    fmt = log_format or settings.app.log_format

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors for all output formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        # JSON format for production - machine readable
        processors: list[Processor] = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        # Configure standard logging to use structlog formatting
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    else:
        # Text format for development - human readable with colors
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to integrate with structlog
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(numeric_level)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("anthropic").setLevel(logging.ERROR)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# =============================================================================
# Specialized Logging Functions for Agent Operations
# =============================================================================

class AgentLogger:
    """
    Specialized logger for agent operations.
    """

    def __init__(self, job_id: str, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.job_id = job_id
        self.logger = logger or get_logger("aira.agent")
        self._step = 0

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        """Internal logging method."""
        self._step += 1
        log_method = getattr(self.logger, level)
        log_method(
            event,
            job_id=self.job_id,
            step=self._step,
            **kwargs,
        )

    def thought(
        self,
        thought_type: str,
        content: str,
        **context: Any,
    ) -> None:
        """
        Log an agent thought/reasoning step.

        Args:
            thought_type: Type of thought (planning, analysis, decision)
            content: The actual thought content
            **context: Additional context
        """
        self._log(
            "info",
            "agent_thought",
            thought_type=thought_type,
            thought=content,
            **context,
        )

    def planning(
        self,
        plan: list[str],
        reasoning: Optional[str] = None,
    ) -> None:
        """
        Log agent planning step.

        Args:
            plan: List of planned tool executions
            reasoning: Reasoning for the plan
        """
        self._log(
            "info",
            "agent_planning",
            thought_type="planning",
            plan=plan,
            reasoning=reasoning,
        )

    def tool_start(
        self,
        tool_name: str,
        input_params: Dict[str, Any],
    ) -> None:
        """
        Log the start of a tool execution.

        Args:
            tool_name: Name of the tool being executed
            input_params: Input parameters to the tool
        """
        self._log(
            "info",
            "tool_execution_start",
            tool=tool_name,
            input=input_params,
        )

    def tool_complete(
        self,
        tool_name: str,
        duration_ms: int,
        result_summary: Optional[str] = None,
        result_count: Optional[int] = None,
    ) -> None:
        """
        Log successful tool completion.

        Args:
            tool_name: Name of the tool
            duration_ms: Execution time in milliseconds
            result_summary: Brief summary of results
            result_count: Number of results (if applicable)
        """
        self._log(
            "info",
            "tool_execution_complete",
            tool=tool_name,
            duration_ms=duration_ms,
            result_summary=result_summary,
            result_count=result_count,
        )

    def tool_error(
        self,
        tool_name: str,
        error: str,
        duration_ms: Optional[int] = None,
    ) -> None:
        """
        Log tool execution error.

        Args:
            tool_name: Name of the tool
            error: Error message
            duration_ms: Execution time before error
        """
        self._log(
            "error",
            "tool_execution_error",
            tool=tool_name,
            error=error,
            duration_ms=duration_ms,
        )

    def reflection(
        self,
        triggered: bool,
        reason: Optional[str] = None,
        action: Optional[str] = None,
        cycle: int = 1,
    ) -> None:
        """
        Log reflection cycle.

        Args:
            triggered: Whether reflection resulted in re-planning
            reason: Reason for the reflection decision
            action: Action to take (if re-planning)
            cycle: Current reflection cycle number
        """
        level = "warning" if triggered else "info"
        self._log(
            level,
            "reflection",
            triggered=triggered,
            reason=reason,
            action=action,
            reflection_cycle=cycle,
        )

    def synthesis_start(self) -> None:
        """Log start of report synthesis."""
        self._log("info", "synthesis_start", thought_type="synthesis")

    def synthesis_complete(
        self,
        summary_length: int,
        findings_count: int,
        sentiment_score: float,
    ) -> None:
        """
        Log completion of report synthesis.

        Args:
            summary_length: Length of the analysis summary
            findings_count: Number of key findings
            sentiment_score: Calculated sentiment score
        """
        self._log(
            "info",
            "synthesis_complete",
            thought_type="synthesis",
            summary_length=summary_length,
            findings_count=findings_count,
            sentiment_score=sentiment_score,
        )

    def iteration(
        self,
        current: int,
        max_iterations: int,
        status: str,
    ) -> None:
        """
        Log iteration progress.

        Args:
            current: Current iteration number
            max_iterations: Maximum allowed iterations
            status: Current status description
        """
        self._log(
            "info",
            "iteration",
            iteration=current,
            max_iterations=max_iterations,
            status=status,
        )

    def analysis_complete(
        self,
        ticker: str,
        duration_ms: int,
        tools_used: list[str],
        reflection_triggered: bool,
    ) -> None:
        """
        Log analysis completion.

        Args:
            ticker: Company ticker analyzed
            duration_ms: Total analysis duration
            tools_used: List of tools used
            reflection_triggered: Whether reflection was triggered
        """
        self._log(
            "info",
            "analysis_complete",
            ticker=ticker,
            duration_ms=duration_ms,
            tools_used=tools_used,
            reflection_triggered=reflection_triggered,
        )

    def analysis_failed(
        self,
        error: str,
        duration_ms: Optional[int] = None,
    ) -> None:
        """
        Log analysis failure.

        Args:
            error: Error message
            duration_ms: Duration before failure
        """
        self._log(
            "error",
            "analysis_failed",
            error=error,
            duration_ms=duration_ms,
        )


def get_agent_logger(job_id: str) -> AgentLogger:
    """
    Get an AgentLogger instance for a specific job.

    Args:
        job_id: Unique job identifier

    Returns:
        AgentLogger configured for the job
    """
    return AgentLogger(job_id)
