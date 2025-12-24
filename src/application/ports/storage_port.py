"""
Storage Port - Abstract interface for data persistence.

This port defines the contract for storage operations.
Supports both structured data and vector embeddings.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.domain.models import (
    AnalysisRecord,
    AnalysisReport,
    AgentThoughtRecord,
    JobStatus,
    MonitoringSchedule,
    ToolExecutionRecord,
)


class StoragePort(ABC):
    """
    Abstract interface for data storage.

    Implementations:
    - PostgresRepository: PostgreSQL + pgvector
    - MemoryRepository: In-memory storage for testing
    """

    # =========================================================================
    # Analysis Job Operations
    # =========================================================================

    @abstractmethod
    async def create_analysis(
        self,
        job_id: str,
        ticker: str,
        query: str,
        company_name: Optional[str] = None,
        analysis_type: str = "ON_DEMAND",
    ) -> AnalysisRecord:
        """
        Create a new analysis job record.

        Args:
            job_id: Unique job identifier
            ticker: Stock ticker symbol
            query: User's analysis query
            company_name: Company name (optional)
            analysis_type: Type of analysis (ON_DEMAND or PROACTIVE_ALERT)

        Returns:
            Created AnalysisRecord
        """
        pass

    @abstractmethod
    async def get_analysis(self, job_id: str) -> Optional[AnalysisRecord]:
        """
        Get an analysis record by job ID.

        Args:
            job_id: Unique job identifier

        Returns:
            AnalysisRecord or None if not found
        """
        pass

    @abstractmethod
    async def update_analysis_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update the status of an analysis job.

        Args:
            job_id: Unique job identifier
            status: New status (PENDING, RUNNING, COMPLETED, FAILED)
            progress: Human-readable progress message
            error_message: Error message if failed
        """
        pass

    @abstractmethod
    async def complete_analysis(
        self,
        job_id: str,
        report: AnalysisReport,
        tools_used: List[str],
        iteration_count: int,
        reflection_triggered: bool,
    ) -> None:
        """
        Mark an analysis as completed with results.

        Args:
            job_id: Unique job identifier
            report: The final AnalysisReport
            tools_used: List of tools that were used
            iteration_count: Number of iterations
            reflection_triggered: Whether reflection was triggered
        """
        pass

    @abstractmethod
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get the current status of a job.

        Args:
            job_id: Unique job identifier

        Returns:
            JobStatus or None if not found
        """
        pass

    @abstractmethod
    async def list_analyses(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[AnalysisRecord]:
        """
        List analysis records with optional filters.

        Args:
            ticker: Filter by ticker
            status: Filter by status
            limit: Maximum records to return
            offset: Number of records to skip

        Returns:
            List of AnalysisRecord objects
        """
        pass

    # =========================================================================
    # Tool Execution Logging
    # =========================================================================

    @abstractmethod
    async def log_tool_execution(
        self,
        analysis_id: str,
        tool_name: str,
        input_params: Dict[str, Any],
        output_result: Optional[Dict[str, Any]],
        execution_time_ms: int,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> ToolExecutionRecord:
        """
        Log a tool execution for debugging.

        Args:
            analysis_id: Parent analysis job ID
            tool_name: Name of the tool
            input_params: Tool input parameters
            output_result: Tool output (if successful)
            execution_time_ms: Execution duration
            success: Whether execution succeeded
            error_message: Error message if failed

        Returns:
            Created ToolExecutionRecord
        """
        pass

    @abstractmethod
    async def get_tool_executions(
        self,
        analysis_id: str,
    ) -> List[ToolExecutionRecord]:
        """
        Get all tool executions for an analysis.

        Args:
            analysis_id: Parent analysis job ID

        Returns:
            List of ToolExecutionRecord objects
        """
        pass

    # =========================================================================
    # Agent Thought Logging
    # =========================================================================

    @abstractmethod
    async def log_agent_thought(
        self,
        analysis_id: str,
        step_number: int,
        thought_type: str,
        thought_content: str,
    ) -> AgentThoughtRecord:
        """
        Log an agent thought for transparency.

        Args:
            analysis_id: Parent analysis job ID
            step_number: Step number in the execution
            thought_type: Type of thought (planning, reflection, etc.)
            thought_content: The actual thought content

        Returns:
            Created AgentThoughtRecord
        """
        pass

    @abstractmethod
    async def get_agent_thoughts(
        self,
        analysis_id: str,
    ) -> List[AgentThoughtRecord]:
        """
        Get all agent thoughts for an analysis.

        Args:
            analysis_id: Parent analysis job ID

        Returns:
            List of AgentThoughtRecord objects
        """
        pass

    # =========================================================================
    # Monitoring Schedule Operations
    # =========================================================================

    @abstractmethod
    async def create_monitoring_schedule(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        interval_hours: int = 24,
    ) -> MonitoringSchedule:
        """
        Create a new monitoring schedule.

        Args:
            ticker: Stock ticker to monitor
            company_name: Company name
            interval_hours: Check interval in hours

        Returns:
            Created MonitoringSchedule
        """
        pass

    @abstractmethod
    async def get_monitoring_schedule(
        self,
        ticker: str,
    ) -> Optional[MonitoringSchedule]:
        """
        Get monitoring schedule for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            MonitoringSchedule or None if not found
        """
        pass

    @abstractmethod
    async def update_monitoring_schedule(
        self,
        ticker: str,
        last_check_at: Optional[datetime] = None,
        last_analysis_id: Optional[str] = None,
        article_hashes: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
    ) -> None:
        """
        Update a monitoring schedule.

        Args:
            ticker: Stock ticker symbol
            last_check_at: Last check timestamp
            last_analysis_id: Last analysis job ID
            article_hashes: Article hashes for deduplication
            is_active: Whether monitoring is active
        """
        pass

    @abstractmethod
    async def get_active_monitoring_schedules(self) -> List[MonitoringSchedule]:
        """
        Get all active monitoring schedules.

        Returns:
            List of active MonitoringSchedule objects
        """
        pass

    @abstractmethod
    async def delete_monitoring_schedule(self, ticker: str) -> bool:
        """
        Delete a monitoring schedule.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if deleted, False if not found
        """
        pass

    # =========================================================================
    # Vector Embedding Operations
    # =========================================================================

    @abstractmethod
    async def store_embedding(
        self,
        analysis_id: str,
        content_type: str,
        content_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a vector embedding.

        Args:
            analysis_id: Parent analysis job ID
            content_type: Type of content (summary, key_finding, etc.)
            content_text: Original text content
            embedding: Vector embedding
            metadata: Additional metadata

        Returns:
            Embedding record ID
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 5,
        ticker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings using vector similarity.

        Args:
            embedding: Query embedding vector
            limit: Maximum results to return
            ticker: Optional filter by ticker

        Returns:
            List of similar records with scores
        """
        pass

    @abstractmethod
    async def get_sentiment_history(
        self,
        ticker: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get sentiment score history for a ticker.

        Retrieves past sentiment scores for trend analysis.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of records to return

        Returns:
            List of dicts with {job_id, sentiment_score, date}
            sorted by date descending (most recent first)
        """
        pass

    # =========================================================================
    # Health Check
    # =========================================================================

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if storage is healthy.

        Returns:
            True if storage is available
        """
        pass
