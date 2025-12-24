"""
In-Memory Repository - For testing without database.

Provides a complete implementation of StoragePort using
in-memory data structures. Useful for unit testing and
development without PostgreSQL.
"""

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.application.ports.storage_port import StoragePort
from src.config.logging import get_logger
from src.domain.models import (
    AnalysisRecord,
    AnalysisReport,
    AgentThoughtRecord,
    JobStatus,
    MonitoringSchedule,
    ToolExecutionRecord,
)

logger = get_logger(__name__)


class MemoryRepository(StoragePort):
    """
    In-memory implementation of the storage port.

    All data is stored in dictionaries and lists.
    Data is lost when the application restarts.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._analyses: Dict[str, AnalysisRecord] = {}
        self._tool_executions: Dict[str, List[ToolExecutionRecord]] = {}
        self._agent_thoughts: Dict[str, List[AgentThoughtRecord]] = {}
        self._monitoring_schedules: Dict[str, MonitoringSchedule] = {}
        self._embeddings: List[Dict[str, Any]] = []

    # =========================================================================
    # Analysis Job Operations
    # =========================================================================

    async def create_analysis(
        self,
        job_id: str,
        ticker: str,
        query: str,
        company_name: Optional[str] = None,
        analysis_type: str = "ON_DEMAND",
    ) -> AnalysisRecord:
        """Create a new analysis job record."""
        now = datetime.now(timezone.utc)

        record = AnalysisRecord(
            id=str(uuid4()),
            job_id=job_id,
            company_ticker=ticker,
            company_name=company_name,
            user_query=query,
            analysis_type=analysis_type,
            status="PENDING",
            created_at=now,
            updated_at=now,
        )

        async with self._lock:
            self._analyses[job_id] = record
            self._tool_executions[job_id] = []
            self._agent_thoughts[job_id] = []

        logger.debug("memory_repo_create_analysis", job_id=job_id, ticker=ticker)

        return record

    async def get_analysis(self, job_id: str) -> Optional[AnalysisRecord]:
        """Get an analysis record by job ID."""
        async with self._lock:
            return self._analyses.get(job_id)

    async def update_analysis_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update the status of an analysis job."""
        async with self._lock:
            if job_id not in self._analyses:
                logger.warning("memory_repo_update_not_found", job_id=job_id)
                return

            record = self._analyses[job_id]

            update_dict = {
                "status": status,
                "updated_at": datetime.now(timezone.utc),
            }

            if progress is not None:
                update_dict["progress"] = progress
            if error_message is not None:
                update_dict["error_message"] = error_message

            self._analyses[job_id] = record.model_copy(update=update_dict)

        logger.debug("memory_repo_update_status", job_id=job_id, status=status, progress=progress)

    async def complete_analysis(
        self,
        job_id: str,
        report: Any,  # Accept AnalysisReport or dict
        tools_used: List[str],
        iteration_count: int,
        reflection_triggered: bool,
    ) -> None:
        """Mark an analysis as completed with results."""
        async with self._lock:
            if job_id not in self._analyses:
                logger.warning("memory_repo_complete_not_found", job_id=job_id)
                return

            record = self._analyses[job_id]
            now = datetime.now(timezone.utc)

            # Handle both AnalysisReport objects and dicts
            if hasattr(report, "model_dump"):
                report_dict = report.model_dump()
            else:
                report_dict = report

            self._analyses[job_id] = record.model_copy(update={
                "status": "COMPLETED",
                "report": report_dict,
                "tools_used": tools_used,
                "iteration_count": iteration_count,
                "reflection_triggered": reflection_triggered,
                "updated_at": now,
                "completed_at": now,
            })

        logger.debug(
            "memory_repo_complete_analysis",
            job_id=job_id,
            tools_used=tools_used,
            iteration_count=iteration_count,
        )

    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the current status of a job."""
        async with self._lock:
            record = self._analyses.get(job_id)
            if not record:
                return None

            # Build report if completed
            report = None
            if record.status == "COMPLETED" and record.report:
                report = AnalysisReport(**record.report)

            return JobStatus(
                job_id=record.job_id,
                status=record.status,
                progress=record.progress,
                created_at=record.created_at,
                updated_at=record.updated_at,
                completed_at=record.completed_at,
                result=report,
                error=record.error_message,
            )

    async def list_analyses(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[AnalysisRecord]:
        """List analysis records with optional filters."""
        async with self._lock:
            results = list(self._analyses.values())

            # Apply filters
            if ticker:
                results = [r for r in results if r.company_ticker == ticker.upper()]
            if status:
                results = [r for r in results if r.status == status]

            # Sort by created_at descending
            results.sort(key=lambda r: r.created_at, reverse=True)

            # Apply pagination
            return results[offset : offset + limit]

    # =========================================================================
    # Tool Execution Logging
    # =========================================================================

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
        """Log a tool execution for debugging."""
        record = ToolExecutionRecord(
            id=str(uuid4()),
            analysis_id=analysis_id,
            tool_name=tool_name,
            input_params=input_params,
            output_result=output_result,
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error_message,
            created_at=datetime.now(timezone.utc),
        )

        async with self._lock:
            if analysis_id not in self._tool_executions:
                self._tool_executions[analysis_id] = []
            self._tool_executions[analysis_id].append(record)

        logger.debug(
            "memory_repo_log_tool",
            analysis_id=analysis_id,
            tool_name=tool_name,
            success=success,
        )

        return record

    async def get_tool_executions(
        self,
        analysis_id: str,
    ) -> List[ToolExecutionRecord]:
        """Get all tool executions for an analysis."""
        async with self._lock:
            return self._tool_executions.get(analysis_id, [])

    # =========================================================================
    # Agent Thought Logging
    # =========================================================================

    async def log_agent_thought(
        self,
        analysis_id: str,
        step_number: int,
        thought_type: str,
        thought_content: str,
    ) -> AgentThoughtRecord:
        """Log an agent thought for transparency."""
        record = AgentThoughtRecord(
            id=str(uuid4()),
            analysis_id=analysis_id,
            step_number=step_number,
            thought_type=thought_type,
            thought_content=thought_content,
            created_at=datetime.now(timezone.utc),
        )

        async with self._lock:
            if analysis_id not in self._agent_thoughts:
                self._agent_thoughts[analysis_id] = []
            self._agent_thoughts[analysis_id].append(record)

        return record

    async def get_agent_thoughts(
        self,
        analysis_id: str,
    ) -> List[AgentThoughtRecord]:
        """Get all agent thoughts for an analysis."""
        async with self._lock:
            return self._agent_thoughts.get(analysis_id, [])

    # =========================================================================
    # Monitoring Schedule Operations
    # =========================================================================

    async def create_monitoring_schedule(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        interval_hours: int = 24,
    ) -> MonitoringSchedule:
        """Create a new monitoring schedule."""
        ticker_upper = ticker.upper()
        now = datetime.now(timezone.utc)

        schedule = MonitoringSchedule(
            id=str(uuid4()),
            ticker=ticker_upper,
            company_name=company_name,
            interval_hours=interval_hours,
            is_active=True,
            created_at=now,
            updated_at=now,
        )

        async with self._lock:
            self._monitoring_schedules[ticker_upper] = schedule

        logger.debug(
            "memory_repo_create_monitoring",
            ticker=ticker_upper,
            interval_hours=interval_hours,
        )

        return schedule

    async def get_monitoring_schedule(
        self,
        ticker: str,
    ) -> Optional[MonitoringSchedule]:
        """Get monitoring schedule for a ticker."""
        async with self._lock:
            return self._monitoring_schedules.get(ticker.upper())

    async def update_monitoring_schedule(
        self,
        ticker: str,
        last_check_at: Optional[datetime] = None,
        last_analysis_id: Optional[str] = None,
        article_hashes: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
    ) -> None:
        """Update a monitoring schedule."""
        ticker_upper = ticker.upper()

        async with self._lock:
            schedule = self._monitoring_schedules.get(ticker_upper)

            if not schedule:
                logger.warning("memory_repo_update_monitoring_not_found", ticker=ticker_upper)
                return

            updated_data = schedule.model_dump()
            updated_data["updated_at"] = datetime.now(timezone.utc)

            if last_check_at is not None:
                updated_data["last_check_at"] = last_check_at
            if last_analysis_id is not None:
                updated_data["last_analysis_id"] = last_analysis_id
            if article_hashes is not None:
                updated_data["article_hashes"] = article_hashes
            if is_active is not None:
                updated_data["is_active"] = is_active

            self._monitoring_schedules[ticker_upper] = MonitoringSchedule(**updated_data)

    async def get_active_monitoring_schedules(self) -> List[MonitoringSchedule]:
        """Get all active monitoring schedules."""
        async with self._lock:
            return [s for s in self._monitoring_schedules.values() if s.is_active]

    async def delete_monitoring_schedule(self, ticker: str) -> bool:
        """Delete a monitoring schedule."""
        ticker_upper = ticker.upper()
        async with self._lock:
            if ticker_upper in self._monitoring_schedules:
                del self._monitoring_schedules[ticker_upper]
                return True
            return False

    # =========================================================================
    # Vector Embedding Operations
    # =========================================================================

    async def store_embedding(
        self,
        analysis_id: str,
        content_type: str,
        content_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a vector embedding."""
        embedding_id = str(uuid4())

        record = {
            "id": embedding_id,
            "analysis_id": analysis_id,
            "content_type": content_type,
            "content_text": content_text,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
        }

        async with self._lock:
            self._embeddings.append(record)

        logger.debug(
            "memory_repo_store_embedding",
            analysis_id=analysis_id,
            content_type=content_type,
        )

        return embedding_id

    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 5,
        ticker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity."""

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(x * x for x in b))

            if magnitude_a == 0 or magnitude_b == 0:
                return 0.0

            return dot_product / (magnitude_a * magnitude_b)

        # Filter by ticker if provided
        if ticker:
            ticker_upper = ticker.upper()
            candidates = [
                e for e in self._embeddings
                if e.get("metadata", {}).get("ticker", "").upper() == ticker_upper
            ]
        else:
            candidates = self._embeddings

        # Calculate similarities
        scored = []
        for record in candidates:
            score = cosine_similarity(embedding, record["embedding"])
            scored.append({
                "id": record["id"],
                "analysis_id": record["analysis_id"],
                "content_type": record["content_type"],
                "content_text": record["content_text"],
                "score": score,
                "metadata": record.get("metadata", {}),
            })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        return scored[:limit]

    async def get_sentiment_history(
        self,
        ticker: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get sentiment score history for a ticker."""
        async with self._lock:
            results = []

            for record in self._analyses.values():
                if (
                    record.company_ticker.upper() == ticker.upper()
                    and record.status == "COMPLETED"
                    and record.report
                ):
                    # Extract sentiment score from report
                    sentiment_score = 0.0
                    if isinstance(record.report, dict):
                        sentiment_score = record.report.get("sentiment_score", 0.0)
                    elif hasattr(record.report, "sentiment_score"):
                        sentiment_score = record.report.sentiment_score

                    results.append({
                        "job_id": record.job_id,
                        "sentiment_score": sentiment_score,
                        "date": record.completed_at.isoformat() if record.completed_at else None,
                    })

            # Sort by date descending (most recent first)
            results.sort(key=lambda x: x["date"] or "", reverse=True)

            return results[:limit]

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> bool:
        """In-memory storage is always healthy."""
        return True

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear(self) -> None:
        """Clear all data (useful for testing)."""
        self._analyses.clear()
        self._tool_executions.clear()
        self._agent_thoughts.clear()
        self._monitoring_schedules.clear()
        self._embeddings.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        return {
            "analyses": len(self._analyses),
            "tool_executions": sum(len(v) for v in self._tool_executions.values()),
            "agent_thoughts": sum(len(v) for v in self._agent_thoughts.values()),
            "monitoring_schedules": len(self._monitoring_schedules),
            "embeddings": len(self._embeddings),
        }
