"""
PostgreSQL Repository - Production storage implementation.

Uses asyncpg for async PostgreSQL operations with connection pooling.
Supports vector embeddings via pgvector extension.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import asyncpg

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


class PostgresRepository(StoragePort):
    """
    PostgreSQL implementation of the storage port.

    Uses asyncpg connection pooling for efficient async operations.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_connections: int = 2,
        max_connections: int = 10,
    ):
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._min_connections = min_connections
        self._max_connections = max_connections
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password,
                min_size=self._min_connections,
                max_size=self._max_connections,
            )
            logger.info(
                "postgres_pool_created",
                host=self._host,
                database=self._database,
            )
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("postgres_pool_closed")

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
        pool = await self._get_pool()
        record_id = str(uuid4())
        now = datetime.now(timezone.utc)

        await pool.execute(
            """
            INSERT INTO analyses (id, job_id, company_ticker, company_name, user_query,
                                  analysis_type, status, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, 'PENDING', $7, $7)
            """,
            record_id,
            job_id,
            ticker.upper(),
            company_name,
            query,
            analysis_type,
            now,
        )

        logger.debug("postgres_create_analysis", job_id=job_id, ticker=ticker)

        return AnalysisRecord(
            id=record_id,
            job_id=job_id,
            company_ticker=ticker.upper(),
            company_name=company_name,
            user_query=query,
            analysis_type=analysis_type,
            status="PENDING",
            created_at=now,
            updated_at=now,
        )

    async def get_analysis(self, job_id: str) -> Optional[AnalysisRecord]:
        """Get an analysis record by job ID."""
        pool = await self._get_pool()

        row = await pool.fetchrow(
            """
            SELECT id, job_id, company_ticker, company_name, user_query, analysis_type,
                   status, progress, report, error_message, tools_used, iteration_count,
                   reflection_triggered, created_at, updated_at, completed_at
            FROM analyses
            WHERE job_id = $1
            """,
            job_id,
        )

        if row is None:
            return None

        return self._row_to_analysis_record(row)

    async def update_analysis_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update the status of an analysis job."""
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)

        await pool.execute(
            """
            UPDATE analyses
            SET status = $2, progress = $3, error_message = $4, updated_at = $5
            WHERE job_id = $1
            """,
            job_id,
            status,
            progress,
            error_message,
            now,
        )

        logger.debug("postgres_update_status", job_id=job_id, status=status)

    async def complete_analysis(
        self,
        job_id: str,
        report: AnalysisReport,
        tools_used: List[str],
        iteration_count: int,
        reflection_triggered: bool,
    ) -> None:
        """Mark an analysis as completed with results."""
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)

        # Serialize report to JSON
        report_json = json.dumps(report.model_dump(mode="json"))

        await pool.execute(
            """
            UPDATE analyses
            SET status = 'COMPLETED', report = $2::jsonb, tools_used = $3,
                iteration_count = $4, reflection_triggered = $5,
                updated_at = $6, completed_at = $6
            WHERE job_id = $1
            """,
            job_id,
            report_json,
            tools_used,
            iteration_count,
            reflection_triggered,
            now,
        )

        logger.debug(
            "postgres_complete_analysis",
            job_id=job_id,
            tools_used=tools_used,
            iteration_count=iteration_count,
        )

    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the current status of a job."""
        pool = await self._get_pool()

        row = await pool.fetchrow(
            """
            SELECT job_id, status, progress, report, error_message,
                   created_at, updated_at, completed_at
            FROM analyses
            WHERE job_id = $1
            """,
            job_id,
        )

        if row is None:
            return None

        # Parse report if completed
        report = None
        if row["status"] == "COMPLETED" and row["report"]:
            report_data = row["report"]
            if isinstance(report_data, str):
                report_data = json.loads(report_data)
            report = AnalysisReport(**report_data)

        return JobStatus(
            job_id=row["job_id"],
            status=row["status"],
            progress=row["progress"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            result=report,
            error=row["error_message"],
        )

    async def list_analyses(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[AnalysisRecord]:
        """List analysis records with optional filters."""
        pool = await self._get_pool()

        # Build query with optional filters
        query = """
            SELECT id, job_id, company_ticker, company_name, user_query, analysis_type,
                   status, progress, report, error_message, tools_used, iteration_count,
                   reflection_triggered, created_at, updated_at, completed_at
            FROM analyses
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if ticker:
            query += f" AND company_ticker = ${param_idx}"
            params.append(ticker.upper())
            param_idx += 1

        if status:
            query += f" AND status = ${param_idx}"
            params.append(status)
            param_idx += 1

        query += f" ORDER BY created_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])

        rows = await pool.fetch(query, *params)

        return [self._row_to_analysis_record(row) for row in rows]

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
        pool = await self._get_pool()
        record_id = str(uuid4())
        now = datetime.now(timezone.utc)

        # First get the analysis UUID from job_id
        analysis_uuid = await self._get_analysis_uuid(analysis_id)

        await pool.execute(
            """
            INSERT INTO tool_executions (id, analysis_id, tool_name, input_params,
                                         output_result, execution_time_ms, success,
                                         error_message, created_at)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7, $8, $9)
            """,
            record_id,
            analysis_uuid,
            tool_name,
            json.dumps(input_params),
            json.dumps(output_result) if output_result else None,
            execution_time_ms,
            success,
            error_message,
            now,
        )

        return ToolExecutionRecord(
            id=record_id,
            analysis_id=analysis_id,
            tool_name=tool_name,
            input_params=input_params,
            output_result=output_result,
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error_message,
            created_at=now,
        )

    async def get_tool_executions(
        self,
        analysis_id: str,
    ) -> List[ToolExecutionRecord]:
        """Get all tool executions for an analysis."""
        pool = await self._get_pool()

        analysis_uuid = await self._get_analysis_uuid(analysis_id)
        if analysis_uuid is None:
            return []

        rows = await pool.fetch(
            """
            SELECT id, tool_name, input_params, output_result, execution_time_ms,
                   success, error_message, created_at
            FROM tool_executions
            WHERE analysis_id = $1
            ORDER BY created_at
            """,
            analysis_uuid,
        )

        return [
            ToolExecutionRecord(
                id=str(row["id"]),
                analysis_id=analysis_id,
                tool_name=row["tool_name"],
                input_params=row["input_params"] or {},
                output_result=row["output_result"],
                execution_time_ms=row["execution_time_ms"],
                success=row["success"],
                error_message=row["error_message"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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
        pool = await self._get_pool()
        record_id = str(uuid4())
        now = datetime.now(timezone.utc)

        analysis_uuid = await self._get_analysis_uuid(analysis_id)

        await pool.execute(
            """
            INSERT INTO agent_thoughts (id, analysis_id, step_number, thought_type,
                                        thought_content, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            record_id,
            analysis_uuid,
            step_number,
            thought_type,
            thought_content,
            now,
        )

        return AgentThoughtRecord(
            id=record_id,
            analysis_id=analysis_id,
            step_number=step_number,
            thought_type=thought_type,
            thought_content=thought_content,
            created_at=now,
        )

    async def get_agent_thoughts(
        self,
        analysis_id: str,
    ) -> List[AgentThoughtRecord]:
        """Get all agent thoughts for an analysis."""
        pool = await self._get_pool()

        analysis_uuid = await self._get_analysis_uuid(analysis_id)
        if analysis_uuid is None:
            return []

        rows = await pool.fetch(
            """
            SELECT id, step_number, thought_type, thought_content, created_at
            FROM agent_thoughts
            WHERE analysis_id = $1
            ORDER BY step_number
            """,
            analysis_uuid,
        )

        return [
            AgentThoughtRecord(
                id=str(row["id"]),
                analysis_id=analysis_id,
                step_number=row["step_number"],
                thought_type=row["thought_type"],
                thought_content=row["thought_content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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
        pool = await self._get_pool()
        ticker_upper = ticker.upper()
        record_id = str(uuid4())
        now = datetime.now(timezone.utc)

        await pool.execute(
            """
            INSERT INTO monitoring_schedules (id, ticker, company_name, interval_hours,
                                              is_active, created_at, updated_at)
            VALUES ($1, $2, $3, $4, TRUE, $5, $5)
            ON CONFLICT (ticker) DO UPDATE SET
                company_name = EXCLUDED.company_name,
                interval_hours = EXCLUDED.interval_hours,
                is_active = TRUE,
                updated_at = EXCLUDED.updated_at
            """,
            record_id,
            ticker_upper,
            company_name,
            interval_hours,
            now,
        )

        logger.debug(
            "postgres_create_monitoring",
            ticker=ticker_upper,
            interval_hours=interval_hours,
        )

        return MonitoringSchedule(
            id=record_id,
            ticker=ticker_upper,
            company_name=company_name,
            interval_hours=interval_hours,
            is_active=True,
            created_at=now,
            updated_at=now,
        )

    async def get_monitoring_schedule(
        self,
        ticker: str,
    ) -> Optional[MonitoringSchedule]:
        """Get monitoring schedule for a ticker."""
        pool = await self._get_pool()

        row = await pool.fetchrow(
            """
            SELECT id, ticker, company_name, interval_hours, last_check_at,
                   last_analysis_id, article_hashes, is_active, created_at, updated_at
            FROM monitoring_schedules
            WHERE ticker = $1
            """,
            ticker.upper(),
        )

        if row is None:
            return None

        return self._row_to_monitoring_schedule(row)

    async def update_monitoring_schedule(
        self,
        ticker: str,
        last_check_at: Optional[datetime] = None,
        last_analysis_id: Optional[str] = None,
        article_hashes: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
    ) -> None:
        """Update a monitoring schedule."""
        pool = await self._get_pool()
        ticker_upper = ticker.upper()
        now = datetime.now(timezone.utc)

        # Build dynamic update
        updates = ["updated_at = $2"]
        params = [ticker_upper, now]
        param_idx = 3

        if last_check_at is not None:
            updates.append(f"last_check_at = ${param_idx}")
            params.append(last_check_at)
            param_idx += 1

        if last_analysis_id is not None:
            # Get analysis UUID
            analysis_uuid = await self._get_analysis_uuid(last_analysis_id)
            updates.append(f"last_analysis_id = ${param_idx}")
            params.append(analysis_uuid)
            param_idx += 1

        if article_hashes is not None:
            updates.append(f"article_hashes = ${param_idx}")
            params.append(article_hashes)
            param_idx += 1

        if is_active is not None:
            updates.append(f"is_active = ${param_idx}")
            params.append(is_active)
            param_idx += 1

        query = f"UPDATE monitoring_schedules SET {', '.join(updates)} WHERE ticker = $1"
        await pool.execute(query, *params)

    async def get_active_monitoring_schedules(self) -> List[MonitoringSchedule]:
        """Get all active monitoring schedules."""
        pool = await self._get_pool()

        rows = await pool.fetch(
            """
            SELECT id, ticker, company_name, interval_hours, last_check_at,
                   last_analysis_id, article_hashes, is_active, created_at, updated_at
            FROM monitoring_schedules
            WHERE is_active = TRUE
            """
        )

        return [self._row_to_monitoring_schedule(row) for row in rows]

    async def delete_monitoring_schedule(self, ticker: str) -> bool:
        """Delete a monitoring schedule."""
        pool = await self._get_pool()

        result = await pool.execute(
            "DELETE FROM monitoring_schedules WHERE ticker = $1",
            ticker.upper(),
        )

        # result is like "DELETE 1" or "DELETE 0"
        deleted = result.split()[-1] != "0"
        return deleted

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
        pool = await self._get_pool()
        record_id = str(uuid4())
        now = datetime.now(timezone.utc)

        analysis_uuid = await self._get_analysis_uuid(analysis_id)

        # Convert embedding list to PostgreSQL vector format
        embedding_str = f"[{','.join(str(f) for f in embedding)}]"

        await pool.execute(
            """
            INSERT INTO analysis_embeddings (id, analysis_id, content_type, content_text,
                                             embedding, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb, $7)
            """,
            record_id,
            analysis_uuid,
            content_type,
            content_text,
            embedding_str,
            json.dumps(metadata) if metadata else None,
            now,
        )

        logger.debug(
            "postgres_store_embedding",
            analysis_id=analysis_id,
            content_type=content_type,
        )

        return record_id

    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 5,
        ticker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings using vector similarity."""
        pool = await self._get_pool()

        embedding_str = f"[{','.join(str(f) for f in embedding)}]"

        if ticker:
            rows = await pool.fetch(
                """
                SELECT e.id, e.analysis_id, e.content_type, e.content_text, e.metadata,
                       1 - (e.embedding <=> $1::vector) AS score
                FROM analysis_embeddings e
                JOIN analyses a ON e.analysis_id = a.id
                WHERE a.company_ticker = $3
                ORDER BY e.embedding <=> $1::vector
                LIMIT $2
                """,
                embedding_str,
                limit,
                ticker.upper(),
            )
        else:
            rows = await pool.fetch(
                """
                SELECT id, analysis_id, content_type, content_text, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM analysis_embeddings
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                embedding_str,
                limit,
            )

        results = []
        for row in rows:
            metadata = row["metadata"]
            # Handle case where metadata is returned as JSON string
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append({
                "id": str(row["id"]),
                "analysis_id": str(row["analysis_id"]),
                "content_type": row["content_type"],
                "content_text": row["content_text"],
                "score": float(row["score"]),
                "metadata": metadata or {},
            })
        return results

    async def get_sentiment_history(
        self,
        ticker: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get sentiment score history for a ticker."""
        pool = await self._get_pool()

        rows = await pool.fetch(
            """
            SELECT
                job_id,
                (report->>'sentiment_score')::float as sentiment_score,
                completed_at
            FROM analyses
            WHERE company_ticker = $1
              AND status = 'COMPLETED'
              AND report IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT $2
            """,
            ticker.upper(),
            limit,
        )

        return [
            {
                "job_id": row["job_id"],
                "sentiment_score": row["sentiment_score"],
                "date": row["completed_at"].isoformat() if row["completed_at"] else None,
            }
            for row in rows
        ]

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        try:
            pool = await self._get_pool()
            await pool.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error("postgres_health_check_failed", error=str(e))
            return False

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_analysis_uuid(self, job_id: str) -> Optional[str]:
        """Get the UUID for an analysis by job_id."""
        pool = await self._get_pool()
        result = await pool.fetchval(
            "SELECT id FROM analyses WHERE job_id = $1",
            job_id,
        )
        return str(result) if result else None

    def _row_to_analysis_record(self, row: asyncpg.Record) -> AnalysisRecord:
        """Convert a database row to AnalysisRecord."""
        report_data = None
        if row["report"]:
            report_data = row["report"]
            if isinstance(report_data, str):
                report_data = json.loads(report_data)

        return AnalysisRecord(
            id=str(row["id"]),
            job_id=row["job_id"],
            company_ticker=row["company_ticker"],
            company_name=row["company_name"],
            user_query=row["user_query"],
            analysis_type=row["analysis_type"],
            status=row["status"],
            progress=row["progress"],
            report=report_data,
            error_message=row["error_message"],
            tools_used=row["tools_used"] or [],
            iteration_count=row["iteration_count"] or 0,
            reflection_triggered=row["reflection_triggered"] or False,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def _row_to_monitoring_schedule(self, row: asyncpg.Record) -> MonitoringSchedule:
        """Convert a database row to MonitoringSchedule."""
        return MonitoringSchedule(
            id=str(row["id"]),
            ticker=row["ticker"],
            company_name=row["company_name"],
            interval_hours=row["interval_hours"],
            last_check_at=row["last_check_at"],
            last_analysis_id=str(row["last_analysis_id"]) if row["last_analysis_id"] else None,
            article_hashes=row["article_hashes"] or [],
            is_active=row["is_active"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
