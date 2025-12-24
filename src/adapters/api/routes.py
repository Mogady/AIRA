"""
FastAPI Routes - REST API endpoints for A.I.R.A.

Endpoints:
- POST /analyze - Start a company analysis
- GET /status/{job_id} - Get analysis status and results
- POST /monitor_start - Start scheduled monitoring
- GET /health - Health check

All endpoints have detailed OpenAPI documentation.
"""

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from arq import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.adapters.api.dependencies import (
    get_arq_pool,
    get_embeddings_provider,
    get_financial_provider,
    get_llm_provider,
    get_news_provider,
    get_search_provider,
    get_storage,
)
from src.application.ports.embeddings_port import EmbeddingsPort
from src.application.ports.financial_port import FinancialPort
from src.application.ports.llm_port import LLMPort
from src.application.ports.news_port import NewsPort
from src.application.ports.search_port import SearchPort
from src.application.ports.storage_port import StoragePort
from src.config.logging import get_logger
from src.config.settings import get_settings
from src.domain.models import (
    AgentThoughtRecord,
    AnalysisRecord,
    AnalyzeRequest,
    AnalyzeResponse,
    JobStatus,
    MonitoringSchedule,
    MonitorStartRequest,
    MonitorStartResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    ToolExecutionRecord,
)

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Analysis Endpoints
# =============================================================================

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start Company Analysis",
    description="""
    Submit a company analysis request. The analysis runs asynchronously.

    **Process:**
    1. Submit your query (e.g., "Analyze the near-term prospects of Tesla, Inc. (TSLA)")
    2. Receive a `job_id` immediately
    3. Poll `/status/{job_id}` to get results when complete

    **Example queries:**
    - "Analyze the near-term prospects of Tesla, Inc. (TSLA)"
    - "Research Apple Inc. (AAPL) stock"
    - "What is the market sentiment for Microsoft (MSFT)?"

    The agent will:
    - Fetch recent news articles
    - Analyze sentiment from news coverage
    - Retrieve financial data
    - Generate a comprehensive analysis report
    """,
    responses={
        202: {
            "description": "Analysis job created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "abc123-def456",
                        "status": "PENDING",
                        "message": "Analysis job created successfully",
                        "status_url": "/status/abc123-def456"
                    }
                }
            }
        },
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"},
    },
)
async def create_analysis(
    request: AnalyzeRequest,
    storage: StoragePort = Depends(get_storage),
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> AnalyzeResponse:
    """
    Create a new analysis job.

    The analysis is enqueued to the Arq worker for background processing.
    Use the returned job_id to poll for status and results.
    """
    job_id = str(uuid.uuid4())

    logger.info(
        "analysis_request",
        job_id=job_id,
        query=request.query[:100],
    )

    # Extract ticker from query for storage
    ticker_match = re.search(r'\(([A-Z]{1,5})\)', request.query)
    if not ticker_match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract stock ticker from query. Please include ticker in parentheses, e.g., 'Analyze Tesla (TSLA)'",
        )
    ticker = ticker_match.group(1)

    # Create job record
    await storage.create_analysis(
        job_id=job_id,
        ticker=ticker,
        query=request.query,
    )

    # Enqueue analysis job to Arq worker
    await arq_pool.enqueue_job("run_analysis", job_id, request.query)

    return AnalyzeResponse(
        job_id=job_id,
        status="PENDING",
        message="Analysis job created successfully",
        status_url=f"/status/{job_id}",
    )


@router.get(
    "/status/{job_id}",
    response_model=JobStatus,
    summary="Get Analysis Status",
    description="""
    Get the current status of an analysis job.

    **Status values:**
    - `PENDING` - Job created, waiting to start
    - `RUNNING` - Analysis in progress
    - `COMPLETED` - Analysis complete, results available
    - `FAILED` - Analysis failed, error message available

    When status is `COMPLETED`, the `result` field contains the full analysis report.
    """,
    responses={
        200: {"description": "Job status retrieved"},
        404: {"description": "Job not found"},
    },
)
async def get_analysis_status(
    job_id: str,
    storage: StoragePort = Depends(get_storage),
) -> JobStatus:
    """Get the status and results of an analysis job."""
    job_status = await storage.get_job_status(job_id)

    if not job_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job not found: {job_id}",
        )

    return job_status


# =============================================================================
# Monitoring Endpoints
# =============================================================================

@router.post(
    "/monitor_start",
    response_model=MonitorStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start Stock Monitoring",
    description="""
    Start scheduled monitoring for a stock ticker.

    The system will check for new news articles at the specified interval
    (default: every 24 hours). If significant new coverage is detected
    (5+ new articles), an automatic analysis will be triggered.

    **Features:**
    - Automatic news monitoring
    - PROACTIVE_ALERT generation when significant news detected
    - Configurable check interval
    """,
    responses={
        201: {"description": "Monitoring schedule created"},
        400: {"description": "Invalid request"},
        409: {"description": "Monitoring already exists for this ticker"},
    },
)
async def start_monitoring(
    request: MonitorStartRequest,
    storage: StoragePort = Depends(get_storage),
) -> MonitorStartResponse:
    """Start scheduled monitoring for a stock."""
    ticker = request.ticker.upper()

    # Check if monitoring already exists
    existing = await storage.get_monitoring_schedule(ticker)
    if existing and existing.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Monitoring already active for ticker: {ticker}",
        )

    # Create monitoring schedule
    schedule = await storage.create_monitoring_schedule(
        ticker=ticker,
        company_name=request.company_name,
        interval_hours=request.interval_hours,
    )

    next_check = datetime.now(timezone.utc) + timedelta(hours=request.interval_hours)

    logger.info(
        "monitoring_started",
        ticker=ticker,
        interval_hours=request.interval_hours,
    )

    return MonitorStartResponse(
        ticker=ticker,
        status="MONITORING_STARTED",
        next_check_at=next_check,
        message=f"Monitoring started for {ticker}. Will check every {request.interval_hours} hours.",
    )


@router.delete(
    "/monitor/{ticker}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Stop Stock Monitoring",
    description="Stop scheduled monitoring for a stock ticker.",
    responses={
        204: {"description": "Monitoring stopped"},
        404: {"description": "No active monitoring for this ticker"},
    },
)
async def stop_monitoring(
    ticker: str,
    storage: StoragePort = Depends(get_storage),
):
    """Stop monitoring for a stock."""
    ticker = ticker.upper()

    deleted = await storage.delete_monitoring_schedule(ticker)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active monitoring for ticker: {ticker}",
        )

    logger.info("monitoring_stopped", ticker=ticker)


# =============================================================================
# List & Query Endpoints (for UI)
# =============================================================================

@router.get(
    "/analyses",
    response_model=List[AnalysisRecord],
    summary="List All Analyses",
    description="""
    List all analysis jobs with optional filters.

    **Filters:**
    - `ticker`: Filter by stock ticker symbol
    - `status`: Filter by job status (PENDING, RUNNING, COMPLETED, FAILED)
    - `analysis_type`: Filter by type (ON_DEMAND, PROACTIVE_ALERT)
    - `limit`: Maximum number of results (default: 20, max: 100)
    - `offset`: Number of results to skip for pagination

    Results are sorted by creation date (newest first).
    """,
    responses={
        200: {"description": "List of analyses retrieved"},
    },
)
async def list_analyses(
    ticker: Optional[str] = Query(default=None, description="Filter by ticker symbol"),
    status_filter: Optional[str] = Query(default=None, alias="status", description="Filter by status"),
    analysis_type: Optional[str] = Query(default=None, description="Filter by analysis type"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip"),
    storage: StoragePort = Depends(get_storage),
) -> List[AnalysisRecord]:
    """List all analyses with optional filters."""
    return await storage.list_analyses(
        ticker=ticker.upper() if ticker else None,
        status=status_filter,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/monitors",
    response_model=List[MonitoringSchedule],
    summary="List Active Monitors",
    description="Get all active stock monitoring schedules.",
    responses={
        200: {"description": "List of active monitoring schedules"},
    },
)
async def list_monitors(
    storage: StoragePort = Depends(get_storage),
) -> List[MonitoringSchedule]:
    """List all active monitoring schedules."""
    return await storage.get_active_monitoring_schedules()


@router.get(
    "/status/{job_id}/thoughts",
    response_model=List[AgentThoughtRecord],
    summary="Get Agent Thoughts",
    description="""
    Get the agent's reasoning steps for an analysis job.

    This endpoint returns the internal thought process of the AI agent,
    including planning, tool selection, reflection, and synthesis steps.
    Useful for debugging and transparency.
    """,
    responses={
        200: {"description": "Agent thoughts retrieved"},
        404: {"description": "Job not found"},
    },
)
async def get_analysis_thoughts(
    job_id: str,
    storage: StoragePort = Depends(get_storage),
) -> List[AgentThoughtRecord]:
    """Get agent thoughts for an analysis job."""
    # First verify the job exists
    job_status = await storage.get_job_status(job_id)
    if not job_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job not found: {job_id}",
        )

    return await storage.get_agent_thoughts(job_id)


@router.get(
    "/status/{job_id}/tools",
    response_model=List[ToolExecutionRecord],
    summary="Get Tool Executions",
    description="""
    Get the tool execution history for an analysis job.

    This endpoint returns details about each tool that was executed,
    including input parameters, outputs, execution time, and success status.
    Useful for debugging and performance analysis.
    """,
    responses={
        200: {"description": "Tool executions retrieved"},
        404: {"description": "Job not found"},
    },
)
async def get_analysis_tools(
    job_id: str,
    storage: StoragePort = Depends(get_storage),
) -> List[ToolExecutionRecord]:
    """Get tool executions for an analysis job."""

    job_status = await storage.get_job_status(job_id)
    if not job_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job not found: {job_id}",
        )

    return await storage.get_tool_executions(job_id)


# =============================================================================
# Semantic Search
# =============================================================================

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic Search",
    description="""
    Search across past analysis summaries and key findings using semantic similarity.

    This endpoint enables intelligent search over historical analyses:
    - Uses vector embeddings for semantic matching (not just keyword search)
    - Returns results ranked by relevance score (0-1)
    - Optionally filter by stock ticker

    **Example queries:**
    - "EV market growth concerns"
    - "positive revenue outlook"
    - "risk factors for tech companies"
    - "analyst price target upgrades"
    """,
    responses={
        200: {
            "description": "Search results",
            "content": {
                "application/json": {
                    "example": {
                        "query": "revenue growth",
                        "results": [
                            {
                                "analysis_id": "job-123",
                                "content_type": "summary",
                                "content_text": "Tesla shows strong revenue growth...",
                                "score": 0.89,
                                "metadata": {"ticker": "TSLA", "sentiment_score": 0.65}
                            }
                        ],
                        "total_results": 1
                    }
                }
            }
        },
        422: {"description": "Validation error (query too short/long)"},
    },
)
async def semantic_search(
    request: SearchRequest,
    storage: StoragePort = Depends(get_storage),
    embeddings: EmbeddingsPort = Depends(get_embeddings_provider),
) -> SearchResponse:
    """
    Search for similar past analyses using semantic similarity.

    The search is performed across:
    - Analysis summaries (content_type: "summary")
    - Key findings (content_type: "key_finding")

    Results are ranked by cosine similarity to the query embedding.
    """
    from src.application.services.embedding_service import EmbeddingService

    logger.info(
        "semantic_search_request",
        query=request.query[:50],
        ticker=request.ticker,
        limit=request.limit,
    )

    service = EmbeddingService(embeddings, storage)

    try:
        results = await service.search_similar_analyses(
            query=request.query,
            ticker=request.ticker.upper() if request.ticker else None,
            limit=request.limit,
        )

        search_results = [
            SearchResult(
                analysis_id=r.get("analysis_id", ""),
                content_type=r.get("content_type", "summary"),
                content_text=r.get("content_text", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

        logger.info(
            "semantic_search_complete",
            query=request.query[:50],
            num_results=len(search_results),
        )

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
        )

    except Exception as e:
        logger.error(
            "semantic_search_failed",
            query=request.query[:50],
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


# =============================================================================
# Health Check
# =============================================================================

@router.get(
    "/health",
    summary="Health Check",
    description="Check the health of the A.I.R.A. service and its dependencies.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "version": "1.0.0",
                        "components": {
                            "storage": "healthy",
                            "llm": "healthy",
                            "news": "healthy",
                            "financial": "healthy",
                            "search": "healthy"
                        }
                    }
                }
            }
        },
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check(
    storage: StoragePort = Depends(get_storage),
    llm: LLMPort = Depends(get_llm_provider),
    news: NewsPort = Depends(get_news_provider),
    financial: FinancialPort = Depends(get_financial_provider),
) -> Dict[str, Any]:
    """Check service health by verifying all providers."""
    settings = get_settings()

    # Check each provider's health
    storage_healthy = await storage.health_check()
    llm_healthy = await llm.health_check()
    news_healthy = await news.health_check()
    financial_healthy = await financial.health_check()

    # Search provider is optional
    search_provider = get_search_provider()
    search_healthy = True
    if search_provider is not None:
        try:
            search_healthy = await search_provider.health_check()
        except Exception:
            search_healthy = False

    components = {
        "storage": "healthy" if storage_healthy else "unhealthy",
        "llm": "healthy" if llm_healthy else "unhealthy",
        "news": "healthy" if news_healthy else "unhealthy",
        "financial": "healthy" if financial_healthy else "unhealthy",
        "search": "healthy" if search_healthy else "unhealthy",
    }

    overall_healthy = all(v == "healthy" for v in components.values())

    if not overall_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "components": components,
            },
        )

    return {
        "status": "healthy",
        "version": settings.app.version,
        "components": components,
    }


