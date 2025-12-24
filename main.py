"""
A.I.R.A. - Autonomous Investment Research Agent

Main FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.adapters.api.dependencies import close_arq_pool
from src.adapters.api.routes import router
from src.adapters.jobs.scheduler import start_scheduler, stop_scheduler
from src.config.logging import setup_logging, get_logger
from src.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    setup_logging(
        log_level=settings.app.log_level,
        log_format=settings.app.log_format,
    )
    logger = get_logger(__name__)
    logger.info(
        "application_startup",
        app_name=settings.app.name,
        version=settings.app.version,
        debug=settings.app.debug,
    )

    # Start the monitoring scheduler
    if settings.monitoring.enabled:
        start_scheduler()
        logger.info("monitoring_scheduler_started")

    yield

    # Shutdown
    stop_scheduler()
    await close_arq_pool()
    logger.info("application_shutdown")


# Create FastAPI app with detailed OpenAPI documentation
settings = get_settings()

app = FastAPI(
    title="A.I.R.A. - Autonomous Investment Research Agent",
    description="""
## Overview

A.I.R.A. is an AI-powered investment research agent that autonomously analyzes companies
using news coverage, sentiment analysis, and financial data.

## Features

- **Autonomous Analysis**: Submit a query and receive a comprehensive investment analysis report
- **Multi-Source Data**: Aggregates news, sentiment, and financial metrics
- **Reflection Loop**: Self-corrects when initial data is insufficient
- **Scheduled Monitoring**: Set up automatic monitoring for stocks of interest
- **Structured Reports**: JSON-formatted reports with citations and key findings

## Workflow

```
1. POST /analyze     → Submit analysis request → Returns job_id
2. GET /status/{id}  → Poll for results → Returns status + report when complete
3. POST /monitor_start → Start 24h monitoring → Returns schedule confirmation
```

## Job Status Flow

```
PENDING → RUNNING → COMPLETED (with report)
                  → FAILED (with error message)
```

## Example Usage

```bash
# Start an analysis
curl -X POST "http://localhost:8000/analyze" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Analyze the near-term prospects of Tesla, Inc. (TSLA)"}'

# Check status
curl "http://localhost:8000/status/{job_id}"
```

## Technologies

- **Agent Framework**: LangGraph with ReAct pattern
- **LLM**: Claude Sonnet 4.5 (Anthropic)
- **Embeddings**: OpenAI text-embedding-ada-002
- **News**: NewsAPI.org
- **Database**: PostgreSQL + pgvector
    """,
    version=settings.app.version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, tags=["Analysis"])


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirects to docs."""
    return {
        "name": "A.I.R.A. - Autonomous Investment Research Agent",
        "version": settings.app.version,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.app.debug,
        workers=settings.api.workers,
    )
