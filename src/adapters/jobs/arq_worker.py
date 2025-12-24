"""
Arq Worker - Background job processing for A.I.R.A.

This module defines the Arq worker configuration and job functions
for processing analysis requests asynchronously.
"""

from typing import Any, Dict

from arq.connections import RedisSettings

from src.adapters.api.dependencies import get_agent, get_storage
from src.config.logging import get_logger, setup_logging
from src.config.settings import get_settings

logger = get_logger(__name__)


async def run_analysis(
    ctx: Dict[str, Any],
    job_id: str,
    query: str,
) -> Dict[str, Any]:
    """
    Arq job function to run company analysis.

    Args:
        ctx: Arq context with redis connection
        job_id: Unique job identifier
        query: User's analysis query

    Returns:
        Analysis result summary
    """
    logger.info("arq_job_start", job_id=job_id, query=query[:100])

    agent = get_agent()
    storage = get_storage()

    try:
        # Update status to RUNNING
        await storage.update_analysis_status(
            job_id=job_id,
            status="RUNNING",
            progress="Analysis in progress...",
        )

        # Run the analysis
        report = await agent.analyze(job_id=job_id, query=query)

        # Save results
        await storage.complete_analysis(
            job_id=job_id,
            report=report,
            tools_used=report.tools_used,
            iteration_count=report.iteration_count,
            reflection_triggered=report.reflection_triggered,
        )

        logger.info(
            "arq_job_complete",
            job_id=job_id,
            ticker=report.company_ticker,
            sentiment_score=report.sentiment_score,
        )

        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "ticker": report.company_ticker,
            "sentiment_score": report.sentiment_score,
        }

    except Exception as e:
        logger.error("arq_job_failed", job_id=job_id, error=str(e))

        await storage.update_analysis_status(
            job_id=job_id,
            status="FAILED",
            error_message=str(e),
        )

        return {
            "job_id": job_id,
            "status": "FAILED",
            "error": str(e),
        }


async def check_monitored_stocks(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    On-demand job to check all monitored stocks for new news.

    This can be triggered manually via the Arq queue. For scheduled
    monitoring, use APScheduler via MonitoringScheduler.start().

    Args:
        ctx: Arq context

    Returns:
        Summary of checks performed
    """
    from src.adapters.jobs.scheduler import MonitoringScheduler

    logger.info("arq_monitoring_check_start")

    scheduler = MonitoringScheduler()
    await scheduler._check_all_monitored_stocks()

    logger.info("arq_monitoring_check_complete")

    return {"status": "completed"}


async def startup(ctx: Dict[str, Any]) -> None:
    """
    Arq worker startup hook.

    Called when the worker starts up. Used to initialize logging
    and reset cached settings/dependencie.
    """
    from src.config.settings import clear_settings_cache
    from src.adapters.api.dependencies import reset_dependencies

    # Reset caches to pick up Docker ENV vars (POSTGRES_HOST, REDIS_HOST, etc.)
    clear_settings_cache()
    reset_dependencies()

    settings = get_settings()
    setup_logging(
        log_level=settings.app.log_level,
        log_format=settings.app.log_format,
    )
    logger.info("arq_worker_startup")


async def shutdown(ctx: Dict[str, Any]) -> None:
    """
    Arq worker shutdown hook.

    Called when the worker shuts down. Used to cleanup connections.
    """
    logger.info("arq_worker_shutdown")


def _get_redis_settings() -> RedisSettings:
    """
    Get Redis settings for Arq worker.
    """
    from src.config.settings import clear_settings_cache
    clear_settings_cache()
    settings = get_settings()
    return RedisSettings(
        host=settings.redis.host,
        port=settings.redis.port,
        database=settings.redis.db,
        password=settings.redis.password or None,
    )


class WorkerSettings:
    """
    Arq worker configuration.

    This class configures the Arq worker with:
    - Redis connection settings
    - Available job functions
    - Startup/shutdown hooks
    """

    # Job functions available to the worker
    functions = [run_analysis, check_monitored_stocks]

    cron_jobs = []

    # Startup and shutdown hooks
    on_startup = startup
    on_shutdown = shutdown

    # Redis connection settings
    redis_settings = _get_redis_settings()

    # Job settings
    max_jobs = 10
    job_timeout = 300  # 5 minutes max per job
    keep_result = 3600  # Keep results for 1 hour
    poll_delay = 0.5  # Poll interval in seconds
