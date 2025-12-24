"""
APScheduler-based monitoring scheduler for A.I.R.A.

Handles scheduled monitoring of stocks, checking for new news
and triggering PROACTIVE_ALERT analyses when significant
coverage is detected.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.adapters.api.dependencies import get_arq_pool, get_news_provider, get_storage
from src.config.logging import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


class MonitoringScheduler:
    """
    Scheduler for stock monitoring tasks.

    Periodically checks monitored stocks for new news coverage
    and triggers analyses when significant changes are detected.
    """

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._settings = get_settings()

    def start(self) -> None:
        """Start the monitoring scheduler."""
        if self._scheduler is not None:
            logger.warning("scheduler_already_running")
            return

        self._scheduler = AsyncIOScheduler()

        # Add the main monitoring job
        self._scheduler.add_job(
            self._check_all_monitored_stocks,
            IntervalTrigger(hours=1),  # Check every hour
            id="monitoring_check",
            name="Check monitored stocks",
            replace_existing=True,
        )

        self._scheduler.start()
        logger.info("monitoring_scheduler_started")

    def stop(self) -> None:
        """Stop the monitoring scheduler."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
            logger.info("monitoring_scheduler_stopped")

    async def _check_all_monitored_stocks(self) -> None:
        """Check all active monitoring schedules."""
        logger.info("monitoring_check_start")

        storage = get_storage()
        news_provider = get_news_provider()

        try:
            schedules = await storage.get_active_monitoring_schedules()

            if not schedules:
                logger.debug("monitoring_check_no_schedules")
                return

            checked = 0
            triggered = 0

            for schedule in schedules:
                # Check if it's time to check this ticker
                if not self._should_check(schedule):
                    continue

                checked += 1

                try:
                    # Fetch new articles
                    articles = await news_provider.get_company_news(
                        company_name=schedule.company_name or schedule.ticker,
                        ticker=schedule.ticker,
                        num_articles=10,
                    )

                    # Calculate new article hashes
                    new_hashes = [
                        self._hash_article(a.title, a.url)
                        for a in articles
                    ]

                    # Find truly new articles
                    existing_hashes = set(schedule.article_hashes or [])
                    new_articles = [
                        h for h in new_hashes if h not in existing_hashes
                    ]

                    # Update schedule with latest check time and hashes
                    await storage.update_monitoring_schedule(
                        ticker=schedule.ticker,
                        last_check_at=datetime.now(timezone.utc),
                        article_hashes=new_hashes,
                    )

                    # Check if we should trigger an analysis
                    min_new = self._settings.monitoring.min_new_articles
                    if len(new_articles) >= min_new:
                        logger.info(
                            "monitoring_trigger_analysis",
                            ticker=schedule.ticker,
                            new_articles=len(new_articles),
                        )

                        # Trigger a PROACTIVE_ALERT analysis
                        await self._trigger_proactive_analysis(schedule.ticker)
                        triggered += 1
                    else:
                        logger.debug(
                            "monitoring_check_ticker",
                            ticker=schedule.ticker,
                            new_articles=len(new_articles),
                            threshold=min_new,
                        )

                except Exception as e:
                    logger.error(
                        "monitoring_check_ticker_error",
                        ticker=schedule.ticker,
                        error=str(e),
                    )

            logger.info(
                "monitoring_check_complete",
                checked=checked,
                triggered=triggered,
            )

        except Exception as e:
            logger.error("monitoring_check_error", error=str(e))

    def _should_check(self, schedule) -> bool:
        """Determine if a schedule should be checked now."""
        if schedule.last_check_at is None:
            return True

        interval = timedelta(hours=schedule.interval_hours)
        next_check = schedule.last_check_at + interval

        return datetime.now(timezone.utc) >= next_check

    @staticmethod
    def _hash_article(title: str, url: str) -> str:
        """Generate a hash for an article to detect duplicates."""
        content = f"{title}|{url}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def _trigger_proactive_analysis(self, ticker: str) -> None:
        """Trigger a PROACTIVE_ALERT analysis for a ticker via Arq."""
        import uuid

        storage = get_storage()
        arq_pool = await get_arq_pool()

        job_id = f"proactive-{ticker}-{str(uuid.uuid4())[:8]}"
        query = f"PROACTIVE_ALERT: Analyze {ticker} due to significant new coverage"

        # Create the analysis job record
        await storage.create_analysis(
            job_id=job_id,
            ticker=ticker,
            query=query,
            analysis_type="PROACTIVE_ALERT",
        )

        # Enqueue to Arq worker for background processing
        await arq_pool.enqueue_job("run_analysis", job_id, query)

        # Update monitoring schedule with this analysis
        await storage.update_monitoring_schedule(
            ticker=ticker,
            last_analysis_id=job_id,
        )

        logger.info(
            "proactive_analysis_enqueued",
            job_id=job_id,
            ticker=ticker,
        )


# Global scheduler instance
_scheduler: Optional[MonitoringScheduler] = None


def get_scheduler() -> MonitoringScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = MonitoringScheduler()
    return _scheduler


def start_scheduler() -> None:
    """Start the global scheduler."""
    get_scheduler().start()


def stop_scheduler() -> None:
    """Stop the global scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None
