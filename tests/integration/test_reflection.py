"""
Integration tests for the agent reflection mechanism.

Tests that reflection:
1. Triggers when data quality issues are detected
2. Uses DIFFERENT strategies on retry (not the same queries)
3. Eventually proceeds to synthesis after max reflections
4. Accumulates results correctly
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from unittest.mock import AsyncMock

from src.adapters.agents.langgraph_agent import (
    AIRAAgent,
    ReflectionReason,
    ReflectionStrategy,
)
from src.adapters.llm.mock_llm import MockLLM
from src.adapters.financial.mock_financial import MockFinancial
from src.adapters.search.mock_search import MockSearch
from src.adapters.storage.memory_repository import MemoryRepository
from src.application.ports.news_port import NewsPort
from src.domain.models import NewsArticle, WebSearchResult
from src.tools.news_retriever import NewsRetrieverTool
from src.tools.sentiment_analyzer import SentimentAnalyzerTool
from src.tools.data_fetcher import DataFetcherTool
from src.tools.web_researcher import WebResearcherTool


class EmptyNewsProvider(NewsPort):
    """News provider that returns no articles - triggers NO_NEWS reflection."""

    def __init__(self):
        self.call_count = 0

    async def search_news(self, query: str, num_articles: int = 5, **kwargs) -> List[NewsArticle]:
        self.call_count += 1
        return []

    async def get_company_news(self, company_name: str, ticker: str, num_articles: int = 5) -> List[NewsArticle]:
        self.call_count += 1
        return []

    async def health_check(self) -> bool:
        return True


class OldNewsProvider(NewsPort):
    """News provider that returns only old articles - triggers NEWS_TOO_OLD reflection."""

    def __init__(self, days_old: int = 60):
        self.days_old = days_old
        self.call_count = 0

    async def search_news(self, query: str, num_articles: int = 5, **kwargs) -> List[NewsArticle]:
        self.call_count += 1
        return self._get_old_articles(num_articles)

    async def get_company_news(self, company_name: str, ticker: str, num_articles: int = 5) -> List[NewsArticle]:
        self.call_count += 1
        return self._get_old_articles(num_articles)

    def _get_old_articles(self, count: int) -> List[NewsArticle]:
        old_date = datetime.now(timezone.utc) - timedelta(days=self.days_old)
        return [
            NewsArticle(
                title=f"Old Article {i}",
                description=f"This is an old article from {self.days_old} days ago",
                url=f"https://example.com/old-article-{i}",
                source="Old News Source",
                published_at=old_date,
                content="Old content",
            )
            for i in range(count)
        ]

    async def health_check(self) -> bool:
        return True


class TrackingSearchProvider:
    """
    Search provider that tracks:
    - All calls made
    - Which research_focus was used (inferred from query patterns)
    """

    def __init__(self):
        self.call_count = 0
        self.queries_received: List[str] = []
        self.call_batches: List[List[str]] = []  # Group queries by call batch
        self._current_batch: List[str] = []

    def start_new_batch(self):
        """Call this to mark a new batch of queries (e.g., new research focus)."""
        if self._current_batch:
            self.call_batches.append(self._current_batch)
            self._current_batch = []

    async def search(self, query: str, num_results: int = 10) -> List[WebSearchResult]:
        self.call_count += 1
        self.queries_received.append(query)
        self._current_batch.append(query)

        # Return results with unique URLs
        return [
            WebSearchResult(
                title=f"Result for: {query[:40]}",
                snippet=f"Search result snippet for query: {query}",
                url=f"https://example.com/result-{self.call_count}-{i}",
                source="example.com",
            )
            for i in range(2)  # Return 2 results per query
        ]

    def finalize(self):
        """Call at end to capture final batch."""
        if self._current_batch:
            self.call_batches.append(self._current_batch)
            self._current_batch = []

    async def health_check(self) -> bool:
        return True

    def get_focus_from_queries(self, queries: List[str]) -> str:
        """Infer which research focus was used from query patterns."""
        query_text = " ".join(queries).lower()

        if "breaking news" in query_text or "latest news today" in query_text:
            return "news"
        elif "analyst rating" in query_text and "upgrade downgrade" in query_text:
            return "analyst_ratings"
        elif "earnings" in query_text and "beat miss" in query_text:
            return "earnings"
        elif "competitors" in query_text or "market share" in query_text:
            return "competitive"
        elif "risks concerns" in query_text or "bearish" in query_text:
            return "risks"
        else:
            return "general"


class EmptySearchProvider:
    """Search provider that returns no results - triggers INSUFFICIENT_RESEARCH reflection."""

    def __init__(self):
        self.call_count = 0
        self.queries_received: List[str] = []
        self.focuses_inferred: List[str] = []

    async def search(self, query: str, num_results: int = 10) -> List:
        self.call_count += 1
        self.queries_received.append(query)
        return []

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def empty_news_provider():
    return EmptyNewsProvider()


@pytest.fixture
def old_news_provider():
    return OldNewsProvider(days_old=60)


@pytest.fixture
def tracking_search_provider():
    return TrackingSearchProvider()


@pytest.fixture
def empty_search_provider():
    return EmptySearchProvider()


@pytest.fixture
def agent_with_empty_news(empty_news_provider, tracking_search_provider):
    """Agent that will trigger NO_NEWS reflection."""
    llm = MockLLM()
    financial = MockFinancial()
    storage = MemoryRepository()

    news_tool = NewsRetrieverTool(news_provider=empty_news_provider)
    sentiment_tool = SentimentAnalyzerTool(llm_provider=llm)
    data_tool = DataFetcherTool(financial_provider=financial)
    web_research_tool = WebResearcherTool(search_provider=tracking_search_provider)

    return AIRAAgent(
        llm_provider=llm,
        news_tool=news_tool,
        sentiment_tool=sentiment_tool,
        data_tool=data_tool,
        web_research_tool=web_research_tool,
        storage=storage,
    )


@pytest.fixture
def agent_with_old_news(old_news_provider, tracking_search_provider):
    """Agent that will trigger NEWS_TOO_OLD reflection."""
    llm = MockLLM()
    financial = MockFinancial()
    storage = MemoryRepository()

    news_tool = NewsRetrieverTool(news_provider=old_news_provider)
    sentiment_tool = SentimentAnalyzerTool(llm_provider=llm)
    data_tool = DataFetcherTool(financial_provider=financial)
    web_research_tool = WebResearcherTool(search_provider=tracking_search_provider)

    return AIRAAgent(
        llm_provider=llm,
        news_tool=news_tool,
        sentiment_tool=sentiment_tool,
        data_tool=data_tool,
        web_research_tool=web_research_tool,
        storage=storage,
    )


@pytest.fixture
def agent_with_empty_search(empty_search_provider):
    """Agent that will trigger INSUFFICIENT_RESEARCH reflection."""
    from src.adapters.news.mock_news import MockNews

    llm = MockLLM()
    news = MockNews()
    financial = MockFinancial()
    storage = MemoryRepository()

    news_tool = NewsRetrieverTool(news_provider=news)
    sentiment_tool = SentimentAnalyzerTool(llm_provider=llm)
    data_tool = DataFetcherTool(financial_provider=financial)
    web_research_tool = WebResearcherTool(search_provider=empty_search_provider)

    return AIRAAgent(
        llm_provider=llm,
        news_tool=news_tool,
        sentiment_tool=sentiment_tool,
        data_tool=data_tool,
        web_research_tool=web_research_tool,
        storage=storage,
    )


class TestReflectionMechanism:
    """Tests for the agent reflection mechanism."""

    @pytest.mark.asyncio
    async def test_reflection_triggered_and_recorded_on_no_news(
        self, agent_with_empty_news, empty_news_provider, tracking_search_provider
    ):
        """
        Test that when news API returns nothing:
        1. Reflection is triggered
        2. The reflection reason is recorded in the report
        3. News API is NOT called again on retry (we use web search instead)
        """
        report = await agent_with_empty_news.analyze(
            job_id="test-no-news-001",
            query="Analyze Tesla (TSLA)",
        )

        # Report should be generated
        assert report is not None
        assert report.company_ticker == "TSLA"

        # Reflection should have been triggered and recorded
        assert report.reflection_triggered is True, "Reflection should be marked as triggered"
        assert report.reflection_notes is not None, "Reflection notes should be recorded"
        assert "No news articles retrieved" in report.reflection_notes

        # News provider should be called only ONCE (not retried - we use web search instead)
        assert empty_news_provider.call_count == 1, (
            f"News provider should be called once, got {empty_news_provider.call_count}"
        )

    @pytest.mark.asyncio
    async def test_retry_uses_news_focus_not_general_focus(
        self, agent_with_empty_news, tracking_search_provider
    ):
        """
        Test that on retry after NO_NEWS, the agent uses 'news' research focus
        which produces DIFFERENT queries than the initial 'general' focus.

        This is the KEY test - verifies the retry actually does something different.
        """
        report = await agent_with_empty_news.analyze(
            job_id="test-different-focus-001",
            query="Analyze Tesla (TSLA)",
        )

        assert report is not None
        queries = tracking_search_provider.queries_received

        # Verify we have queries from multiple passes
        assert len(queries) > 4, f"Expected multiple query batches, got {len(queries)} queries"

        # Find queries that are specific to "news" focus
        # These patterns are from web_researcher.py "news" focus
        news_focus_patterns = [
            "stock latest news today",
            "breaking news market update",
            "recent developments announcements",
        ]

        # Find queries specific to "general" focus
        general_focus_patterns = [
            "analyst rating price target",
            "investment outlook",
        ]

        news_focus_queries = [
            q for q in queries
            if any(pattern in q.lower() for pattern in news_focus_patterns)
        ]
        general_focus_queries = [
            q for q in queries
            if any(pattern in q.lower() for pattern in general_focus_patterns)
        ]

        # Should have BOTH general (initial) AND news (retry) queries
        assert len(general_focus_queries) > 0, (
            f"Should have general focus queries from initial pass. Queries: {queries}"
        )
        assert len(news_focus_queries) > 0, (
            f"Should have news focus queries from retry pass. Queries: {queries}"
        )

    @pytest.mark.asyncio
    async def test_old_news_triggers_reflection_and_uses_web_search(
        self, agent_with_old_news, old_news_provider, tracking_search_provider
    ):
        """
        Test that when all news articles are too old (>30 days):
        1. Reflection triggers due to NEWS_TOO_OLD
        2. Agent uses web search with 'news' focus to find recent coverage
        """
        report = await agent_with_old_news.analyze(
            job_id="test-old-news-001",
            query="Analyze Apple (AAPL)",
        )

        assert report is not None
        assert report.company_ticker == "AAPL"

        # Reflection should be triggered due to old news
        assert report.reflection_triggered is True
        assert "older than 30 days" in (report.reflection_notes or "")

        # Verify news focus was used on retry
        queries = tracking_search_provider.queries_received
        news_focus_queries = [
            q for q in queries
            if "latest news" in q.lower() or "breaking news" in q.lower()
        ]
        assert len(news_focus_queries) > 0, (
            f"Should use news-specific queries on retry. Got: {queries}"
        )

    @pytest.mark.asyncio
    async def test_empty_search_triggers_different_research_focus(
        self, agent_with_empty_search, empty_search_provider
    ):
        """
        Test that when web research returns no results:
        1. Reflection triggers due to INSUFFICIENT_RESEARCH
        2. Agent tries a DIFFERENT research focus (analyst_ratings, earnings, etc.)
        """
        report = await agent_with_empty_search.analyze(
            job_id="test-empty-search-001",
            query="Analyze Google (GOOGL)",
        )

        assert report is not None

        queries = empty_search_provider.queries_received

        # Should have tried multiple times
        # Initial pass: ~4 queries (general focus)
        # If reflection triggered, retry with different focus: ~4-5 more queries
        assert len(queries) >= 8, (
            f"Expected queries from multiple focuses, got {len(queries)}: {queries}"
        )

        # Verify different focus patterns appear
        # Look for analyst_ratings focus patterns (first alternative tried)
        analyst_patterns = ["upgrade downgrade", "wall street consensus", "recommendations buy sell"]
        has_analyst_queries = any(
            any(pattern in q.lower() for pattern in analyst_patterns)
            for q in queries
        )

        assert has_analyst_queries, (
            f"Expected analyst_ratings focus queries on retry. Got: {queries}"
        )

    @pytest.mark.asyncio
    async def test_max_reflection_cycles_respected(self, agent_with_empty_news):
        """
        Test that agent stops reflecting after max_reflection_cycles (default: 2).
        Should complete analysis even if issues persist.
        """
        report = await agent_with_empty_news.analyze(
            job_id="test-max-cycles-001",
            query="Analyze Tesla (TSLA)",
        )

        assert report is not None
        assert report.company_ticker == "TSLA"

        # Analysis should complete successfully
        assert report.analysis_summary is not None
        assert len(report.analysis_summary) > 0

        # Reflection was triggered (issues persisted)
        assert report.reflection_triggered is True

    @pytest.mark.asyncio
    async def test_no_reflection_when_data_is_good(self):
        """
        Test that reflection does NOT trigger when:
        - News is recent
        - Financial data is available
        - Web research returns results
        """
        from src.adapters.news.mock_news import MockNews

        llm = MockLLM()
        news = MockNews()  # Returns recent, good news data
        financial = MockFinancial()
        search = MockSearch()
        storage = MemoryRepository()

        news_tool = NewsRetrieverTool(news_provider=news)
        sentiment_tool = SentimentAnalyzerTool(llm_provider=llm)
        data_tool = DataFetcherTool(financial_provider=financial)
        web_research_tool = WebResearcherTool(search_provider=search)

        agent = AIRAAgent(
            llm_provider=llm,
            news_tool=news_tool,
            sentiment_tool=sentiment_tool,
            data_tool=data_tool,
            web_research_tool=web_research_tool,
            storage=storage,
        )

        report = await agent.analyze(
            job_id="test-good-data-001",
            query="Analyze Tesla (TSLA)",
        )

        assert report is not None
        assert report.company_ticker == "TSLA"

        # Reflection should NOT have been triggered
        assert report.reflection_triggered is False, "Reflection should not trigger with good data"
        assert report.reflection_notes is None, "No reflection notes when not triggered"

    @pytest.mark.asyncio
    async def test_tools_used_not_duplicated(self, agent_with_empty_news):
        """
        Test that tools_used list doesn't contain duplicates even after
        multiple reflection cycles.
        """
        report = await agent_with_empty_news.analyze(
            job_id="test-no-dupes-001",
            query="Analyze Apple (AAPL)",
        )

        assert report is not None

        tools = report.tools_used
        unique_tools = set(tools)

        assert len(tools) == len(unique_tools), (
            f"Duplicate tools found: {tools}"
        )

    @pytest.mark.asyncio
    async def test_web_research_results_accumulated(
        self, agent_with_empty_news, tracking_search_provider
    ):
        """
        Test that web research results from retry are ADDED to existing results,
        not replacing them.
        """
        report = await agent_with_empty_news.analyze(
            job_id="test-accumulate-001",
            query="Analyze Microsoft (MSFT)",
        )

        assert report is not None
        assert report.web_research is not None

        # Check queries were accumulated
        queries_used = report.web_research.get("queries_used", [])
        results = report.web_research.get("results", [])

        # Initial "general" focus: 4 queries
        # Retry "news" focus: 5 queries
        # With 2 reflection cycles: should have queries from 3 passes
        assert len(queries_used) >= 9, (
            f"Expected accumulated queries from multiple passes, got {len(queries_used)}"
        )

        # Results should also be accumulated
        # Each query returns 2 results, so should have many results
        assert len(results) >= 8, (
            f"Expected accumulated results, got {len(results)}"
        )

        # Verify results have unique URLs (deduplication working)
        urls = [r.get("url") for r in results]
        assert len(urls) == len(set(urls)), "Results should have unique URLs"

    @pytest.mark.asyncio
    async def test_reflection_notes_contain_actual_reason(self, agent_with_empty_news):
        """
        Test that reflection_notes in the final report accurately describe
        why reflection was triggered.
        """
        report = await agent_with_empty_news.analyze(
            job_id="test-notes-001",
            query="Analyze Tesla (TSLA)",
        )

        assert report is not None
        assert report.reflection_triggered is True
        assert report.reflection_notes is not None

        # Should mention the actual issue
        assert "No news articles retrieved" in report.reflection_notes or \
               "no news" in report.reflection_notes.lower(), (
            f"Reflection notes should explain the issue. Got: {report.reflection_notes}"
        )
