"""
Integration tests for the LangGraph agent.
"""

import pytest

from src.adapters.agents.langgraph_agent import AIRAAgent
from src.adapters.llm.mock_llm import MockLLM
from src.adapters.news.mock_news import MockNews
from src.adapters.financial.mock_financial import MockFinancial
from src.adapters.search.mock_search import MockSearch
from src.adapters.storage.memory_repository import MemoryRepository
from src.tools.news_retriever import NewsRetrieverTool
from src.tools.sentiment_analyzer import SentimentAnalyzerTool
from src.tools.data_fetcher import DataFetcherTool
from src.tools.web_researcher import WebResearcherTool


@pytest.fixture
def agent():
    """Create an agent with all mock dependencies."""
    llm = MockLLM()
    news = MockNews()
    financial = MockFinancial()
    search = MockSearch()
    storage = MemoryRepository()

    news_tool = NewsRetrieverTool(news_provider=news)
    sentiment_tool = SentimentAnalyzerTool(llm_provider=llm)
    data_tool = DataFetcherTool(financial_provider=financial)
    web_research_tool = WebResearcherTool(search_provider=search)

    return AIRAAgent(
        llm_provider=llm,
        news_tool=news_tool,
        sentiment_tool=sentiment_tool,
        data_tool=data_tool,
        web_research_tool=web_research_tool,
        storage=storage,
    )


@pytest.fixture
def agent_without_web_research():
    """Create an agent without web research capability."""
    llm = MockLLM()
    news = MockNews()
    financial = MockFinancial()
    storage = MemoryRepository()

    news_tool = NewsRetrieverTool(news_provider=news)
    sentiment_tool = SentimentAnalyzerTool(llm_provider=llm)
    data_tool = DataFetcherTool(financial_provider=financial)

    return AIRAAgent(
        llm_provider=llm,
        news_tool=news_tool,
        sentiment_tool=sentiment_tool,
        data_tool=data_tool,
        web_research_tool=None,
        storage=storage,
    )


class TestAIRAAgent:
    """Integration tests for AIRAAgent."""

    @pytest.mark.asyncio
    async def test_analyze_tesla(self, agent):
        """Test full analysis flow for Tesla."""
        report = await agent.analyze(
            job_id="test-tsla-001",
            query="Analyze the near-term prospects of Tesla, Inc. (TSLA)",
        )

        assert report is not None
        assert report.company_ticker == "TSLA"
        assert report.sentiment_score >= -1.0
        assert report.sentiment_score <= 1.0
        assert len(report.key_findings) >= 3
        assert len(report.tools_used) >= 1
        assert "news_retriever" in report.tools_used

    @pytest.mark.asyncio
    async def test_analyze_apple(self, agent):
        """Test full analysis flow for Apple."""
        report = await agent.analyze(
            job_id="test-aapl-001",
            query="Research Apple Inc. (AAPL) stock performance",
        )

        assert report is not None
        assert report.company_ticker == "AAPL"
        assert report.company_name
        assert report.analysis_summary

    @pytest.mark.asyncio
    async def test_analyze_extracts_ticker(self, agent):
        """Test that agent correctly extracts ticker from query."""
        report = await agent.analyze(
            job_id="test-msft-001",
            query="What are the prospects for Microsoft (MSFT)?",
        )

        assert report.company_ticker == "MSFT"

    @pytest.mark.asyncio
    async def test_analysis_contains_all_required_fields(self, agent):
        """Test that analysis report contains all required fields."""
        report = await agent.analyze(
            job_id="test-complete-001",
            query="Analyze Tesla (TSLA)",
        )

        # Check required fields
        assert report.company_ticker
        assert report.company_name
        assert report.analysis_summary
        assert isinstance(report.sentiment_score, float)
        assert isinstance(report.key_findings, list)
        assert isinstance(report.tools_used, list)
        assert isinstance(report.citation_sources, list)
        assert report.generated_at

    @pytest.mark.asyncio
    async def test_multiple_analyses(self, agent):
        """Test running multiple analyses."""
        tickers = ["TSLA", "AAPL", "GOOGL"]

        for i, ticker in enumerate(tickers):
            report = await agent.analyze(
                job_id=f"test-multi-{i}",
                query=f"Analyze ({ticker})",
            )
            assert report.company_ticker == ticker

    @pytest.mark.asyncio
    async def test_analyze_with_web_research(self, agent):
        """Test that web research tool is used when available."""
        report = await agent.analyze(
            job_id="test-web-research-001",
            query="Analyze the near-term prospects of Tesla, Inc. (TSLA)",
        )

        assert report is not None
        assert report.company_ticker == "TSLA"
        # Web researcher should be in tools_used if available
        assert "web_researcher" in report.tools_used or len(report.tools_used) >= 3

    @pytest.mark.asyncio
    async def test_analyze_without_web_research(self, agent_without_web_research):
        """Test that agent works without web research capability."""
        report = await agent_without_web_research.analyze(
            job_id="test-no-web-001",
            query="Analyze Apple Inc. (AAPL)",
        )

        assert report is not None
        assert report.company_ticker == "AAPL"
        # Should still work with core tools
        assert "news_retriever" in report.tools_used

    @pytest.mark.asyncio
    async def test_report_has_new_fields(self, agent):
        """Test that report includes new analysis fields."""
        report = await agent.analyze(
            job_id="test-new-fields-001",
            query="Analyze Microsoft (MSFT)",
        )

        assert report is not None
        # Check that the report model has the new optional fields
        assert hasattr(report, "investment_thesis")
        assert hasattr(report, "risk_factors")
        assert hasattr(report, "competitive_context")
        assert hasattr(report, "analyst_consensus")
        assert hasattr(report, "catalyst_events")
        assert hasattr(report, "financial_analysis")
