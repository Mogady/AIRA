"""
Unit tests for agent tools.
"""

import pytest
from datetime import datetime, timezone

from src.domain.models import NewsArticle


class TestNewsRetrieverTool:
    """Tests for NewsRetrieverTool."""

    @pytest.mark.asyncio
    async def test_execute_returns_articles(self, news_tool):
        """Test that news retriever returns articles."""
        result = await news_tool.execute(
            company="Tesla, Inc.",
            ticker="TSLA",
            num_articles=5,
        )

        assert result is not None
        assert len(result.articles) > 0
        assert result.query_used
        assert all(isinstance(a, NewsArticle) for a in result.articles)

    @pytest.mark.asyncio
    async def test_execute_with_different_ticker(self, news_tool):
        """Test news retriever with different ticker."""
        result = await news_tool.execute(
            company="Apple Inc.",
            ticker="AAPL",
            num_articles=3,
        )

        assert result is not None
        assert len(result.articles) <= 3

    @pytest.mark.asyncio
    async def test_execute_from_dict(self, news_tool):
        """Test execution from dictionary parameters."""
        result = await news_tool.execute_from_dict({
            "company": "Microsoft",
            "ticker": "MSFT",
            "num_articles": 5,
        })

        assert result is not None
        assert len(result.articles) > 0

    def test_tool_definition(self, news_tool):
        """Test tool definition for LLM."""
        definition = news_tool.get_tool_definition()

        assert definition["name"] == "news_retriever"
        assert "description" in definition
        assert "parameters" in definition
        assert "company" in definition["parameters"]["properties"]
        assert "ticker" in definition["parameters"]["properties"]


class TestSentimentAnalyzerTool:
    """Tests for SentimentAnalyzerTool."""

    @pytest.mark.asyncio
    async def test_execute_returns_sentiment(self, sentiment_tool):
        """Test that sentiment analyzer returns results."""
        articles = [
            NewsArticle(
                title="Tesla Reports Strong Earnings",
                description="Beat expectations",
                url="https://example.com",
                source="Test",
                published_at=datetime.now(timezone.utc),
            ),
            NewsArticle(
                title="Tesla Faces Competition",
                description="Market challenges ahead",
                url="https://example.com",
                source="Test",
                published_at=datetime.now(timezone.utc),
            ),
        ]

        result = await sentiment_tool.execute(
            articles=articles,
            ticker="TSLA",
            company_name="Tesla, Inc.",
        )

        assert result is not None
        assert result.sentiment_score >= -1.0
        assert result.sentiment_score <= 1.0
        assert result.overall_sentiment in ["positive", "negative", "neutral", "mixed"]

    @pytest.mark.asyncio
    async def test_execute_empty_articles(self, sentiment_tool):
        """Test sentiment analyzer with empty articles."""
        result = await sentiment_tool.execute(
            articles=[],
            ticker="TSLA",
            company_name="Tesla",
        )

        assert result.sentiment_score == 0.0
        assert result.overall_sentiment == "neutral"

    def test_tool_definition(self, sentiment_tool):
        """Test tool definition for LLM."""
        definition = sentiment_tool.get_tool_definition()

        assert definition["name"] == "sentiment_analyzer"
        assert "description" in definition


class TestDataFetcherTool:
    """Tests for DataFetcherTool."""

    @pytest.mark.asyncio
    async def test_execute_returns_data(self, data_tool):
        """Test that data fetcher returns financial data."""
        result = await data_tool.execute(ticker="TSLA")

        assert result is not None
        assert result.ticker == "TSLA"
        assert result.current_price is not None
        assert result.market_cap is not None

    @pytest.mark.asyncio
    async def test_execute_with_unknown_ticker(self, data_tool):
        """Test data fetcher with unknown ticker."""
        result = await data_tool.execute(ticker="UNKNOWN")

        assert result is not None
        assert result.ticker == "UNKNOWN"
        # Should return default data

    @pytest.mark.asyncio
    async def test_execute_from_dict(self, data_tool):
        """Test execution from dictionary parameters."""
        result = await data_tool.execute_from_dict({"ticker": "AAPL"})

        assert result is not None
        assert result.ticker == "AAPL"

    def test_format_for_report(self, data_tool, mock_financial):
        """Test report formatting."""
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            data_tool.execute(ticker="TSLA")
        )
        formatted = data_tool.format_for_report(result)

        assert "ticker" in formatted
        assert "market_cap_formatted" in formatted
        assert "recent_revenue" in formatted

    def test_tool_definition(self, data_tool):
        """Test tool definition for LLM."""
        definition = data_tool.get_tool_definition()

        assert definition["name"] == "data_fetcher"
        assert "ticker" in definition["parameters"]["properties"]
