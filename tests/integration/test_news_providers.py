"""
Integration tests for News Providers.

Tests both MockNews and NewsAPIClient implementations to ensure
they correctly implement the NewsPort interface.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

from src.adapters.news.mock_news import MockNews
from src.adapters.news.newsapi_client import NewsAPIClient
from src.domain.exceptions import NewsRetrievalError
from src.domain.models import NewsArticle


# =============================================================================
# MockNews Integration Tests
# =============================================================================


class TestMockNews:
    """Integration tests for the MockNews provider."""

    @pytest.fixture
    def mock_news(self):
        """Create a fresh MockNews instance for each test."""
        return MockNews()

    @pytest.mark.asyncio
    async def test_search_news_returns_articles(self, mock_news):
        """Test that search_news returns a list of NewsArticle objects."""
        articles = await mock_news.search_news(query="TSLA", num_articles=3)

        assert isinstance(articles, list)
        assert len(articles) <= 3
        assert all(isinstance(a, NewsArticle) for a in articles)

    @pytest.mark.asyncio
    async def test_search_news_known_ticker(self, mock_news):
        """Test search returns ticker-specific articles for known tickers."""
        articles = await mock_news.search_news(query="Tesla TSLA", num_articles=5)

        assert len(articles) > 0
        # TSLA articles should mention Tesla in title or description
        assert any("Tesla" in a.title or "Tesla" in (a.description or "") for a in articles)

    @pytest.mark.asyncio
    async def test_search_news_unknown_ticker(self, mock_news):
        """Test search returns default articles for unknown tickers."""
        articles = await mock_news.search_news(query="UNKNOWN_TICKER_XYZ", num_articles=3)

        assert len(articles) > 0
        # Should return default market news
        assert all(isinstance(a, NewsArticle) for a in articles)

    @pytest.mark.asyncio
    async def test_search_news_respects_num_articles(self, mock_news):
        """Test that num_articles parameter limits results."""
        articles_2 = await mock_news.search_news(query="AAPL", num_articles=2)
        articles_5 = await mock_news.search_news(query="AAPL", num_articles=5)

        assert len(articles_2) <= 2
        assert len(articles_5) <= 5
        assert len(articles_5) >= len(articles_2)

    @pytest.mark.asyncio
    async def test_search_news_date_filter(self, mock_news):
        """Test that from_date and to_date filters work."""
        now = datetime.now(timezone.utc)
        from_date = now - timedelta(days=7)

        articles = await mock_news.search_news(
            query="MSFT",
            num_articles=10,
            from_date=from_date,
        )

        # All articles should be within the date range
        for article in articles:
            assert article.published_at >= from_date

    @pytest.mark.asyncio
    async def test_get_company_news_returns_articles(self, mock_news):
        """Test that get_company_news returns articles."""
        articles = await mock_news.get_company_news(
            company_name="Apple Inc.",
            ticker="AAPL",
            num_articles=3,
        )

        assert isinstance(articles, list)
        assert len(articles) <= 3
        assert all(isinstance(a, NewsArticle) for a in articles)

    @pytest.mark.asyncio
    async def test_get_company_news_ticker_specific(self, mock_news):
        """Test that get_company_news returns ticker-specific articles."""
        tsla_articles = await mock_news.get_company_news(
            company_name="Tesla Inc.",
            ticker="TSLA",
            num_articles=3,
        )
        aapl_articles = await mock_news.get_company_news(
            company_name="Apple Inc.",
            ticker="AAPL",
            num_articles=3,
        )

        # Different tickers should return different articles
        tsla_titles = {a.title for a in tsla_articles}
        aapl_titles = {a.title for a in aapl_articles}
        assert tsla_titles != aapl_titles

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self, mock_news):
        """Test that health_check always returns True for mock."""
        result = await mock_news.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_call_count_tracking(self, mock_news):
        """Test that call count is properly tracked."""
        assert mock_news.get_call_count() == 0

        await mock_news.search_news(query="test")
        assert mock_news.get_call_count() == 1

        await mock_news.get_company_news("Test", "TEST")
        assert mock_news.get_call_count() == 2

        mock_news.reset()
        assert mock_news.get_call_count() == 0

    @pytest.mark.asyncio
    async def test_article_has_required_fields(self, mock_news):
        """Test that returned articles have all required fields."""
        articles = await mock_news.search_news(query="GOOGL", num_articles=1)

        assert len(articles) == 1
        article = articles[0]

        # Check required fields
        assert article.title is not None
        assert isinstance(article.title, str)
        assert len(article.title) > 0

        assert article.url is not None
        assert isinstance(article.url, str)

        assert article.source is not None
        assert isinstance(article.source, str)

        assert article.published_at is not None
        assert isinstance(article.published_at, datetime)

    @pytest.mark.asyncio
    async def test_all_known_tickers_have_data(self, mock_news):
        """Test that all known tickers return specific data."""
        known_tickers = ["TSLA", "AAPL", "GOOGL", "MSFT"]

        for ticker in known_tickers:
            articles = await mock_news.get_company_news(
                company_name=f"{ticker} Company",
                ticker=ticker,
                num_articles=3,
            )
            assert len(articles) > 0, f"No articles for ticker {ticker}"


# =============================================================================
# NewsAPIClient Integration Tests
# =============================================================================


@pytest.fixture
def newsapi_key():
    """Get NewsAPI key from settings, skip if not available."""
    from src.config.settings import get_settings

    settings = get_settings()
    api_key = settings.news.api_key
    if not api_key:
        pytest.skip("NEWS_API_KEY not configured in settings")
    return api_key


@pytest.fixture
async def newsapi_client(newsapi_key):
    """Create and cleanup NewsAPIClient."""
    client = NewsAPIClient(api_key=newsapi_key)
    yield client
    await client.close()


class TestNewsAPIClient:
    """Integration tests for the real NewsAPI client.

    These tests make actual API calls and require NEWS_API_KEY env var.
    They use minimal requests to conserve the free tier quota (100/day).
    """

    @pytest.mark.asyncio
    async def test_search_news_returns_articles(self, newsapi_client):
        """Test that search_news returns real articles."""
        # Use a popular topic to ensure results
        articles = await newsapi_client.search_news(
            query="technology",
            num_articles=2,
        )

        assert isinstance(articles, list)
        assert len(articles) <= 2
        assert all(isinstance(a, NewsArticle) for a in articles)

    @pytest.mark.asyncio
    async def test_search_news_article_fields(self, newsapi_client):
        """Test that real articles have valid fields."""
        articles = await newsapi_client.search_news(
            query="stock market",
            num_articles=1,
        )

        if len(articles) > 0:
            article = articles[0]

            # Validate fields
            assert article.title is not None
            assert len(article.title) > 0

            assert article.url is not None
            assert article.url.startswith("http")

            assert article.source is not None

            assert article.published_at is not None
            assert isinstance(article.published_at, datetime)

    @pytest.mark.asyncio
    async def test_search_news_with_date_range(self, newsapi_client):
        """Test that date filters are applied."""
        now = datetime.now(timezone.utc)
        from_date = now - timedelta(days=7)
        to_date = now

        articles = await newsapi_client.search_news(
            query="business",
            num_articles=2,
            from_date=from_date,
            to_date=to_date,
        )

        # All returned articles should be within date range
        for article in articles:
            assert article.published_at >= from_date - timedelta(days=1)  # Allow 1 day tolerance

    @pytest.mark.asyncio
    async def test_get_company_news(self, newsapi_client):
        """Test fetching news for a specific company."""
        articles = await newsapi_client.get_company_news(
            company_name="Apple",
            ticker="AAPL",
            num_articles=2,
        )

        assert isinstance(articles, list)
        # May or may not have results depending on current news

    @pytest.mark.asyncio
    async def test_health_check(self, newsapi_client):
        """Test that health check works with valid API key."""
        result = await newsapi_client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_error(self):
        """Test that invalid API key raises appropriate error."""
        client = NewsAPIClient(api_key="invalid_key_12345")

        with pytest.raises(NewsRetrievalError) as exc_info:
            await client.search_news(query="test", num_articles=1)

        assert "NewsAPI error" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_with_invalid_key(self):
        """Test that health check fails with invalid API key."""
        client = NewsAPIClient(api_key="invalid_key_12345")
        result = await client.health_check()
        assert result is False
        await client.close()


# =============================================================================
# Contract Tests (Both Implementations)
# =============================================================================


class TestNewsPortContract:
    """Tests that verify both implementations follow the same contract."""

    @pytest.fixture(params=["mock"])
    def news_provider(self, request):
        """Parameterized fixture for testing different implementations."""
        if request.param == "mock":
            return MockNews()
        # Real NewsAPI would be added here if API key is available

    @pytest.mark.asyncio
    async def test_search_news_signature(self, news_provider):
        """Test that search_news accepts all documented parameters."""
        now = datetime.now(timezone.utc)

        # Should not raise any exceptions
        articles = await news_provider.search_news(
            query="test",
            num_articles=2,
            from_date=now - timedelta(days=7),
            to_date=now,
            language="en",
        )

        assert isinstance(articles, list)

    @pytest.mark.asyncio
    async def test_get_company_news_signature(self, news_provider):
        """Test that get_company_news accepts all documented parameters."""
        articles = await news_provider.get_company_news(
            company_name="Test Company",
            ticker="TEST",
            num_articles=2,
        )

        assert isinstance(articles, list)

    @pytest.mark.asyncio
    async def test_health_check_signature(self, news_provider):
        """Test that health_check returns a boolean."""
        result = await news_provider.health_check()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_articles_are_newsarticle_type(self, news_provider):
        """Test that all returned articles are NewsArticle instances."""
        articles = await news_provider.search_news(query="test", num_articles=3)

        for article in articles:
            assert isinstance(article, NewsArticle)

    @pytest.mark.asyncio
    async def test_empty_query_handled(self, news_provider):
        """Test that empty queries are handled gracefully."""
        # Should not crash, may return empty or default results
        articles = await news_provider.search_news(query="", num_articles=2)
        assert isinstance(articles, list)

    @pytest.mark.asyncio
    async def test_zero_articles_requested(self, news_provider):
        """Test that requesting zero articles returns empty list."""
        articles = await news_provider.search_news(query="test", num_articles=0)
        assert isinstance(articles, list)
        assert len(articles) == 0
