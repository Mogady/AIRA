"""
Integration tests for Search Providers.

Tests both MockSearch and DuckDuckGoClient implementations to ensure
they correctly implement the SearchPort interface.
"""

import pytest

from src.adapters.search.mock_search import MockSearch
from src.adapters.search.duckduckgo_client import DuckDuckGoClient
from src.domain.models import WebSearchResult


# =============================================================================
# MockSearch Integration Tests
# =============================================================================


class TestMockSearch:
    """Integration tests for the MockSearch provider."""

    @pytest.fixture
    def mock_search(self):
        """Create a fresh MockSearch instance for each test."""
        return MockSearch()

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_search):
        """Test that search returns a list of WebSearchResult objects."""
        results = await mock_search.search(query="TSLA stock analysis", num_results=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(r, WebSearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_known_ticker(self, mock_search):
        """Test search returns ticker-specific results for known tickers."""
        results = await mock_search.search(query="Tesla TSLA analyst rating", num_results=5)

        assert len(results) > 0
        # TSLA results should mention Tesla in title or snippet
        assert any("Tesla" in r.title or "TSLA" in r.title for r in results)

    @pytest.mark.asyncio
    async def test_search_unknown_ticker(self, mock_search):
        """Test search returns default results for unknown tickers."""
        results = await mock_search.search(query="UNKNOWN_TICKER_XYZ", num_results=3)

        assert len(results) > 0
        # Should return default market news
        assert all(isinstance(r, WebSearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_num_results(self, mock_search):
        """Test that num_results parameter limits results."""
        results_2 = await mock_search.search(query="AAPL", num_results=2)
        results_5 = await mock_search.search(query="AAPL", num_results=5)

        assert len(results_2) <= 2
        assert len(results_5) <= 5

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self, mock_search):
        """Test that health_check always returns True for mock."""
        result = await mock_search.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_call_count_tracking(self, mock_search):
        """Test that call count is properly tracked."""
        assert mock_search.get_call_count() == 0

        await mock_search.search(query="test")
        assert mock_search.get_call_count() == 1

        await mock_search.search(query="another test")
        assert mock_search.get_call_count() == 2

        mock_search.reset()
        assert mock_search.get_call_count() == 0

    @pytest.mark.asyncio
    async def test_query_tracking(self, mock_search):
        """Test that queries are properly tracked."""
        await mock_search.search(query="TSLA analysis")
        await mock_search.search(query="AAPL price target")

        queries = mock_search.get_queries()
        assert len(queries) == 2
        assert "TSLA analysis" in queries
        assert "AAPL price target" in queries

        mock_search.reset()
        assert len(mock_search.get_queries()) == 0

    @pytest.mark.asyncio
    async def test_result_has_required_fields(self, mock_search):
        """Test that returned results have all required fields."""
        results = await mock_search.search(query="GOOGL", num_results=1)

        assert len(results) >= 1
        result = results[0]

        # Check required fields
        assert result.title is not None
        assert isinstance(result.title, str)
        assert len(result.title) > 0

        assert result.url is not None
        assert isinstance(result.url, str)

        assert result.snippet is not None
        assert isinstance(result.snippet, str)

        assert result.source is not None
        assert isinstance(result.source, str)

    @pytest.mark.asyncio
    async def test_all_known_tickers_have_data(self, mock_search):
        """Test that all known tickers return specific data."""
        known_tickers = ["TSLA", "AAPL", "GOOGL", "MSFT"]

        for ticker in known_tickers:
            results = await mock_search.search(query=f"{ticker} stock", num_results=3)
            assert len(results) > 0, f"No results for ticker {ticker}"


# =============================================================================
# DuckDuckGoClient Integration Tests
# =============================================================================


class TestDuckDuckGoClient:
    """Integration tests for the real DuckDuckGo client.

    These tests make actual API calls to DuckDuckGo.
    They use minimal requests to avoid rate limiting.
    """

    @pytest.fixture
    def ddg_client(self):
        """Create DuckDuckGoClient instance."""
        return DuckDuckGoClient(timeout=30)

    @pytest.mark.asyncio
    async def test_search_returns_results(self, ddg_client):
        """Test that search returns real results."""
        # Use a popular topic to ensure results
        results = await ddg_client.search(
            query="Apple stock AAPL",
            num_results=3,
        )

        assert isinstance(results, list)
        # DuckDuckGo may return fewer results than requested
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_result_fields(self, ddg_client):
        """Test that real results have valid fields."""
        results = await ddg_client.search(
            query="Microsoft stock market",
            num_results=2,
        )

        if len(results) > 0:
            result = results[0]

            # Validate fields
            assert result.title is not None
            assert len(result.title) > 0

            assert result.url is not None
            assert result.url.startswith("http")

            assert result.source is not None

    @pytest.mark.asyncio
    async def test_search_for_financial_news(self, ddg_client):
        """Test searching for financial news."""
        results = await ddg_client.search(
            query="Tesla TSLA stock news 2024",
            num_results=3,
        )

        # Should return some results (may be empty if rate limited)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_health_check(self, ddg_client):
        """Test that health check works."""
        result = await ddg_client.health_check()
        # May return False if rate limited, so just check it returns bool
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_empty_query_handled(self, ddg_client):
        """Test that empty queries are handled gracefully."""
        results = await ddg_client.search(query="", num_results=3)
        # Should not crash, may return empty list
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_special_characters_handled(self, ddg_client):
        """Test that special characters in query are handled."""
        results = await ddg_client.search(
            query="Tesla Inc. (TSLA) stock price",
            num_results=2,
        )
        assert isinstance(results, list)


# =============================================================================
# Contract Tests (Both Implementations)
# =============================================================================


class TestSearchPortContract:
    """Tests that verify both implementations follow the same contract."""

    @pytest.fixture(params=["mock"])
    def search_provider(self, request):
        """Parameterized fixture for testing different implementations."""
        if request.param == "mock":
            return MockSearch()
        # Real DuckDuckGo would be added here but skipped by default

    @pytest.mark.asyncio
    async def test_search_signature(self, search_provider):
        """Test that search accepts all documented parameters."""
        # Should not raise any exceptions
        results = await search_provider.search(
            query="test query",
            num_results=5,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_health_check_signature(self, search_provider):
        """Test that health_check returns a boolean."""
        result = await search_provider.health_check()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_results_are_websearchresult_type(self, search_provider):
        """Test that all returned results are WebSearchResult instances."""
        results = await search_provider.search(query="test", num_results=3)

        for result in results:
            assert isinstance(result, WebSearchResult)

    @pytest.mark.asyncio
    async def test_zero_results_requested(self, search_provider):
        """Test that requesting zero results returns empty list."""
        results = await search_provider.search(query="test", num_results=0)
        assert isinstance(results, list)
        assert len(results) == 0
