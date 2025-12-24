"""
Integration tests for Financial Data Providers.

Tests both MockFinancial and YFinanceClient implementations to ensure
they correctly implement the FinancialPort interface.
"""

from datetime import datetime, timezone

import pytest

from src.adapters.financial.mock_financial import MockFinancial
from src.adapters.financial.yfinance_client import YFinanceClient
from src.domain.exceptions import FinancialDataError
from src.domain.models import FinancialData, QuarterlyRevenue


# =============================================================================
# MockFinancial Integration Tests
# =============================================================================


class TestMockFinancial:
    """Integration tests for the MockFinancial provider."""

    @pytest.fixture
    def mock_financial(self):
        """Create a fresh MockFinancial instance for each test."""
        return MockFinancial()

    @pytest.mark.asyncio
    async def test_get_company_data_returns_financial_data(self, mock_financial):
        """Test that get_company_data returns a FinancialData object."""
        data = await mock_financial.get_company_data(ticker="TSLA")

        assert isinstance(data, FinancialData)
        assert data.ticker == "TSLA"

    @pytest.mark.asyncio
    async def test_get_company_data_known_ticker(self, mock_financial):
        """Test get_company_data returns ticker-specific data for known tickers."""
        data = await mock_financial.get_company_data(ticker="AAPL")

        assert data.ticker == "AAPL"
        assert data.company_name == "Apple Inc."
        assert data.current_price is not None
        assert data.current_price > 0

    @pytest.mark.asyncio
    async def test_get_company_data_unknown_ticker(self, mock_financial):
        """Test get_company_data returns default data for unknown tickers."""
        data = await mock_financial.get_company_data(ticker="UNKNOWN_XYZ")

        assert data.ticker == "UNKNOWN_XYZ"
        assert data.company_name == "UNKNOWN_XYZ Inc."
        assert data.current_price is not None

    @pytest.mark.asyncio
    async def test_get_company_data_case_insensitive(self, mock_financial):
        """Test that ticker lookup is case-insensitive."""
        data_lower = await mock_financial.get_company_data(ticker="tsla")
        data_upper = await mock_financial.get_company_data(ticker="TSLA")

        assert data_lower.ticker == data_upper.ticker
        assert data_lower.company_name == data_upper.company_name

    @pytest.mark.asyncio
    async def test_get_stock_price_returns_float(self, mock_financial):
        """Test that get_stock_price returns a float."""
        price = await mock_financial.get_stock_price(ticker="GOOGL")

        assert isinstance(price, float)
        assert price > 0

    @pytest.mark.asyncio
    async def test_get_stock_price_known_ticker(self, mock_financial):
        """Test get_stock_price returns specific price for known tickers."""
        tsla_price = await mock_financial.get_stock_price(ticker="TSLA")
        aapl_price = await mock_financial.get_stock_price(ticker="AAPL")

        # Different tickers should have different prices
        assert tsla_price != aapl_price

    @pytest.mark.asyncio
    async def test_get_company_name_returns_string(self, mock_financial):
        """Test that get_company_name returns a string."""
        name = await mock_financial.get_company_name(ticker="MSFT")

        assert isinstance(name, str)
        assert len(name) > 0

    @pytest.mark.asyncio
    async def test_get_company_name_known_ticker(self, mock_financial):
        """Test get_company_name returns correct name for known tickers."""
        name = await mock_financial.get_company_name(ticker="NVDA")

        assert name == "NVIDIA Corporation"

    @pytest.mark.asyncio
    async def test_get_company_name_unknown_ticker(self, mock_financial):
        """Test get_company_name returns generic name for unknown tickers."""
        name = await mock_financial.get_company_name(ticker="XYZ123")

        assert "XYZ123" in name

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self, mock_financial):
        """Test that health_check always returns True for mock."""
        result = await mock_financial.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_call_count_tracking(self, mock_financial):
        """Test that call count is properly tracked."""
        assert mock_financial.get_call_count() == 0

        await mock_financial.get_company_data("TSLA")
        assert mock_financial.get_call_count() == 1

        await mock_financial.get_stock_price("AAPL")
        assert mock_financial.get_call_count() == 2

        await mock_financial.get_company_name("GOOGL")
        assert mock_financial.get_call_count() == 3

        mock_financial.reset()
        assert mock_financial.get_call_count() == 0

    @pytest.mark.asyncio
    async def test_financial_data_has_required_fields(self, mock_financial):
        """Test that returned FinancialData has all required fields."""
        data = await mock_financial.get_company_data(ticker="META")

        # Check required fields
        assert data.ticker is not None
        assert isinstance(data.ticker, str)

        assert data.company_name is not None
        assert isinstance(data.company_name, str)

        assert data.data_timestamp is not None
        assert isinstance(data.data_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_financial_data_has_optional_fields(self, mock_financial):
        """Test that FinancialData includes optional financial metrics."""
        data = await mock_financial.get_company_data(ticker="TSLA")

        # Check optional fields are populated for known tickers
        assert data.current_price is not None
        assert data.price_change_percent is not None
        assert data.market_cap is not None
        assert data.pe_ratio is not None
        assert data.fifty_two_week_high is not None
        assert data.fifty_two_week_low is not None

    @pytest.mark.asyncio
    async def test_quarterly_revenue_structure(self, mock_financial):
        """Test that quarterly_revenue has correct structure."""
        data = await mock_financial.get_company_data(ticker="MSFT")

        assert isinstance(data.quarterly_revenue, list)
        assert len(data.quarterly_revenue) > 0

        for quarter in data.quarterly_revenue:
            assert isinstance(quarter, QuarterlyRevenue)
            assert quarter.quarter is not None
            assert quarter.revenue is not None
            assert quarter.revenue > 0

    @pytest.mark.asyncio
    async def test_all_known_tickers_have_data(self, mock_financial):
        """Test that all known tickers return specific data."""
        known_tickers = ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "META"]

        for ticker in known_tickers:
            data = await mock_financial.get_company_data(ticker=ticker)
            assert data.ticker == ticker, f"Ticker mismatch for {ticker}"
            assert data.current_price is not None, f"No price for {ticker}"


# =============================================================================
# YFinanceClient Integration Tests
# =============================================================================


@pytest.fixture
def yfinance_enabled():
    """Check if yfinance provider is enabled in settings."""
    from src.config.settings import get_settings

    settings = get_settings()
    if settings.financial.provider != "yfinance":
        pytest.skip("YFinance provider not enabled in settings (set financial.provider='yfinance')")
    return True


@pytest.fixture
def yfinance_client():
    """Create YFinanceClient instance."""
    return YFinanceClient(timeout=30)


class TestYFinanceClient:
    """Integration tests for the real YFinance client.

    These tests make actual API calls to Yahoo Finance.
    They use well-known tickers to ensure data availability.
    """

    @pytest.mark.asyncio
    async def test_get_company_data_returns_financial_data(self, yfinance_client):
        """Test that get_company_data returns real financial data."""
        data = await yfinance_client.get_company_data(ticker="AAPL")

        assert isinstance(data, FinancialData)
        assert data.ticker == "AAPL"
        assert data.company_name is not None
        assert "Apple" in data.company_name

    @pytest.mark.asyncio
    async def test_get_company_data_has_price(self, yfinance_client):
        """Test that real data includes current price."""
        data = await yfinance_client.get_company_data(ticker="MSFT")

        assert data.current_price is not None
        assert data.current_price > 0

    @pytest.mark.asyncio
    async def test_get_company_data_has_market_metrics(self, yfinance_client):
        """Test that real data includes market metrics."""
        data = await yfinance_client.get_company_data(ticker="GOOGL")

        # These should be available for major tickers
        assert data.market_cap is not None
        assert data.market_cap > 0

    @pytest.mark.asyncio
    async def test_get_company_data_has_52_week_range(self, yfinance_client):
        """Test that real data includes 52-week range."""
        data = await yfinance_client.get_company_data(ticker="TSLA")

        assert data.fifty_two_week_high is not None
        assert data.fifty_two_week_low is not None
        assert data.fifty_two_week_high >= data.fifty_two_week_low

    @pytest.mark.asyncio
    async def test_get_stock_price_returns_float(self, yfinance_client):
        """Test that get_stock_price returns a float."""
        price = await yfinance_client.get_stock_price(ticker="AAPL")

        assert price is not None
        assert isinstance(price, float)
        assert price > 0

    @pytest.mark.asyncio
    async def test_get_stock_price_different_tickers(self, yfinance_client):
        """Test getting prices for different tickers."""
        aapl_price = await yfinance_client.get_stock_price(ticker="AAPL")
        msft_price = await yfinance_client.get_stock_price(ticker="MSFT")

        assert aapl_price is not None
        assert msft_price is not None
        # Prices should be different (extremely unlikely to be identical)
        # But we don't assert this since it's theoretically possible

    @pytest.mark.asyncio
    async def test_get_company_name_returns_string(self, yfinance_client):
        """Test that get_company_name returns a valid string."""
        name = await yfinance_client.get_company_name(ticker="NVDA")

        assert name is not None
        assert isinstance(name, str)
        assert "NVIDIA" in name or "Nvidia" in name

    @pytest.mark.asyncio
    async def test_get_company_name_known_companies(self, yfinance_client):
        """Test get_company_name for well-known companies."""
        test_cases = [
            ("AAPL", "Apple"),
            ("MSFT", "Microsoft"),
            ("GOOGL", "Alphabet"),
        ]

        for ticker, expected_substring in test_cases:
            name = await yfinance_client.get_company_name(ticker=ticker)
            assert name is not None, f"No name returned for {ticker}"
            assert expected_substring in name, f"Expected '{expected_substring}' in '{name}' for {ticker}"

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self, yfinance_client):
        """Test that health_check returns True when Yahoo Finance is available."""
        result = await yfinance_client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_ticker_handling(self, yfinance_client):
        """Test that invalid tickers are handled gracefully."""
        # YFinance typically returns empty/default data for invalid tickers
        # rather than raising an error
        data = await yfinance_client.get_company_data(ticker="INVALIDTICKER12345")

        # Should still return a FinancialData object
        assert isinstance(data, FinancialData)
        assert data.ticker == "INVALIDTICKER12345"

    @pytest.mark.asyncio
    async def test_data_timestamp_is_recent(self, yfinance_client):
        """Test that data_timestamp is set to current time."""
        data = await yfinance_client.get_company_data(ticker="AAPL")

        now = datetime.now(timezone.utc)
        # Timestamp should be within the last minute
        time_diff = (now - data.data_timestamp).total_seconds()
        assert time_diff < 60, f"Timestamp is {time_diff} seconds old"


# =============================================================================
# Contract Tests (Both Implementations)
# =============================================================================


class TestFinancialPortContract:
    """Tests that verify both implementations follow the same contract."""

    @pytest.fixture(params=["mock"])
    def financial_provider(self, request):
        """Parameterized fixture for testing different implementations."""
        if request.param == "mock":
            return MockFinancial()
        # Real YFinance would be added here if enabled

    @pytest.mark.asyncio
    async def test_get_company_data_signature(self, financial_provider):
        """Test that get_company_data accepts ticker parameter."""
        data = await financial_provider.get_company_data(ticker="TEST")

        assert isinstance(data, FinancialData)
        assert data.ticker is not None

    @pytest.mark.asyncio
    async def test_get_stock_price_signature(self, financial_provider):
        """Test that get_stock_price returns Optional[float]."""
        price = await financial_provider.get_stock_price(ticker="TEST")

        assert price is None or isinstance(price, float)

    @pytest.mark.asyncio
    async def test_get_company_name_signature(self, financial_provider):
        """Test that get_company_name returns Optional[str]."""
        name = await financial_provider.get_company_name(ticker="TEST")

        assert name is None or isinstance(name, str)

    @pytest.mark.asyncio
    async def test_health_check_signature(self, financial_provider):
        """Test that health_check returns a boolean."""
        result = await financial_provider.health_check()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_financial_data_is_correct_type(self, financial_provider):
        """Test that get_company_data returns FinancialData instance."""
        data = await financial_provider.get_company_data(ticker="AAPL")

        assert isinstance(data, FinancialData)

    @pytest.mark.asyncio
    async def test_quarterly_revenue_is_list(self, financial_provider):
        """Test that quarterly_revenue is always a list."""
        data = await financial_provider.get_company_data(ticker="MSFT")

        assert isinstance(data.quarterly_revenue, list)
        for item in data.quarterly_revenue:
            assert isinstance(item, QuarterlyRevenue)

    @pytest.mark.asyncio
    async def test_empty_ticker_handled(self, financial_provider):
        """Test that empty tickers are handled gracefully."""
        # Should not crash
        data = await financial_provider.get_company_data(ticker="")
        assert isinstance(data, FinancialData)

    @pytest.mark.asyncio
    async def test_ticker_is_uppercased(self, financial_provider):
        """Test that returned ticker is uppercased."""
        data = await financial_provider.get_company_data(ticker="aapl")
        assert data.ticker == data.ticker.upper()
