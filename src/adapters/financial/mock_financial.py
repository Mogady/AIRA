"""
Mock Financial Data Provider - Realistic mock financial data for testing.

Provides pre-defined financial data for common tickers.
Supports testing without external financial API calls.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from src.application.ports.financial_port import FinancialPort
from src.config.logging import get_logger
from src.domain.models import FinancialData, QuarterlyRevenue

logger = get_logger(__name__)


# Pre-defined mock financial data for common tickers
MOCK_FINANCIAL_DATA: Dict[str, Dict] = {
    "TSLA": {
        "ticker": "TSLA",
        "company_name": "Tesla, Inc.",
        "current_price": 248.50,
        "price_change_percent": 2.35,
        "market_cap": 789_000_000_000,
        "pe_ratio": 62.5,
        "quarterly_revenue": [
            {"quarter": "Q3 2024", "revenue": 25_180_000_000, "year_over_year_change": 8.2},
            {"quarter": "Q2 2024", "revenue": 24_930_000_000, "year_over_year_change": 7.5},
            {"quarter": "Q1 2024", "revenue": 21_300_000_000, "year_over_year_change": -8.5},
            {"quarter": "Q4 2023", "revenue": 25_170_000_000, "year_over_year_change": 3.0},
        ],
        "fifty_two_week_high": 299.29,
        "fifty_two_week_low": 138.80,
        # New fields
        "gross_margin": 0.178,
        "operating_margin": 0.079,
        "profit_margin": 0.130,
        "total_debt": 5_200_000_000,
        "total_cash": 26_100_000_000,
        "debt_to_equity": 0.11,
        "current_ratio": 1.73,
        "revenue_growth": 0.02,
        "earnings_growth": -0.23,
        "forward_pe": 95.5,
        "peg_ratio": 3.2,
        "price_to_book": 11.2,
        "price_to_sales": 8.1,
        "enterprise_value": 768_000_000_000,
        "ev_to_ebitda": 52.3,
        "analyst_target_mean": 210.00,
        "analyst_target_low": 85.00,
        "analyst_target_high": 380.00,
        "analyst_recommendation": "hold",
        "number_of_analysts": 45,
        "dividend_yield": None,
        "payout_ratio": None,
        "sector": "Consumer Cyclical",
        "industry": "Auto Manufacturers",
        "beta": 2.31,
        "short_ratio": 1.8,
    },
    "AAPL": {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "current_price": 178.25,
        "price_change_percent": 0.85,
        "market_cap": 2_780_000_000_000,
        "pe_ratio": 28.4,
        "quarterly_revenue": [
            {"quarter": "Q4 2024", "revenue": 89_500_000_000, "year_over_year_change": 5.5},
            {"quarter": "Q3 2024", "revenue": 85_780_000_000, "year_over_year_change": 4.9},
            {"quarter": "Q2 2024", "revenue": 90_750_000_000, "year_over_year_change": -4.0},
            {"quarter": "Q1 2024", "revenue": 119_580_000_000, "year_over_year_change": 2.0},
        ],
        "fifty_two_week_high": 199.62,
        "fifty_two_week_low": 164.08,
        # New fields
        "gross_margin": 0.438,
        "operating_margin": 0.297,
        "profit_margin": 0.253,
        "total_debt": 110_000_000_000,
        "total_cash": 65_000_000_000,
        "debt_to_equity": 1.95,
        "current_ratio": 0.99,
        "revenue_growth": 0.05,
        "earnings_growth": 0.08,
        "forward_pe": 26.5,
        "peg_ratio": 2.5,
        "price_to_book": 47.8,
        "price_to_sales": 7.5,
        "enterprise_value": 2_850_000_000_000,
        "ev_to_ebitda": 21.5,
        "analyst_target_mean": 200.00,
        "analyst_target_low": 160.00,
        "analyst_target_high": 250.00,
        "analyst_recommendation": "buy",
        "number_of_analysts": 40,
        "dividend_yield": 0.005,
        "payout_ratio": 0.15,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "beta": 1.28,
        "short_ratio": 1.2,
    },
    "GOOGL": {
        "ticker": "GOOGL",
        "company_name": "Alphabet Inc.",
        "current_price": 141.80,
        "price_change_percent": 1.25,
        "market_cap": 1_780_000_000_000,
        "pe_ratio": 24.2,
        "quarterly_revenue": [
            {"quarter": "Q3 2024", "revenue": 84_700_000_000, "year_over_year_change": 11.0},
            {"quarter": "Q2 2024", "revenue": 81_420_000_000, "year_over_year_change": 14.3},
            {"quarter": "Q1 2024", "revenue": 80_540_000_000, "year_over_year_change": 15.4},
            {"quarter": "Q4 2023", "revenue": 86_310_000_000, "year_over_year_change": 13.5},
        ],
        "fifty_two_week_high": 153.78,
        "fifty_two_week_low": 115.83,
        # New fields
        "gross_margin": 0.572,
        "operating_margin": 0.287,
        "profit_margin": 0.245,
        "total_debt": 28_500_000_000,
        "total_cash": 110_900_000_000,
        "debt_to_equity": 0.10,
        "current_ratio": 2.12,
        "revenue_growth": 0.14,
        "earnings_growth": 0.31,
        "forward_pe": 22.5,
        "peg_ratio": 1.2,
        "price_to_book": 5.8,
        "price_to_sales": 5.5,
        "enterprise_value": 1_700_000_000_000,
        "ev_to_ebitda": 15.2,
        "analyst_target_mean": 175.00,
        "analyst_target_low": 140.00,
        "analyst_target_high": 215.00,
        "analyst_recommendation": "buy",
        "number_of_analysts": 50,
        "dividend_yield": 0.005,
        "payout_ratio": 0.03,
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "beta": 1.05,
        "short_ratio": 1.5,
    },
    "MSFT": {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "current_price": 378.90,
        "price_change_percent": 1.55,
        "market_cap": 2_810_000_000_000,
        "pe_ratio": 34.8,
        "quarterly_revenue": [
            {"quarter": "Q1 FY25", "revenue": 65_590_000_000, "year_over_year_change": 16.0},
            {"quarter": "Q4 FY24", "revenue": 64_730_000_000, "year_over_year_change": 15.2},
            {"quarter": "Q3 FY24", "revenue": 61_860_000_000, "year_over_year_change": 17.0},
            {"quarter": "Q2 FY24", "revenue": 62_020_000_000, "year_over_year_change": 18.0},
        ],
        "fifty_two_week_high": 420.82,
        "fifty_two_week_low": 309.45,
        # New fields
        "gross_margin": 0.697,
        "operating_margin": 0.440,
        "profit_margin": 0.358,
        "total_debt": 75_000_000_000,
        "total_cash": 80_000_000_000,
        "debt_to_equity": 0.42,
        "current_ratio": 1.77,
        "revenue_growth": 0.16,
        "earnings_growth": 0.21,
        "forward_pe": 31.2,
        "peg_ratio": 2.1,
        "price_to_book": 12.5,
        "price_to_sales": 11.8,
        "enterprise_value": 2_805_000_000_000,
        "ev_to_ebitda": 22.8,
        "analyst_target_mean": 450.00,
        "analyst_target_low": 380.00,
        "analyst_target_high": 520.00,
        "analyst_recommendation": "strong_buy",
        "number_of_analysts": 38,
        "dividend_yield": 0.008,
        "payout_ratio": 0.25,
        "sector": "Technology",
        "industry": "Software - Infrastructure",
        "beta": 0.90,
        "short_ratio": 1.1,
    },
    "AMZN": {
        "ticker": "AMZN",
        "company_name": "Amazon.com, Inc.",
        "current_price": 178.45,
        "price_change_percent": -0.45,
        "market_cap": 1_860_000_000_000,
        "pe_ratio": 58.2,
        "quarterly_revenue": [
            {"quarter": "Q3 2024", "revenue": 158_877_000_000, "year_over_year_change": 12.6},
            {"quarter": "Q2 2024", "revenue": 147_977_000_000, "year_over_year_change": 10.1},
            {"quarter": "Q1 2024", "revenue": 143_313_000_000, "year_over_year_change": 12.5},
            {"quarter": "Q4 2023", "revenue": 169_961_000_000, "year_over_year_change": 13.9},
        ],
        "fifty_two_week_high": 201.20,
        "fifty_two_week_low": 118.35,
        # New fields
        "gross_margin": 0.478,
        "operating_margin": 0.092,
        "profit_margin": 0.068,
        "total_debt": 165_000_000_000,
        "total_cash": 88_000_000_000,
        "debt_to_equity": 0.95,
        "current_ratio": 1.05,
        "revenue_growth": 0.12,
        "earnings_growth": 0.55,
        "forward_pe": 38.5,
        "peg_ratio": 1.8,
        "price_to_book": 7.2,
        "price_to_sales": 2.9,
        "enterprise_value": 1_940_000_000_000,
        "ev_to_ebitda": 18.5,
        "analyst_target_mean": 210.00,
        "analyst_target_low": 165.00,
        "analyst_target_high": 260.00,
        "analyst_recommendation": "buy",
        "number_of_analysts": 52,
        "dividend_yield": None,
        "payout_ratio": None,
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
        "beta": 1.15,
        "short_ratio": 1.3,
    },
    "NVDA": {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "current_price": 475.25,
        "price_change_percent": 3.45,
        "market_cap": 1_170_000_000_000,
        "pe_ratio": 65.8,
        "quarterly_revenue": [
            {"quarter": "Q3 FY25", "revenue": 35_100_000_000, "year_over_year_change": 94.0},
            {"quarter": "Q2 FY25", "revenue": 30_040_000_000, "year_over_year_change": 122.0},
            {"quarter": "Q1 FY25", "revenue": 26_040_000_000, "year_over_year_change": 262.0},
            {"quarter": "Q4 FY24", "revenue": 22_100_000_000, "year_over_year_change": 265.0},
        ],
        "fifty_two_week_high": 505.48,
        "fifty_two_week_low": 222.97,
        # New fields
        "gross_margin": 0.748,
        "operating_margin": 0.618,
        "profit_margin": 0.553,
        "total_debt": 11_000_000_000,
        "total_cash": 34_800_000_000,
        "debt_to_equity": 0.41,
        "current_ratio": 4.17,
        "revenue_growth": 1.22,
        "earnings_growth": 1.68,
        "forward_pe": 42.5,
        "peg_ratio": 0.9,
        "price_to_book": 52.3,
        "price_to_sales": 32.8,
        "enterprise_value": 1_145_000_000_000,
        "ev_to_ebitda": 55.2,
        "analyst_target_mean": 540.00,
        "analyst_target_low": 400.00,
        "analyst_target_high": 700.00,
        "analyst_recommendation": "strong_buy",
        "number_of_analysts": 48,
        "dividend_yield": 0.0003,
        "payout_ratio": 0.01,
        "sector": "Technology",
        "industry": "Semiconductors",
        "beta": 1.68,
        "short_ratio": 0.8,
    },
    "META": {
        "ticker": "META",
        "company_name": "Meta Platforms, Inc.",
        "current_price": 512.30,
        "price_change_percent": 0.95,
        "market_cap": 1_310_000_000_000,
        "pe_ratio": 27.5,
        "quarterly_revenue": [
            {"quarter": "Q3 2024", "revenue": 40_589_000_000, "year_over_year_change": 23.0},
            {"quarter": "Q2 2024", "revenue": 39_071_000_000, "year_over_year_change": 22.1},
            {"quarter": "Q1 2024", "revenue": 36_455_000_000, "year_over_year_change": 27.3},
            {"quarter": "Q4 2023", "revenue": 40_111_000_000, "year_over_year_change": 24.7},
        ],
        "fifty_two_week_high": 542.81,
        "fifty_two_week_low": 279.40,
        # New fields
        "gross_margin": 0.812,
        "operating_margin": 0.410,
        "profit_margin": 0.347,
        "total_debt": 37_000_000_000,
        "total_cash": 58_000_000_000,
        "debt_to_equity": 0.32,
        "current_ratio": 2.68,
        "revenue_growth": 0.24,
        "earnings_growth": 0.68,
        "forward_pe": 23.8,
        "peg_ratio": 0.8,
        "price_to_book": 8.1,
        "price_to_sales": 8.4,
        "enterprise_value": 1_290_000_000_000,
        "ev_to_ebitda": 15.8,
        "analyst_target_mean": 580.00,
        "analyst_target_low": 450.00,
        "analyst_target_high": 700.00,
        "analyst_recommendation": "buy",
        "number_of_analysts": 55,
        "dividend_yield": 0.004,
        "payout_ratio": 0.05,
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "beta": 1.25,
        "short_ratio": 1.4,
    },
}

# Default data for unknown tickers
DEFAULT_FINANCIAL_DATA = {
    "current_price": 100.00,
    "price_change_percent": 0.50,
    "market_cap": 50_000_000_000,
    "pe_ratio": 20.0,
    "quarterly_revenue": [
        {"quarter": "Q3 2024", "revenue": 5_000_000_000, "year_over_year_change": 5.0},
        {"quarter": "Q2 2024", "revenue": 4_800_000_000, "year_over_year_change": 4.5},
        {"quarter": "Q1 2024", "revenue": 4_600_000_000, "year_over_year_change": 4.0},
        {"quarter": "Q4 2023", "revenue": 4_400_000_000, "year_over_year_change": 3.5},
    ],
    "fifty_two_week_high": 120.00,
    "fifty_two_week_low": 80.00,
    # Default new fields
    "gross_margin": 0.35,
    "operating_margin": 0.15,
    "profit_margin": 0.10,
    "total_debt": 10_000_000_000,
    "total_cash": 5_000_000_000,
    "debt_to_equity": 0.50,
    "current_ratio": 1.50,
    "revenue_growth": 0.05,
    "earnings_growth": 0.05,
    "forward_pe": 18.0,
    "peg_ratio": 1.5,
    "price_to_book": 3.0,
    "price_to_sales": 2.0,
    "enterprise_value": 55_000_000_000,
    "ev_to_ebitda": 12.0,
    "analyst_target_mean": 110.00,
    "analyst_target_low": 85.00,
    "analyst_target_high": 130.00,
    "analyst_recommendation": "hold",
    "number_of_analysts": 10,
    "dividend_yield": 0.02,
    "payout_ratio": 0.30,
    "sector": "Technology",
    "industry": "Software",
    "beta": 1.0,
    "short_ratio": 2.0,
}

# Company names for common tickers
TICKER_TO_COMPANY: Dict[str, str] = {
    "TSLA": "Tesla, Inc.",
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms, Inc.",
    "NFLX": "Netflix, Inc.",
    "INTC": "Intel Corporation",
    "AMD": "Advanced Micro Devices, Inc.",
    "CRM": "Salesforce, Inc.",
    "ORCL": "Oracle Corporation",
    "IBM": "International Business Machines Corporation",
    "CSCO": "Cisco Systems, Inc.",
    "ADBE": "Adobe Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "SQ": "Block, Inc.",
    "SHOP": "Shopify Inc.",
    "UBER": "Uber Technologies, Inc.",
    "LYFT": "Lyft, Inc.",
    "TWTR": "X Corp.",
    "SNAP": "Snap Inc.",
    "PINS": "Pinterest, Inc.",
    "ZM": "Zoom Video Communications, Inc.",
    "DOCU": "DocuSign, Inc.",
    "COIN": "Coinbase Global, Inc.",
    "PLTR": "Palantir Technologies Inc.",
}


class MockFinancial(FinancialPort):
    """
    Mock financial data provider with realistic pre-defined data.

    Provides consistent, realistic financial data for testing
    without requiring external API calls.
    """

    def __init__(self):
        self._call_count = 0

    async def get_company_data(
        self,
        ticker: str,
    ) -> FinancialData:
        """Get comprehensive financial data (mock implementation)."""
        self._call_count += 1

        ticker_upper = ticker.upper()

        logger.debug(
            "mock_financial_get_data",
            ticker=ticker_upper,
            call_count=self._call_count,
        )

        # Get data for ticker or use default
        if ticker_upper in MOCK_FINANCIAL_DATA:
            data = MOCK_FINANCIAL_DATA[ticker_upper]
        else:
            # Create default data with the requested ticker
            data = {
                "ticker": ticker_upper,
                "company_name": TICKER_TO_COMPANY.get(ticker_upper, f"{ticker_upper} Inc."),
                **DEFAULT_FINANCIAL_DATA,
            }

        # Convert quarterly revenue to proper format
        quarterly_revenue = [
            QuarterlyRevenue(**rev) for rev in data.get("quarterly_revenue", [])
        ]

        return FinancialData(
            ticker=data["ticker"],
            company_name=data["company_name"],
            current_price=data.get("current_price"),
            price_change_percent=data.get("price_change_percent"),
            market_cap=data.get("market_cap"),
            pe_ratio=data.get("pe_ratio"),
            quarterly_revenue=quarterly_revenue,
            fifty_two_week_high=data.get("fifty_two_week_high"),
            fifty_two_week_low=data.get("fifty_two_week_low"),
            data_timestamp=datetime.now(timezone.utc),
            # New fields
            gross_margin=data.get("gross_margin"),
            operating_margin=data.get("operating_margin"),
            profit_margin=data.get("profit_margin"),
            total_debt=data.get("total_debt"),
            total_cash=data.get("total_cash"),
            debt_to_equity=data.get("debt_to_equity"),
            current_ratio=data.get("current_ratio"),
            revenue_growth=data.get("revenue_growth"),
            earnings_growth=data.get("earnings_growth"),
            forward_pe=data.get("forward_pe"),
            peg_ratio=data.get("peg_ratio"),
            price_to_book=data.get("price_to_book"),
            price_to_sales=data.get("price_to_sales"),
            enterprise_value=data.get("enterprise_value"),
            ev_to_ebitda=data.get("ev_to_ebitda"),
            analyst_target_mean=data.get("analyst_target_mean"),
            analyst_target_low=data.get("analyst_target_low"),
            analyst_target_high=data.get("analyst_target_high"),
            analyst_recommendation=data.get("analyst_recommendation"),
            number_of_analysts=data.get("number_of_analysts"),
            dividend_yield=data.get("dividend_yield"),
            payout_ratio=data.get("payout_ratio"),
            sector=data.get("sector"),
            industry=data.get("industry"),
            beta=data.get("beta"),
            short_ratio=data.get("short_ratio"),
        )

    async def get_stock_price(
        self,
        ticker: str,
    ) -> Optional[float]:
        """Get current stock price (mock implementation)."""
        self._call_count += 1

        ticker_upper = ticker.upper()

        if ticker_upper in MOCK_FINANCIAL_DATA:
            return MOCK_FINANCIAL_DATA[ticker_upper].get("current_price")

        return DEFAULT_FINANCIAL_DATA["current_price"]

    async def get_company_name(
        self,
        ticker: str,
    ) -> Optional[str]:
        """Get company name from ticker (mock implementation)."""
        self._call_count += 1

        ticker_upper = ticker.upper()

        # First check full financial data
        if ticker_upper in MOCK_FINANCIAL_DATA:
            return MOCK_FINANCIAL_DATA[ticker_upper]["company_name"]

        # Then check ticker-to-company mapping
        if ticker_upper in TICKER_TO_COMPANY:
            return TICKER_TO_COMPANY[ticker_upper]

        # Return a generic name
        return f"{ticker_upper} Inc."

    async def health_check(self) -> bool:
        """Mock is always healthy."""
        return True

    def get_call_count(self) -> int:
        """Get the number of calls made."""
        return self._call_count

    def reset(self) -> None:
        """Reset the mock state."""
        self._call_count = 0
