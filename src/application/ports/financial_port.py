"""
Financial Data Provider Port - Abstract interface for financial data retrieval.

This port defines the contract for financial data providers.
Implementations must support stock data and company financials.
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.domain.models import FinancialData


class FinancialPort(ABC):
    """
    Abstract interface for financial data providers.

    Implementations:
    - MockFinancial: Mock implementation with realistic data
    - YFinanceClient: Optional real data via yfinance
    """

    @abstractmethod
    async def get_company_data(
        self,
        ticker: str,
    ) -> FinancialData:
        """
        Get comprehensive financial data for a company.

        Args:
            ticker: Stock ticker symbol (e.g., "TSLA", "AAPL")

        Returns:
            FinancialData object with stock price, market cap, revenue, etc.
        """
        pass

    @abstractmethod
    async def get_stock_price(
        self,
        ticker: str,
    ) -> Optional[float]:
        """
        Get current stock price.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current stock price or None if unavailable
        """
        pass

    @abstractmethod
    async def get_company_name(
        self,
        ticker: str,
    ) -> Optional[str]:
        """
        Get company name from ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company name or None if not found
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the financial data provider is healthy.

        Returns:
            True if provider is available
        """
        pass
