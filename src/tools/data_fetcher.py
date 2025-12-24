"""
Data Fetcher Tool - Fetches financial data for a company.

This tool queries the financial data provider (mock or real)
to retrieve stock prices, market cap, and revenue data.
"""

import time
from typing import Any, Dict

from src.application.ports.financial_port import FinancialPort
from src.config.logging import get_logger
from src.domain.models import DataFetcherInput, FinancialData

logger = get_logger(__name__)


class DataFetcherTool:
    """
    Tool for fetching financial data about a company.

    Wraps the FinancialPort to provide a consistent interface
    for the LangGraph agent.
    """

    name: str = "data_fetcher"
    description: str = (
        "Fetches financial data for a company including stock price, "
        "market cap, P/E ratio, and recent quarterly revenue. "
        "Input should be the stock ticker symbol."
    )

    # JSON Schema for function calling
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., TSLA, AAPL, GOOGL)",
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, financial_provider: FinancialPort):
        """
        Initialize the data fetcher tool.

        Args:
            financial_provider: The financial data provider implementation to use
        """
        self._financial_provider = financial_provider

    async def execute(
        self,
        ticker: str,
    ) -> FinancialData:
        """
        Execute the financial data fetch.

        Args:
            ticker: Stock ticker symbol

        Returns:
            FinancialData with comprehensive financial information
        """
        start_time = time.time()

        logger.info(
            "data_fetcher_start",
            ticker=ticker,
        )

        try:
            # Validate input
            input_data = DataFetcherInput(ticker=ticker)

            # Fetch financial data
            data = await self._financial_provider.get_company_data(
                ticker=input_data.ticker
            )

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "data_fetcher_complete",
                ticker=ticker,
                current_price=data.current_price,
                market_cap=data.market_cap,
                duration_ms=duration_ms,
            )

            return data

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "data_fetcher_error",
                ticker=ticker,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise

    async def execute_from_dict(self, params: Dict[str, Any]) -> FinancialData:
        """
        Execute from dictionary parameters (for LangGraph integration).

        Args:
            params: Dictionary with ticker

        Returns:
            FinancialData with financial information
        """
        return await self.execute(ticker=params["ticker"])

    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Get the tool definition for LLM function calling.

        Returns:
            Tool definition dictionary compatible with OpenAI/Claude format
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
