"""
News Retriever Tool - Fetches recent news articles about a company.

This tool queries the news provider (mock or NewsAPI) to retrieve
relevant news articles for sentiment analysis.
"""

import time
from typing import Any, Dict, Optional

from src.application.ports.news_port import NewsPort
from src.config.logging import get_logger
from src.domain.models import NewsRetrieverInput, NewsRetrieverOutput

logger = get_logger(__name__)


class NewsRetrieverTool:
    """
    Tool for retrieving news articles about a company.

    Wraps the NewsPort to provide a consistent interface
    for the LangGraph agent.
    """

    name: str = "news_retriever"
    description: str = (
        "Retrieves recent news articles about a company. "
        "Use this tool to gather news coverage for sentiment analysis. "
        "Input should include the company name and stock ticker."
    )

    # JSON Schema for function calling
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "company": {
                "type": "string",
                "description": "The company name to search for",
            },
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., TSLA, AAPL)",
            },
            "num_articles": {
                "type": "integer",
                "description": "Number of articles to retrieve (default: 5)",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["company", "ticker"],
    }

    def __init__(self, news_provider: NewsPort):
        """
        Initialize the news retriever tool.

        Args:
            news_provider: The news provider implementation to use
        """
        self._news_provider = news_provider

    async def execute(
        self,
        company: str,
        ticker: str,
        num_articles: int = 5,
    ) -> NewsRetrieverOutput:
        """
        Execute the news retrieval.

        Args:
            company: Company name to search for
            ticker: Stock ticker symbol
            num_articles: Number of articles to retrieve

        Returns:
            NewsRetrieverOutput with retrieved articles
        """
        start_time = time.time()

        logger.info(
            "news_retriever_start",
            company=company,
            ticker=ticker,
            num_articles=num_articles,
        )

        try:
            # Validate input
            input_data = NewsRetrieverInput(
                company=company,
                ticker=ticker,
                num_articles=num_articles,
            )

            # Fetch news articles
            articles = await self._news_provider.get_company_news(
                company_name=input_data.company,
                ticker=input_data.ticker,
                num_articles=input_data.num_articles,
            )

            # Build the query string that was used
            query_used = f"{input_data.company} {input_data.ticker} stock"

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "news_retriever_complete",
                ticker=ticker,
                articles_found=len(articles),
                duration_ms=duration_ms,
            )

            return NewsRetrieverOutput(
                articles=articles,
                query_used=query_used,
                total_results=len(articles),
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "news_retriever_error",
                ticker=ticker,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise

    async def execute_from_dict(self, params: Dict[str, Any]) -> NewsRetrieverOutput:
        """
        Execute from dictionary parameters (for LangGraph integration).

        Args:
            params: Dictionary with company, ticker, and optional num_articles

        Returns:
            NewsRetrieverOutput with retrieved articles
        """
        return await self.execute(
            company=params["company"],
            ticker=params["ticker"],
            num_articles=params.get("num_articles", 5),
        )

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
