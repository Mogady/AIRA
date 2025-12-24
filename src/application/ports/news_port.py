"""
News Provider Port - Abstract interface for news data retrieval.

This port defines the contract for news data providers.
Implementations must support company news search.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from src.domain.models import NewsArticle


class NewsPort(ABC):
    """
    Abstract interface for news providers.

    Implementations:
    - NewsAPIClient: NewsAPI.org integration
    - MockNews: Mock implementation for testing
    """

    @abstractmethod
    async def search_news(
        self,
        query: str,
        num_articles: int = 5,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
    ) -> List[NewsArticle]:
        """
        Search for news articles.

        Args:
            query: Search query (company name, ticker, keywords)
            num_articles: Number of articles to retrieve
            from_date: Filter articles from this date
            to_date: Filter articles until this date
            language: Article language code

        Returns:
            List of NewsArticle objects
        """
        pass

    @abstractmethod
    async def get_company_news(
        self,
        company_name: str,
        ticker: str,
        num_articles: int = 5,
    ) -> List[NewsArticle]:
        """
        Get news specifically about a company.

        Args:
            company_name: Full company name
            ticker: Stock ticker symbol
            num_articles: Number of articles to retrieve

        Returns:
            List of NewsArticle objects
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the news provider is healthy.

        Returns:
            True if provider is available
        """
        pass
