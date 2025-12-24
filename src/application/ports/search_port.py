"""
Search Port - Abstract interface for web search providers.

This module defines the contract for web search implementations
that can be used by the agent to gather additional research context.
"""

from abc import ABC, abstractmethod
from typing import List

from src.domain.models import WebSearchResult


class SearchPort(ABC):
    """
    Abstract base class for web search providers.

    Implementations should handle:
    - Web search queries
    - Result parsing and normalization
    - Error handling and rate limiting
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[WebSearchResult]:
        """
        Execute a web search and return results.

        Args:
            query: Search query string
            num_results: Maximum number of results to return

        Returns:
            List of WebSearchResult objects
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the search provider is available.

        Returns:
            True if the provider is healthy, False otherwise
        """
        pass
