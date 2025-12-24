"""
DuckDuckGo Search Client - Free web search with no API key required.

This module implements the Search port using DuckDuckGo search,
which provides free, unlimited web searches without requiring an API key.
"""

import asyncio
from typing import List

from src.application.ports.search_port import SearchPort
from src.config.logging import get_logger
from src.domain.models import WebSearchResult


logger = get_logger(__name__)


class DuckDuckGoClient(SearchPort):
    """
    DuckDuckGo search client for free web searches.

    Uses the duckduckgo-search library which doesn't require an API key.
    Searches are run in a thread pool to avoid blocking the event loop.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize the DuckDuckGo client.

        Args:
            timeout: Request timeout in seconds
        """
        self._timeout = timeout

    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[WebSearchResult]:
        """
        Execute a web search using DuckDuckGo.

        Args:
            query: Search query string
            num_results: Maximum number of results to return

        Returns:
            List of WebSearchResult objects
        """
        logger.debug("duckduckgo_search_start", query=query, num_results=num_results)

        try:
            results = await asyncio.to_thread(
                self._search_sync, query, num_results
            )

            logger.info(
                "duckduckgo_search_complete",
                query=query,
                results_count=len(results),
            )

            return results

        except Exception as e:
            logger.error(
                "duckduckgo_search_error",
                query=query,
                error=str(e),
            )
            # Return empty list on error instead of raising
            return []

    def _search_sync(self, query: str, num_results: int) -> List[WebSearchResult]:
        """Synchronous helper to perform DuckDuckGo search."""
        try:
            from ddgs import DDGS
        except ImportError:
            logger.error("duckduckgo_search_not_installed")
            return []

        results = []

        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=num_results))

            for r in search_results:
                url = r.get("href", "")
                # Extract domain from URL
                source = ""
                if url:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        source = parsed.netloc
                    except Exception:
                        source = url.split("/")[2] if len(url.split("/")) > 2 else ""

                results.append(
                    WebSearchResult(
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                        url=url,
                        source=source,
                    )
                )

        except Exception as e:
            logger.warning(
                "duckduckgo_search_partial_error",
                error=str(e),
            )

        return results

    async def health_check(self) -> bool:
        """Check if DuckDuckGo search is available."""
        try:
            results = await self.search("test", num_results=1)
            return len(results) > 0
        except Exception as e:
            logger.error("duckduckgo_health_check_failed", error=str(e))
            return False
