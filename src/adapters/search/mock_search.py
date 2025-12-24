"""
Mock Search Provider - Realistic mock search results for testing.

Provides pre-defined search results for common tickers.
Supports testing without external API calls.
"""

from typing import Dict, List

from src.application.ports.search_port import SearchPort
from src.config.logging import get_logger
from src.domain.models import WebSearchResult

logger = get_logger(__name__)


# Pre-defined mock search results for common tickers
MOCK_SEARCH_RESULTS: Dict[str, List[Dict]] = {
    "TSLA": [
        {
            "title": "Tesla Q3 2024 Earnings: What Analysts Are Saying",
            "snippet": "Tesla reported strong Q3 earnings with revenue beating expectations. Analysts remain bullish on the EV maker's long-term prospects despite near-term challenges.",
            "url": "https://seekingalpha.com/article/tesla-q3-earnings",
            "source": "seekingalpha.com",
        },
        {
            "title": "TSLA Stock: Morgan Stanley Raises Price Target",
            "snippet": "Morgan Stanley analyst Adam Jonas raised Tesla price target to $310 from $265, citing improved margins and strong demand for Cybertruck.",
            "url": "https://www.cnbc.com/tesla-price-target",
            "source": "cnbc.com",
        },
        {
            "title": "Tesla vs BYD: The EV Competition Heats Up",
            "snippet": "Tesla faces increasing competition from BYD in China. Market share battle intensifies as both companies launch new models.",
            "url": "https://www.bloomberg.com/tesla-byd-competition",
            "source": "bloomberg.com",
        },
        {
            "title": "Tesla Cybertruck Production Ramps Up",
            "snippet": "Tesla has significantly increased Cybertruck production at Gigafactory Texas. Delivery times have shortened as production efficiency improves.",
            "url": "https://www.reuters.com/tesla-cybertruck",
            "source": "reuters.com",
        },
    ],
    "AAPL": [
        {
            "title": "Apple iPhone 16 Sales Exceed Expectations",
            "snippet": "Apple's iPhone 16 launch is outperforming analyst expectations with strong demand in key markets including China and Europe.",
            "url": "https://www.bloomberg.com/apple-iphone-16",
            "source": "bloomberg.com",
        },
        {
            "title": "AAPL: Warren Buffett's Berkshire Trims Apple Stake",
            "snippet": "Berkshire Hathaway reduced its Apple holdings by 25% in Q3, though it remains the largest public position in the portfolio.",
            "url": "https://finance.yahoo.com/buffett-apple",
            "source": "finance.yahoo.com",
        },
        {
            "title": "Apple Vision Pro: Analyst Ratings Update",
            "snippet": "Wall Street analysts are mixed on Apple Vision Pro's long-term potential. Some see it as a game-changer while others cite high price concerns.",
            "url": "https://www.tipranks.com/apple-vision-pro",
            "source": "tipranks.com",
        },
    ],
    "GOOGL": [
        {
            "title": "Alphabet AI Investments Paying Off",
            "snippet": "Google's massive AI investments are beginning to show returns with improved search and cloud revenue growth.",
            "url": "https://www.marketwatch.com/alphabet-ai",
            "source": "marketwatch.com",
        },
        {
            "title": "GOOGL Stock Analysis: Buy, Hold or Sell?",
            "snippet": "Analysts debate Alphabet's valuation amid regulatory concerns and AI competition. Most maintain Buy ratings with targets around $160.",
            "url": "https://www.fool.com/googl-analysis",
            "source": "fool.com",
        },
    ],
    "MSFT": [
        {
            "title": "Microsoft Azure Growth Accelerates",
            "snippet": "Microsoft Cloud revenue grew 22% YoY with Azure leading the way. AI services are driving enterprise adoption.",
            "url": "https://www.cnbc.com/microsoft-azure",
            "source": "cnbc.com",
        },
        {
            "title": "MSFT Price Target Raised by Goldman Sachs",
            "snippet": "Goldman Sachs raised Microsoft price target to $450 citing strong cloud momentum and Copilot adoption.",
            "url": "https://www.barrons.com/microsoft-price-target",
            "source": "barrons.com",
        },
    ],
}

# Default results for unknown queries
DEFAULT_SEARCH_RESULTS = [
    {
        "title": "Stock Market Analysis: Key Trends to Watch",
        "snippet": "Market analysts highlight key sectors and trends for investors to monitor in the current economic environment.",
        "url": "https://www.marketwatch.com/general-analysis",
        "source": "marketwatch.com",
    },
    {
        "title": "Investment Outlook: Expert Perspectives",
        "snippet": "Financial experts share their outlook on market conditions and investment strategies for the coming quarters.",
        "url": "https://www.bloomberg.com/investment-outlook",
        "source": "bloomberg.com",
    },
]


class MockSearch(SearchPort):
    """
    Mock search provider with realistic pre-defined results.

    Provides consistent, realistic search results for testing
    without requiring external API calls.
    """

    def __init__(self, latency_ms: int = 0):
        """
        Initialize the mock search provider.

        Args:
            latency_ms: Simulated latency in milliseconds (for testing)
        """
        self._call_count = 0
        self._latency_ms = latency_ms
        self._queries: List[str] = []

    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[WebSearchResult]:
        """
        Execute a mock web search.

        Args:
            query: Search query string
            num_results: Maximum number of results to return

        Returns:
            List of WebSearchResult objects
        """
        self._call_count += 1
        self._queries.append(query)

        logger.debug(
            "mock_search_execute",
            query=query,
            num_results=num_results,
            call_count=self._call_count,
        )

        # Simulate latency if configured
        if self._latency_ms > 0:
            import asyncio
            await asyncio.sleep(self._latency_ms / 1000)

        # Find matching results based on ticker in query
        query_upper = query.upper()
        results_data = None

        for ticker in MOCK_SEARCH_RESULTS:
            if ticker in query_upper:
                results_data = MOCK_SEARCH_RESULTS[ticker]
                break

        if results_data is None:
            results_data = DEFAULT_SEARCH_RESULTS

        # Convert to WebSearchResult and limit
        results = [
            WebSearchResult(
                title=r["title"],
                snippet=r["snippet"],
                url=r["url"],
                source=r["source"],
            )
            for r in results_data[:num_results]
        ]

        return results

    async def health_check(self) -> bool:
        """Mock is always healthy."""
        return True

    def get_call_count(self) -> int:
        """Get the number of calls made."""
        return self._call_count

    def get_queries(self) -> List[str]:
        """Get all queries made."""
        return self._queries.copy()

    def reset(self) -> None:
        """Reset the mock state."""
        self._call_count = 0
        self._queries.clear()
