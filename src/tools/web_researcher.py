"""
Web Researcher Tool - Gathers additional research context from the web.

This tool performs targeted web searches to gather:
- Analyst opinions and ratings
- Recent company news and developments
- Competitive analysis
- Earnings and financial commentary
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.application.ports.search_port import SearchPort
from src.config.logging import get_logger
from src.domain.models import WebResearchInput, WebResearchOutput, WebSearchResult

logger = get_logger(__name__)


class WebResearcherTool:
    """
    Tool for conducting web research beyond news articles.

    Searches for analyst opinions, company updates, competitive context,
    and other relevant investment information.
    """

    name: str = "web_researcher"
    description: str = (
        "Searches the web for analyst opinions, recent company news, "
        "competitive analysis, earnings commentary, and market context."
    )

    def __init__(self, search_provider: SearchPort):
        """
        Initialize the web researcher tool.

        Args:
            search_provider: Search provider implementing SearchPort
        """
        self._search = search_provider

    async def execute(
        self,
        company: str,
        ticker: str,
        research_focus: str = "general",
    ) -> WebResearchOutput:
        """
        Execute web research for a company.

        Args:
            company: Company name
            ticker: Stock ticker symbol
            research_focus: Focus area for research
                - "general": Broad research coverage
                - "analyst_ratings": Focus on analyst opinions and price targets
                - "competitive": Focus on competitive positioning
                - "earnings": Focus on earnings and financial results
                - "risks": Focus on risk factors and concerns

        Returns:
            WebResearchOutput with search results
        """
        logger.info(
            "web_researcher_start",
            company=company,
            ticker=ticker,
            focus=research_focus,
        )

        # Build targeted queries based on research focus
        queries = self._build_queries(company, ticker, research_focus)

        all_results: List[WebSearchResult] = []
        seen_urls: set = set()

        for query in queries:
            try:
                results = await self._search.search(query, num_results=8)

                # Deduplicate by URL
                for result in results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        all_results.append(result)

            except Exception as e:
                logger.warning(
                    "web_researcher_query_error",
                    query=query,
                    error=str(e),
                )
                continue

        # Sort by relevance (title containing ticker or company name first)
        all_results = self._sort_by_relevance(all_results, company, ticker)

        # Limit total results
        final_results = all_results[:15]

        logger.info(
            "web_researcher_complete",
            company=company,
            ticker=ticker,
            total_results=len(final_results),
            queries_used=len(queries),
        )

        return WebResearchOutput(
            results=final_results,
            queries_used=queries,
            research_focus=research_focus,
            total_results=len(final_results),
        )

    def _build_queries(
        self,
        company: str,
        ticker: str,
        research_focus: str,
    ) -> List[str]:
        """Build search queries based on research focus."""
        current_year = str(datetime.now().year)

        # Base queries that always run
        base_queries = [
            f"{company} {ticker} stock analysis {current_year}",
        ]

        # Focus-specific queries
        focus_queries = {
            "general": [
                f"{ticker} analyst rating price target",
                f"{company} stock news today",
                f"{ticker} investment outlook",
            ],
            "news": [
                f"{company} {ticker} news {current_year}",
                f"{ticker} stock latest news today",
                f"{company} recent developments announcements",
                f"{ticker} breaking news market update",
            ],
            "analyst_ratings": [
                f"{ticker} analyst rating upgrade downgrade",
                f"{ticker} price target {current_year}",
                f"{company} analyst recommendations buy sell hold",
                f"{ticker} wall street consensus",
            ],
            "competitive": [
                f"{company} vs competitors comparison",
                f"{company} market share industry",
                f"{ticker} competitive advantage moat",
                f"{company} industry position",
            ],
            "earnings": [
                f"{ticker} earnings results {current_year}",
                f"{company} quarterly earnings beat miss",
                f"{ticker} revenue growth guidance",
                f"{company} financial results analysis",
            ],
            "risks": [
                f"{company} risks concerns",
                f"{ticker} bearish case analysis",
                f"{company} challenges headwinds",
                f"{ticker} short interest why",
            ],
        }

        # Get focus-specific queries or default to general
        specific_queries = focus_queries.get(research_focus, focus_queries["general"])

        return base_queries + specific_queries

    def _sort_by_relevance(
        self,
        results: List[WebSearchResult],
        company: str,
        ticker: str,
    ) -> List[WebSearchResult]:
        """Sort results by relevance to the company."""
        company_lower = company.lower()
        ticker_lower = ticker.lower()

        def relevance_score(result: WebSearchResult) -> int:
            score = 0
            title_lower = result.title.lower()
            snippet_lower = result.snippet.lower()

            # Ticker in title is highly relevant
            if ticker_lower in title_lower:
                score += 10

            # Company name in title
            if company_lower in title_lower:
                score += 8

            # Ticker in snippet
            if ticker_lower in snippet_lower:
                score += 5

            # Company in snippet
            if company_lower in snippet_lower:
                score += 3

            # Boost for financial sources
            trusted_sources = [
                "seekingalpha", "yahoo", "bloomberg", "reuters",
                "cnbc", "marketwatch", "fool", "barrons", "wsj",
                "investing.com", "tipranks", "zacks"
            ]
            source_lower = result.source.lower()
            if any(src in source_lower for src in trusted_sources):
                score += 5

            return score

        return sorted(results, key=relevance_score, reverse=True)

    async def execute_from_dict(self, params: Dict[str, Any]) -> WebResearchOutput:
        """Execute from dictionary parameters (for tool calling)."""
        return await self.execute(
            company=params["company"],
            ticker=params["ticker"],
            research_focus=params.get("research_focus", "general"),
        )

    @staticmethod
    def get_tool_definition() -> Dict[str, Any]:
        """Get the tool definition for LLM function calling."""
        return {
            "name": "web_researcher",
            "description": (
                "Searches the web for additional research context including "
                "analyst opinions, competitive analysis, and market commentary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "Company name to research",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "research_focus": {
                        "type": "string",
                        "enum": ["general", "news", "analyst_ratings", "competitive", "earnings", "risks"],
                        "description": "Focus area for research",
                        "default": "general",
                    },
                },
                "required": ["company", "ticker"],
            },
        }
