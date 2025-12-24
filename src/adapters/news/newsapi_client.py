"""
NewsAPI Client - Real news data from NewsAPI.org.

This module implements the News port using NewsAPI.org
for fetching real news articles about companies.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import httpx

from src.application.ports.news_port import NewsPort
from src.config.logging import get_logger
from src.domain.exceptions import NewsRetrievalError
from src.domain.models import NewsArticle

logger = get_logger(__name__)


class NewsAPIClient(NewsPort):
    """
    NewsAPI.org client for fetching real news articles.

    Free tier limits:
    - 100 requests per day
    - Articles up to 1 month old
    - Developer use only
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://newsapi.org/v2",
        timeout: int = 30,
    ):
        """
        Initialize the NewsAPI client.

        Args:
            api_key: NewsAPI.org API key
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def search_news(
        self,
        query: str,
        num_articles: int = 5,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
    ) -> List[NewsArticle]:
        """Search for news articles using NewsAPI."""
        try:
            # Build request parameters
            params = {
                "q": query,
                "apiKey": self._api_key,
                "language": language,
                "pageSize": min(num_articles, 100),  # NewsAPI max is 100
                "sortBy": "relevancy",  # Use relevancy for better quality results
            }

            # Add date filters
            if from_date:
                params["from"] = from_date.strftime("%Y-%m-%d")
            if to_date:
                params["to"] = to_date.strftime("%Y-%m-%d")

            logger.debug(
                "newsapi_search",
                query=query,
                num_articles=num_articles,
            )

            # Make API request
            response = await self._client.get(
                f"{self._base_url}/everything",
                params=params,
            )

            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', 'Unknown error')
                except (ValueError, KeyError):
                    error_message = f"HTTP {response.status_code}: {response.text[:200]}"
                raise NewsRetrievalError(
                    message=f"NewsAPI error: {error_message}",
                    query=query,
                )

            try:
                data = response.json()
            except ValueError as e:
                raise NewsRetrievalError(
                    message=f"Invalid JSON response from NewsAPI: {e}",
                    query=query,
                )

            # Parse articles
            articles = []
            for article_data in data.get("articles", [])[:num_articles]:
                try:
                    # Parse published date
                    published_str = article_data.get("publishedAt", "")
                    if published_str:
                        # Handle ISO format with Z suffix
                        published_str = published_str.replace("Z", "+00:00")
                        published_at = datetime.fromisoformat(published_str)
                    else:
                        published_at = datetime.now(timezone.utc)

                    article = NewsArticle(
                        title=article_data.get("title", "No title"),
                        description=article_data.get("description"),
                        url=article_data.get("url", ""),
                        source=article_data.get("source", {}).get("name", "Unknown"),
                        published_at=published_at,
                        content=article_data.get("content"),
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(
                        "newsapi_parse_article_error",
                        error=str(e),
                        article=article_data.get("title", "Unknown"),
                    )
                    continue

            logger.info(
                "newsapi_search_complete",
                query=query,
                articles_found=len(articles),
            )

            return articles

        except httpx.TimeoutException:
            raise NewsRetrievalError(
                message="NewsAPI request timed out",
                query=query,
            )
        except httpx.HTTPError as e:
            raise NewsRetrievalError(
                message=f"HTTP error: {e}",
                query=query,
            )

    async def get_company_news(
        self,
        company_name: str,
        ticker: str,
        num_articles: int = 5,
    ) -> List[NewsArticle]:
        """Get news specifically about a company with relevancy filtering."""
        # Clean company name for search
        clean_name = (
            company_name
            .replace(", Inc.", "")
            .replace(" Inc.", "")
            .replace(" Corp.", "")
            .replace(" Corporation", "")
            .replace(" Ltd.", "")
            .strip()
        )

        # Default to last 30 days
        from_date = datetime.now(timezone.utc) - timedelta(days=30)

        # Fetch more articles than needed, then filter for relevancy
        fetch_multiplier = 4  # Fetch 4x to ensure we get enough relevant ones
        raw_articles = await self._search_news_raw(
            query=clean_name,
            num_articles=num_articles * fetch_multiplier,
            from_date=from_date,
        )

        # Filter: keep only articles that mention company/ticker in title or description
        search_terms = [clean_name.lower(), ticker.lower()]
        relevant_articles = []

        for article in raw_articles:
            title_lower = article.title.lower()
            desc_lower = (article.description or "").lower()

            # Check if any search term appears in title or description
            is_relevant = any(
                term in title_lower or term in desc_lower
                for term in search_terms
            )

            if is_relevant:
                relevant_articles.append(article)
                if len(relevant_articles) >= num_articles:
                    break

        logger.info(
            "newsapi_filtered",
            company=clean_name,
            fetched=len(raw_articles),
            relevant=len(relevant_articles),
        )

        return relevant_articles

    async def _search_news_raw(
        self,
        query: str,
        num_articles: int = 20,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
    ) -> List[NewsArticle]:
        """Internal method to fetch articles without filtering."""
        try:
            params = {
                "q": query,
                "apiKey": self._api_key,
                "language": language,
                "pageSize": min(num_articles, 100),
                "sortBy": "publishedAt",  # Get recent articles
            }

            if from_date:
                params["from"] = from_date.strftime("%Y-%m-%d")
            if to_date:
                params["to"] = to_date.strftime("%Y-%m-%d")

            response = await self._client.get(
                f"{self._base_url}/everything",
                params=params,
            )

            if response.status_code != 200:
                return []

            data = response.json()
            articles = []

            for article_data in data.get("articles", [])[:num_articles]:
                try:
                    published_str = article_data.get("publishedAt", "")
                    if published_str:
                        published_str = published_str.replace("Z", "+00:00")
                        published_at = datetime.fromisoformat(published_str)
                    else:
                        published_at = datetime.now(timezone.utc)

                    article = NewsArticle(
                        title=article_data.get("title", "No title"),
                        description=article_data.get("description"),
                        url=article_data.get("url", ""),
                        source=article_data.get("source", {}).get("name", "Unknown"),
                        published_at=published_at,
                        content=article_data.get("content"),
                    )
                    articles.append(article)
                except Exception:
                    continue

            return articles

        except Exception:
            return []

    async def health_check(self) -> bool:
        """Check if NewsAPI is available."""
        try:
            # Make a minimal request
            response = await self._client.get(
                f"{self._base_url}/everything",
                params={
                    "q": "test",
                    "apiKey": self._api_key,
                    "pageSize": 1,
                },
            )
            return response.status_code == 200
        except Exception as e:
            logger.error("newsapi_health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
