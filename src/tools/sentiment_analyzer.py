"""
Sentiment Analyzer Tool - Analyzes sentiment from news articles.

This tool uses the LLM provider to analyze sentiment distribution
across news articles, providing structured sentiment analysis.
"""

import asyncio
import time
from typing import Any, Dict, List

from src.application.ports.llm_port import LLMPort
from src.config.logging import get_logger
from src.domain.models import (
    ArticleSentiment,
    NewsArticle,
    SentimentResult,
)

logger = get_logger(__name__)


class SentimentAnalyzerTool:
    """
    Tool for analyzing sentiment from news articles.

    Uses the LLM provider to perform sophisticated sentiment
    analysis with reasoning.
    """

    name: str = "sentiment_analyzer"
    description: str = (
        "Analyzes sentiment from news articles about a company. "
        "Provides sentiment scores, distribution, and key insights. "
        "Requires a list of news articles to analyze."
    )

    # JSON Schema for function calling
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol for context",
            },
            "company_name": {
                "type": "string",
                "description": "The company name for context",
            },
        },
        "required": ["ticker", "company_name"],
    }

    def __init__(self, llm_provider: LLMPort):
        """
        Initialize the sentiment analyzer tool.

        Args:
            llm_provider: The LLM provider implementation to use
        """
        self._llm_provider = llm_provider

    async def execute(
        self,
        articles: List[NewsArticle],
        ticker: str,
        company_name: str,
    ) -> SentimentResult:
        """
        Execute sentiment analysis on articles.

        Args:
            articles: List of news articles to analyze
            ticker: Stock ticker for context
            company_name: Company name for context

        Returns:
            SentimentResult with detailed sentiment analysis
        """
        start_time = time.time()

        logger.info(
            "sentiment_analyzer_start",
            ticker=ticker,
            article_count=len(articles),
        )

        try:
            # Handle empty articles before validation
            if not articles:
                # Return neutral result for empty input
                return SentimentResult(
                    positive_count=0,
                    negative_count=0,
                    neutral_count=0,
                    positive_ratio=0.0,
                    negative_ratio=0.0,
                    neutral_ratio=0.0,
                    overall_sentiment="neutral",
                    sentiment_score=0.0,
                    article_sentiments=[],
                    analysis_reasoning="No articles provided for analysis.",
                )

            # Analyze all articles in parallel using asyncio.gather
            async def analyze_single_article(article: NewsArticle) -> ArticleSentiment:
                """Analyze a single article and return ArticleSentiment."""
                sentiment_data = await self._llm_provider.analyze_sentiment(
                    text=f"{article.title}\n\n{article.description or ''}\n\n{article.content or ''}",
                    context=f"{company_name} ({ticker})",
                )
                return ArticleSentiment(
                    title=article.title,
                    sentiment=sentiment_data.get("sentiment", "neutral"),
                    confidence=sentiment_data.get("confidence", 0.5),
                    key_phrases=sentiment_data.get("key_phrases", []),
                )

            # Run all sentiment analyses concurrently
            article_sentiments = await asyncio.gather(
                *[analyze_single_article(article) for article in articles]
            )

            # Calculate aggregate metrics
            positive_count = sum(1 for s in article_sentiments if s.sentiment == "positive")
            negative_count = sum(1 for s in article_sentiments if s.sentiment == "negative")
            neutral_count = sum(1 for s in article_sentiments if s.sentiment == "neutral")
            total = len(article_sentiments)

            positive_ratio = positive_count / total if total > 0 else 0.0
            negative_ratio = negative_count / total if total > 0 else 0.0
            neutral_ratio = neutral_count / total if total > 0 else 0.0

            # Calculate sentiment score
            # Positive articles contribute positively, negative contribute negatively
            # Weighted by confidence
            score_sum = 0.0
            weight_sum = 0.0
            for sentiment in article_sentiments:
                weight = sentiment.confidence
                if sentiment.sentiment == "positive":
                    score_sum += weight
                elif sentiment.sentiment == "negative":
                    score_sum -= weight
                weight_sum += weight

            sentiment_score = score_sum / weight_sum if weight_sum > 0 else 0.0
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            # Determine overall sentiment
            if abs(sentiment_score) < 0.1:
                overall_sentiment = "neutral"
            elif positive_count > negative_count and positive_ratio > 0.5:
                overall_sentiment = "positive"
            elif negative_count > positive_count and negative_ratio > 0.5:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "mixed"

            # Build reasoning
            reasoning = self._build_reasoning(
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                sentiment_score=sentiment_score,
                company_name=company_name,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "sentiment_analyzer_complete",
                ticker=ticker,
                sentiment_score=round(sentiment_score, 3),
                overall_sentiment=overall_sentiment,
                duration_ms=duration_ms,
            )

            return SentimentResult(
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                positive_ratio=round(positive_ratio, 3),
                negative_ratio=round(negative_ratio, 3),
                neutral_ratio=round(neutral_ratio, 3),
                overall_sentiment=overall_sentiment,
                sentiment_score=round(sentiment_score, 3),
                article_sentiments=article_sentiments,
                analysis_reasoning=reasoning,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "sentiment_analyzer_error",
                ticker=ticker,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise

    async def execute_from_dict(
        self,
        params: Dict[str, Any],
        articles: List[NewsArticle],
    ) -> SentimentResult:
        """
        Execute from dictionary parameters (for LangGraph integration).

        Args:
            params: Dictionary with ticker and company_name
            articles: List of articles to analyze (from previous tool)

        Returns:
            SentimentResult with sentiment analysis
        """
        return await self.execute(
            articles=articles,
            ticker=params["ticker"],
            company_name=params["company_name"],
        )

    def _build_reasoning(
        self,
        positive_count: int,
        negative_count: int,
        neutral_count: int,
        sentiment_score: float,
        company_name: str,
    ) -> str:
        """Build human-readable reasoning for the sentiment analysis."""
        total = positive_count + negative_count + neutral_count

        if total == 0:
            return "No articles were analyzed."

        parts = []

        # Describe distribution
        parts.append(
            f"Analyzed {total} articles about {company_name}: "
            f"{positive_count} positive, {negative_count} negative, {neutral_count} neutral."
        )

        # Describe score
        if sentiment_score > 0.5:
            parts.append("Overall sentiment is strongly positive.")
        elif sentiment_score > 0.2:
            parts.append("Overall sentiment is moderately positive.")
        elif sentiment_score > -0.2:
            parts.append("Overall sentiment is neutral to mixed.")
        elif sentiment_score > -0.5:
            parts.append("Overall sentiment is moderately negative.")
        else:
            parts.append("Overall sentiment is strongly negative.")

        return " ".join(parts)

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
