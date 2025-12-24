"""
Embedding Service - Coordinates embedding generation and storage.

This service handles:
- Embedding analysis summaries and key findings after completion
- Semantic search across past analyses
- Retrieving historical context for enhanced synthesis
"""

from typing import Any, Dict, List, Optional

from src.application.ports.embeddings_port import EmbeddingsPort
from src.application.ports.storage_port import StoragePort
from src.config.logging import get_logger
from src.domain.models import AnalysisReport

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating and managing analysis embeddings."""

    def __init__(
        self,
        embeddings_provider: EmbeddingsPort,
        storage: StoragePort,
    ):
        """
        Initialize the embedding service.

        Args:
            embeddings_provider: Provider for generating embeddings
            storage: Storage backend for persisting embeddings
        """
        self._embeddings = embeddings_provider
        self._storage = storage

    async def embed_analysis(
        self,
        job_id: str,
        report: AnalysisReport,
    ) -> List[str]:
        """
        Generate and store embeddings for a completed analysis.

        Embeds:
        - analysis_summary (content_type: "summary")
        - Each key_finding (content_type: "key_finding")

        Args:
            job_id: The analysis job ID
            report: The completed analysis report

        Returns:
            List of embedding record IDs
        """
        embedding_ids: List[str] = []
        ticker = report.company_ticker

        logger.info(
            "embedding_analysis",
            job_id=job_id,
            ticker=ticker,
            num_findings=len(report.key_findings),
        )

        try:
            # 1. Embed the analysis summary
            summary_embedding = await self._embeddings.embed_text(report.analysis_summary)
            summary_id = await self._storage.store_embedding(
                analysis_id=job_id,
                content_type="summary",
                content_text=report.analysis_summary,
                embedding=summary_embedding,
                metadata={
                    "ticker": ticker,
                    "company_name": report.company_name,
                    "sentiment_score": report.sentiment_score,
                    "generated_at": report.generated_at.isoformat(),
                    "analysis_type": report.analysis_type,
                },
            )
            embedding_ids.append(summary_id)

            # 2. Embed each key finding
            for i, finding in enumerate(report.key_findings):
                finding_embedding = await self._embeddings.embed_text(finding)
                finding_id = await self._storage.store_embedding(
                    analysis_id=job_id,
                    content_type="key_finding",
                    content_text=finding,
                    embedding=finding_embedding,
                    metadata={
                        "ticker": ticker,
                        "company_name": report.company_name,
                        "finding_index": i,
                        "sentiment_score": report.sentiment_score,
                        "generated_at": report.generated_at.isoformat(),
                    },
                )
                embedding_ids.append(finding_id)

            logger.info(
                "embeddings_stored",
                job_id=job_id,
                ticker=ticker,
                num_embeddings=len(embedding_ids),
            )

        except Exception as e:
            logger.error(
                "embedding_failed",
                job_id=job_id,
                ticker=ticker,
                error=str(e),
            )
            raise

        return embedding_ids

    async def search_similar_analyses(
        self,
        query: str,
        ticker: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar past analyses using semantic similarity.

        Args:
            query: Search query text
            ticker: Optional ticker to filter results
            limit: Maximum number of results to return

        Returns:
            List of similar analysis records with scores
        """
        logger.info(
            "searching_similar_analyses",
            query=query[:100],
            ticker=ticker,
            limit=limit,
        )

        try:
            # Generate embedding for search query
            query_embedding = await self._embeddings.embed_text(query)

            # Search for similar embeddings
            results = await self._storage.search_similar(
                embedding=query_embedding,
                limit=limit,
                ticker=ticker,
            )

            logger.info(
                "search_complete",
                query=query[:50],
                num_results=len(results),
            )

            return results

        except Exception as e:
            logger.error(
                "search_failed",
                query=query[:50],
                error=str(e),
            )
            raise

    async def get_historical_context(
        self,
        ticker: str,
        current_summary: str,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """
        Retrieve historical context for a ticker to enrich synthesis.

        This method retrieves:
        - Semantically similar past analyses for context
        - Sentiment history showing how sentiment has changed over time

        Args:
            ticker: Stock ticker symbol
            current_summary: Current analysis summary for similarity search
            limit: Maximum similar analyses to retrieve

        Returns:
            Dictionary containing:
            - similar_analyses: List of semantically similar past analyses
            - sentiment_history: List of past sentiment scores with dates
        """
        logger.info(
            "retrieving_historical_context",
            ticker=ticker,
            limit=limit,
        )

        try:
            # Get semantically similar past analyses for this ticker
            similar_analyses: List[Dict[str, Any]] = []
            try:
                query_embedding = await self._embeddings.embed_text(current_summary)
                similar_analyses = await self._storage.search_similar(
                    embedding=query_embedding,
                    limit=limit,
                    ticker=ticker,
                )
            except Exception as e:
                logger.warning(
                    "similar_analysis_search_failed",
                    ticker=ticker,
                    error=str(e),
                )

            # Get sentiment history
            sentiment_history: List[Dict[str, Any]] = []
            try:
                sentiment_history = await self._storage.get_sentiment_history(
                    ticker=ticker,
                    limit=10,
                )
            except Exception as e:
                logger.warning(
                    "sentiment_history_fetch_failed",
                    ticker=ticker,
                    error=str(e),
                )

            context = {
                "similar_analyses": similar_analyses,
                "sentiment_history": sentiment_history,
            }

            logger.info(
                "historical_context_retrieved",
                ticker=ticker,
                num_similar=len(similar_analyses),
                num_sentiment_points=len(sentiment_history),
            )

            return context

        except Exception as e:
            logger.error(
                "historical_context_failed",
                ticker=ticker,
                error=str(e),
            )
            # Return empty context on failure rather than raising
            return {
                "similar_analyses": [],
                "sentiment_history": [],
            }

    def format_historical_context_for_prompt(
        self,
        context: Dict[str, Any],
    ) -> str:
        """
        Format historical context for inclusion in synthesis prompt.

        Args:
            context: Historical context from get_historical_context()

        Returns:
            Formatted string for prompt injection
        """
        parts: List[str] = []

        # Format similar past analyses
        similar = context.get("similar_analyses", [])
        if similar:
            parts.append("### Similar Past Analyses")
            for i, analysis in enumerate(similar[:3], 1):
                content = analysis.get("content_text", "")[:200]
                score = analysis.get("score", 0)
                metadata = analysis.get("metadata", {})
                date = metadata.get("generated_at", "")[:10] if metadata.get("generated_at") else "Unknown"
                parts.append(
                    f"{i}. [{date}] (relevance: {score:.0%}): \"{content}...\""
                )
        else:
            parts.append("No similar past analyses found for this company.")

        # Format sentiment trend
        sentiment_history = context.get("sentiment_history", [])
        if sentiment_history:
            parts.append("\n### Sentiment Trend")

            # Calculate trend direction
            if len(sentiment_history) >= 2:
                recent = sentiment_history[0].get("sentiment_score", 0)
                oldest = sentiment_history[-1].get("sentiment_score", 0)
                diff = recent - oldest

                if diff > 0.1:
                    trend = "IMPROVING"
                elif diff < -0.1:
                    trend = "DECLINING"
                else:
                    trend = "STABLE"

                parts.append(f"Overall trend: {trend} ({diff:+.2f} change)")

            # Show recent history
            for point in sentiment_history[:5]:
                date = point.get("date", "")[:10] if point.get("date") else "Unknown"
                score = point.get("sentiment_score", 0)
                parts.append(f"- {date}: sentiment {score:+.2f}")
        else:
            parts.append("\nNo sentiment history available (first analysis for this ticker).")

        return "\n".join(parts)
