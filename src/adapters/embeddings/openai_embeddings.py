"""
OpenAI Embeddings Provider - Text embeddings via OpenAI API.

This module implements the Embeddings port using OpenAI's
text-embedding-ada-002 model for vector embeddings.
"""

from typing import List

import openai

from src.application.ports.embeddings_port import EmbeddingsPort
from src.config.logging import get_logger
from src.domain.exceptions import LLMProviderError

logger = get_logger(__name__)


class OpenAIEmbeddings(EmbeddingsPort):
    """
    OpenAI embeddings provider using text-embedding-ada-002.

    Generates 1536-dimensional embeddings for semantic search.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        dimensions: int = 1536,
    ):
        """
        Initialize the OpenAI embeddings provider.

        Args:
            api_key: OpenAI API key
            model: Model identifier
            dimensions: Embedding dimensions
        """
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    async def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """Generate embedding for a single text."""
        try:
            logger.debug(
                "openai_embed_single",
                text_length=len(text),
                model=self._model,
            )

            # Truncate text if too long (max 8191 tokens for ada-002)
            # Rough estimate: 1 token ~= 4 characters
            max_chars = 8191 * 4
            if len(text) > max_chars:
                text = text[:max_chars]

            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
            )

            embedding = response.data[0].embedding

            logger.debug(
                "openai_embed_complete",
                dimensions=len(embedding),
            )

            return embedding

        except openai.RateLimitError as e:
            logger.error("openai_rate_limit", error=str(e))
            raise LLMProviderError(
                message="OpenAI rate limit exceeded",
                provider="openai",
            )
        except openai.APIError as e:
            logger.error("openai_api_error", error=str(e))
            raise LLMProviderError(
                message=f"OpenAI API error: {e}",
                provider="openai",
            )

    async def embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            logger.debug(
                "openai_embed_batch",
                batch_size=len(texts),
                model=self._model,
            )

            # Truncate texts if needed
            max_chars = 8191 * 4
            processed_texts = [
                text[:max_chars] if len(text) > max_chars else text
                for text in texts
            ]

            response = await self._client.embeddings.create(
                model=self._model,
                input=processed_texts,
            )

            embeddings = [item.embedding for item in response.data]

            logger.debug(
                "openai_embed_batch_complete",
                embeddings_count=len(embeddings),
            )

            return embeddings

        except openai.RateLimitError as e:
            logger.error("openai_rate_limit", error=str(e))
            raise LLMProviderError(
                message="OpenAI rate limit exceeded",
                provider="openai",
            )
        except openai.APIError as e:
            logger.error("openai_api_error", error=str(e))
            raise LLMProviderError(
                message=f"OpenAI API error: {e}",
                provider="openai",
            )

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions

    async def health_check(self) -> bool:
        """Check if OpenAI is available."""
        try:
            # Make a minimal embedding request
            response = await self._client.embeddings.create(
                model=self._model,
                input="test",
            )
            return len(response.data) > 0
        except Exception as e:
            logger.error("openai_health_check_failed", error=str(e))
            return False
