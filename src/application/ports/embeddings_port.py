"""
Embeddings Provider Port - Abstract interface for vector embeddings.

This port defines the contract for embedding providers.
Used for semantic search in the long-term memory feature.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingsPort(ABC):
    """
    Abstract interface for embedding providers.

    Implementations:
    - OpenAIEmbeddings: OpenAI text-embedding-ada-002
    - MockEmbeddings: Mock implementation for testing
    """

    @abstractmethod
    async def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings.

        Returns:
            Number of dimensions in embedding vectors
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the embeddings provider is healthy.

        Returns:
            True if provider is available
        """
        pass
