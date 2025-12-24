"""
Mock Embeddings Provider - For testing without OpenAI API calls.

Generates deterministic embeddings based on text content.
Useful for testing vector search functionality.
"""

import hashlib
import math
from typing import List

from src.application.ports.embeddings_port import EmbeddingsPort
from src.config.logging import get_logger

logger = get_logger(__name__)


class MockEmbeddings(EmbeddingsPort):
    """
    Mock embeddings provider that generates deterministic vectors.

    Generates embeddings based on text hash, ensuring:
    - Same text always produces same embedding
    - Similar texts produce somewhat similar embeddings
    - Dimensions match OpenAI ada-002 (1536)
    """

    def __init__(self, dimensions: int = 1536):
        self._dimensions = dimensions
        self._call_count = 0

    async def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """Generate a deterministic embedding for text."""
        self._call_count += 1

        logger.debug(
            "mock_embeddings_single",
            text_length=len(text),
            call_count=self._call_count,
        )

        return self._generate_embedding(text)

    async def embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._call_count += 1

        logger.debug(
            "mock_embeddings_batch",
            batch_size=len(texts),
            call_count=self._call_count,
        )

        return [self._generate_embedding(text) for text in texts]

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions

    async def health_check(self) -> bool:
        """Mock is always healthy."""
        return True

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate a deterministic embedding from text.

        Uses SHA-256 hash to create a seed, then generates
        normalized vectors with that seed.
        """
        # Create deterministic seed from text
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Generate embedding values from hash
        embedding = []
        for i in range(self._dimensions):
            # Use different parts of the hash for each dimension
            hash_segment = text_hash[(i * 2) % 60 : (i * 2 + 2) % 60 + 2]
            if not hash_segment:
                hash_segment = text_hash[:2]

            # Convert hex to float in range [-1, 1]
            value = (int(hash_segment, 16) / 255.0) * 2 - 1

            # Add some variation based on position
            variation = math.sin(i * 0.1) * 0.1
            value = max(-1.0, min(1.0, value + variation))

            embedding.append(value)

        # Normalize the embedding to unit length
        magnitude = math.sqrt(sum(v * v for v in embedding))
        if magnitude > 0:
            embedding = [v / magnitude for v in embedding]

        return embedding

    def get_call_count(self) -> int:
        """Get the number of calls made."""
        return self._call_count

    def reset(self) -> None:
        """Reset the mock state."""
        self._call_count = 0
