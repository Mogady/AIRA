"""
Application Services.

This module contains application-level services that coordinate
between domain models and infrastructure adapters.
"""

from src.application.services.embedding_service import EmbeddingService

__all__ = ["EmbeddingService"]
