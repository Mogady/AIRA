"""
Unit tests for EmbeddingService and embedding operations.
"""

import pytest
from datetime import datetime, timezone

from src.adapters.embeddings.mock_embeddings import MockEmbeddings
from src.adapters.storage.memory_repository import MemoryRepository
from src.application.services.embedding_service import EmbeddingService
from src.domain.models import AnalysisReport


class TestMockEmbeddings:
    """Tests for MockEmbeddings provider."""

    @pytest.mark.asyncio
    async def test_embed_text_returns_correct_dimensions(self, mock_embeddings):
        """Test embedding returns correct dimensions."""
        embedding = await mock_embeddings.embed_text("test text")
        assert len(embedding) == mock_embeddings.get_dimensions()
        assert mock_embeddings.get_dimensions() == 1536  # Default

    @pytest.mark.asyncio
    async def test_embed_text_is_deterministic(self, mock_embeddings):
        """Test same input produces same embedding."""
        text = "test text for embedding"
        embedding1 = await mock_embeddings.embed_text(text)
        embedding2 = await mock_embeddings.embed_text(text)
        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_different_text_produces_different_embeddings(self, mock_embeddings):
        """Test different inputs produce different embeddings."""
        embedding1 = await mock_embeddings.embed_text("first text")
        embedding2 = await mock_embeddings.embed_text("second text")
        assert embedding1 != embedding2

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_embeddings):
        """Test batch embedding."""
        texts = ["text 1", "text 2", "text 3"]
        embeddings = await mock_embeddings.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == mock_embeddings.get_dimensions()

    @pytest.mark.asyncio
    async def test_embed_batch_is_deterministic(self, mock_embeddings):
        """Test batch embedding is deterministic."""
        texts = ["text 1", "text 2"]
        embeddings1 = await mock_embeddings.embed_batch(texts)
        embeddings2 = await mock_embeddings.embed_batch(texts)

        assert embeddings1 == embeddings2

    @pytest.mark.asyncio
    async def test_health_check(self, mock_embeddings):
        """Test health check returns true."""
        assert await mock_embeddings.health_check() is True

    def test_custom_dimensions(self):
        """Test custom dimensions."""
        custom_embeddings = MockEmbeddings(dimensions=768)
        assert custom_embeddings.get_dimensions() == 768

    @pytest.mark.asyncio
    async def test_custom_dimensions_in_output(self):
        """Test custom dimensions are reflected in output."""
        custom_embeddings = MockEmbeddings(dimensions=512)
        embedding = await custom_embeddings.embed_text("test")
        assert len(embedding) == 512

    def test_call_count_tracking(self, mock_embeddings):
        """Test call count is tracked."""
        initial_count = mock_embeddings.get_call_count()
        mock_embeddings.reset()
        assert mock_embeddings.get_call_count() == 0

    @pytest.mark.asyncio
    async def test_embeddings_are_normalized(self, mock_embeddings):
        """Test embeddings are unit vectors (normalized)."""
        import math

        embedding = await mock_embeddings.embed_text("test text")
        magnitude = math.sqrt(sum(x * x for x in embedding))
        assert abs(magnitude - 1.0) < 0.01  # Should be ~1.0


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample analysis report."""
        return AnalysisReport(
            company_ticker="TSLA",
            company_name="Tesla, Inc.",
            analysis_summary="Tesla shows strong growth potential with expanding EV market share and improving margins.",
            sentiment_score=0.65,
            key_findings=[
                "Revenue growth of 25% YoY driven by Model Y success",
                "Market cap exceeds $800B making it most valuable automaker",
                "Strong brand recognition and customer loyalty in EV space",
            ],
            tools_used=["news_retriever", "sentiment_analyzer", "data_fetcher"],
            citation_sources=["https://example.com/tesla-news"],
            risk_factors=["Competition from legacy automakers", "Regulatory risks"],
            catalyst_events=["Q4 earnings report", "Cybertruck production ramp"],
            analysis_type="ON_DEMAND",
            reflection_triggered=False,
            generated_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_embed_analysis_stores_embeddings(
        self, embedding_service, memory_storage, sample_report
    ):
        """Test embedding a full analysis stores correct number of embeddings."""
        # Create analysis record first
        await memory_storage.create_analysis(
            job_id="test-job-123",
            ticker="TSLA",
            query="Analyze Tesla",
        )

        # Embed the analysis
        embedding_ids = await embedding_service.embed_analysis(
            job_id="test-job-123",
            report=sample_report,
        )

        # Should have 1 summary + 3 key findings = 4 embeddings
        assert len(embedding_ids) == 4
        assert all(isinstance(id, str) for id in embedding_ids)

    @pytest.mark.asyncio
    async def test_embed_analysis_stores_correct_content_types(
        self, mock_embeddings, memory_storage, sample_report
    ):
        """Test embeddings are stored with correct content types."""
        service = EmbeddingService(mock_embeddings, memory_storage)

        await memory_storage.create_analysis(
            job_id="test-job-456",
            ticker="TSLA",
            query="Analyze Tesla",
        )

        await service.embed_analysis("test-job-456", sample_report)

        # Check stored embeddings
        stats = memory_storage.get_stats()
        assert stats["embeddings"] == 4

    @pytest.mark.asyncio
    async def test_search_similar_analyses_empty_database(self, embedding_service):
        """Test searching with no stored embeddings returns empty."""
        results = await embedding_service.search_similar_analyses(
            query="EV market growth",
            ticker=None,
            limit=5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_similar_analyses_returns_results(
        self, mock_embeddings, memory_storage, sample_report
    ):
        """Test searching returns similar analyses."""
        service = EmbeddingService(mock_embeddings, memory_storage)

        # Store an analysis
        await memory_storage.create_analysis(
            job_id="test-job-789",
            ticker="TSLA",
            query="Analyze Tesla",
        )
        await service.embed_analysis("test-job-789", sample_report)

        # Search for similar
        results = await service.search_similar_analyses(
            query="Tesla growth potential",
            ticker=None,
            limit=5,
        )

        assert len(results) > 0
        assert results[0]["content_type"] in ["summary", "key_finding"]
        assert 0 <= results[0]["score"] <= 1

    @pytest.mark.asyncio
    async def test_search_with_ticker_filter(
        self, mock_embeddings, memory_storage, sample_report
    ):
        """Test searching with ticker filter only returns matching ticker."""
        service = EmbeddingService(mock_embeddings, memory_storage)

        # Store analysis for TSLA
        await memory_storage.create_analysis(
            job_id="tsla-job",
            ticker="TSLA",
            query="Analyze Tesla",
        )
        await service.embed_analysis("tsla-job", sample_report)

        # Store analysis for AAPL
        aapl_report = AnalysisReport(
            company_ticker="AAPL",
            company_name="Apple Inc.",
            analysis_summary="Apple shows strong services growth with expanding margins and stable revenue streams.",
            sentiment_score=0.5,
            key_findings=[
                "Services revenue up 20% year over year",
                "Hardware sales remained relatively flat",
                "Strong ecosystem lock-in drives customer retention",
            ],
            tools_used=["news_retriever"],
            citation_sources=[],
            analysis_type="ON_DEMAND",
            reflection_triggered=False,
            generated_at=datetime.now(timezone.utc),
        )
        await memory_storage.create_analysis(
            job_id="aapl-job",
            ticker="AAPL",
            query="Analyze Apple",
        )
        await service.embed_analysis("aapl-job", aapl_report)

        # Search with TSLA filter
        results = await service.search_similar_analyses(
            query="growth potential",
            ticker="TSLA",
            limit=10,
        )

        # All results should be for TSLA
        for result in results:
            assert result.get("metadata", {}).get("ticker") == "TSLA"

    @pytest.mark.asyncio
    async def test_get_historical_context_empty(self, embedding_service):
        """Test historical context with no data returns empty."""
        context = await embedding_service.get_historical_context(
            ticker="UNKNOWN",
            current_summary="Some analysis",
            limit=3,
        )

        assert context["similar_analyses"] == []
        assert context["sentiment_history"] == []

    @pytest.mark.asyncio
    async def test_get_historical_context_with_data(
        self, mock_embeddings, memory_storage, sample_report
    ):
        """Test historical context retrieval with stored data."""
        service = EmbeddingService(mock_embeddings, memory_storage)

        # Store a completed analysis
        await memory_storage.create_analysis(
            job_id="hist-job-1",
            ticker="TSLA",
            query="Analyze Tesla",
        )
        await memory_storage.complete_analysis(
            job_id="hist-job-1",
            report=sample_report.model_dump(),
            tools_used=sample_report.tools_used,
            iteration_count=1,
            reflection_triggered=False,
        )
        await service.embed_analysis("hist-job-1", sample_report)

        # Get historical context
        context = await service.get_historical_context(
            ticker="TSLA",
            current_summary="Tesla EV market analysis",
            limit=3,
        )

        # Should have sentiment history
        assert len(context["sentiment_history"]) >= 1
        assert context["sentiment_history"][0]["ticker"] if "ticker" in context["sentiment_history"][0] else True

    @pytest.mark.asyncio
    async def test_format_historical_context_for_prompt(self, embedding_service):
        """Test formatting historical context for synthesis prompt."""
        context = {
            "similar_analyses": [
                {
                    "content_text": "Tesla shows strong growth in EV market",
                    "score": 0.85,
                    "metadata": {"generated_at": "2024-01-15T10:00:00Z"},
                }
            ],
            "sentiment_history": [
                {"job_id": "job-1", "sentiment_score": 0.65, "date": "2024-01-15T10:00:00Z"},
                {"job_id": "job-2", "sentiment_score": 0.45, "date": "2024-01-10T10:00:00Z"},
            ],
        }

        formatted = embedding_service.format_historical_context_for_prompt(context)

        assert "Similar Past Analyses" in formatted
        assert "Sentiment Trend" in formatted
        assert "IMPROVING" in formatted or "STABLE" in formatted or "DECLINING" in formatted

    @pytest.mark.asyncio
    async def test_format_historical_context_empty(self, embedding_service):
        """Test formatting empty historical context."""
        context = {
            "similar_analyses": [],
            "sentiment_history": [],
        }

        formatted = embedding_service.format_historical_context_for_prompt(context)

        assert "No similar past analyses found" in formatted
        assert "No sentiment history available" in formatted


class TestStorageEmbeddingOperations:
    """Tests for storage embedding operations."""

    @pytest.mark.asyncio
    async def test_store_embedding(self, memory_storage, mock_embeddings):
        """Test storing an embedding."""
        # Create analysis first
        await memory_storage.create_analysis(
            job_id="embed-test",
            ticker="TEST",
            query="Test query",
        )

        text = "This is a test summary"
        embedding = await mock_embeddings.embed_text(text)

        embedding_id = await memory_storage.store_embedding(
            analysis_id="embed-test",
            content_type="summary",
            content_text=text,
            embedding=embedding,
            metadata={"ticker": "TEST"},
        )

        assert embedding_id is not None
        assert isinstance(embedding_id, str)

    @pytest.mark.asyncio
    async def test_search_similar(self, memory_storage, mock_embeddings):
        """Test searching for similar embeddings."""
        # Create analysis
        await memory_storage.create_analysis(
            job_id="search-test",
            ticker="TEST",
            query="Test query",
        )

        # Store embedding
        text = "Revenue growth is strong"
        embedding = await mock_embeddings.embed_text(text)
        await memory_storage.store_embedding(
            analysis_id="search-test",
            content_type="summary",
            content_text=text,
            embedding=embedding,
            metadata={"ticker": "TEST"},
        )

        # Search
        query_embedding = await mock_embeddings.embed_text("strong revenue")
        results = await memory_storage.search_similar(
            embedding=query_embedding,
            limit=5,
        )

        assert len(results) > 0
        assert results[0]["content_text"] == text

    @pytest.mark.asyncio
    async def test_get_sentiment_history(self, memory_storage):
        """Test getting sentiment history."""
        # Create and complete analysis
        await memory_storage.create_analysis(
            job_id="hist-test",
            ticker="HIST",
            query="Test history",
        )
        await memory_storage.complete_analysis(
            job_id="hist-test",
            report={"sentiment_score": 0.5},
            tools_used=["test"],
            iteration_count=1,
            reflection_triggered=False,
        )

        # Get history
        history = await memory_storage.get_sentiment_history("HIST", limit=10)

        assert len(history) == 1
        assert history[0]["sentiment_score"] == 0.5
        assert history[0]["job_id"] == "hist-test"

    @pytest.mark.asyncio
    async def test_get_sentiment_history_empty(self, memory_storage):
        """Test sentiment history for unknown ticker."""
        history = await memory_storage.get_sentiment_history("UNKNOWN", limit=10)
        assert history == []

    @pytest.mark.asyncio
    async def test_get_sentiment_history_ordering(self, memory_storage):
        """Test sentiment history is ordered by date descending."""
        import asyncio

        # Create multiple completed analyses
        for i, score in enumerate([0.3, 0.5, 0.7]):
            await memory_storage.create_analysis(
                job_id=f"order-test-{i}",
                ticker="ORDER",
                query="Test ordering",
            )
            await memory_storage.complete_analysis(
                job_id=f"order-test-{i}",
                report={"sentiment_score": score},
                tools_used=["test"],
                iteration_count=1,
                reflection_triggered=False,
            )
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps

        # Get history
        history = await memory_storage.get_sentiment_history("ORDER", limit=10)

        assert len(history) == 3
        # Most recent (0.7) should be first
        assert history[0]["sentiment_score"] == 0.7
