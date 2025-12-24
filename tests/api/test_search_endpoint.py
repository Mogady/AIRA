"""
Integration tests for /search endpoint.
"""

import asyncio

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from main import app
from src.adapters.api.dependencies import (
    reset_dependencies,
    get_storage,
    get_embeddings_provider,
)
from src.adapters.embeddings.mock_embeddings import MockEmbeddings
from src.adapters.storage.memory_repository import MemoryRepository


@pytest.fixture
def client():
    """Create test client with fresh dependencies using in-memory storage."""
    # Reset singletons before each test
    reset_dependencies()

    # Override to use in-memory storage for tests
    memory_storage = MemoryRepository()
    mock_embeddings = MockEmbeddings()
    app.dependency_overrides[get_storage] = lambda: memory_storage
    app.dependency_overrides[get_embeddings_provider] = lambda: mock_embeddings

    yield TestClient(app)

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def seeded_client():
    """Create test client with seeded analysis data using in-memory storage."""
    from src.domain.models import AnalysisReport
    from src.application.services.embedding_service import EmbeddingService

    # Create fresh in-memory instances for this test
    memory_storage = MemoryRepository()
    mock_embeddings = MockEmbeddings()

    # Override dependencies to use in-memory storage
    app.dependency_overrides[get_storage] = lambda: memory_storage
    app.dependency_overrides[get_embeddings_provider] = lambda: mock_embeddings

    async def seed_data():
        service = EmbeddingService(mock_embeddings, memory_storage)

        # Create and complete a TSLA analysis
        await memory_storage.create_analysis(
            job_id="search-test-tsla",
            ticker="TSLA",
            query="Analyze Tesla",
            company_name="Tesla, Inc.",
        )

        tsla_report = AnalysisReport(
            company_ticker="TSLA",
            company_name="Tesla, Inc.",
            analysis_summary="Tesla shows strong growth in EV market with expanding margins.",
            sentiment_score=0.65,
            key_findings=[
                "Revenue growth of 25% year over year",
                "Market leader in electric vehicle sales",
                "Strong brand recognition globally",
            ],
            tools_used=["news_retriever", "data_fetcher"],
            citation_sources=["https://example.com/tesla"],
            analysis_type="ON_DEMAND",
            reflection_triggered=False,
            generated_at=datetime.now(timezone.utc),
        )

        await memory_storage.complete_analysis(
            job_id="search-test-tsla",
            report=tsla_report,
            tools_used=tsla_report.tools_used,
            iteration_count=1,
            reflection_triggered=False,
        )

        # Embed the analysis
        await service.embed_analysis("search-test-tsla", tsla_report)

        # Create and complete an AAPL analysis
        await memory_storage.create_analysis(
            job_id="search-test-aapl",
            ticker="AAPL",
            query="Analyze Apple",
            company_name="Apple Inc.",
        )

        aapl_report = AnalysisReport(
            company_ticker="AAPL",
            company_name="Apple Inc.",
            analysis_summary="Apple demonstrates stable services growth with iPhone sales plateauing.",
            sentiment_score=0.45,
            key_findings=[
                "Services revenue increased by 15%",
                "Hardware sales remained flat",
                "Strong ecosystem lock-in effect",
            ],
            tools_used=["news_retriever"],
            citation_sources=["https://example.com/apple"],
            analysis_type="ON_DEMAND",
            reflection_triggered=False,
            generated_at=datetime.now(timezone.utc),
        )

        await memory_storage.complete_analysis(
            job_id="search-test-aapl",
            report=aapl_report,
            tools_used=aapl_report.tools_used,
            iteration_count=1,
            reflection_triggered=False,
        )

        await service.embed_analysis("search-test-aapl", aapl_report)

    asyncio.run(seed_data())

    client = TestClient(app)
    yield client

    # Clean up overrides after test
    app.dependency_overrides.clear()


class TestSearchEndpoint:
    """Tests for /search endpoint."""

    def test_search_requires_query(self, client):
        """Test that query is required."""
        response = client.post("/search", json={})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_search_rejects_short_query(self, client):
        """Test that short queries are rejected (min_length=3)."""
        response = client.post("/search", json={"query": "ab"})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_search_accepts_minimum_query(self, client):
        """Test that minimum length query is accepted."""
        response = client.post("/search", json={"query": "abc"})

        assert response.status_code == 200

    def test_search_empty_results(self, client):
        """Test search with no matching data returns empty results."""
        response = client.post(
            "/search",
            json={"query": "some random search query"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "some random search query"
        assert data["results"] == []
        assert data["total_results"] == 0

    def test_search_returns_results(self, seeded_client):
        """Test search returns matching results from seeded data."""
        response = seeded_client.post(
            "/search",
            json={"query": "electric vehicle market growth"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] > 0
        assert len(data["results"]) > 0

        # Check result structure
        result = data["results"][0]
        assert "analysis_id" in result
        assert "content_type" in result
        assert result["content_type"] in ["summary", "key_finding"]
        assert "content_text" in result
        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert "metadata" in result

    def test_search_with_ticker_filter(self, seeded_client):
        """Test searching with ticker filter only returns matching ticker."""
        response = seeded_client.post(
            "/search",
            json={
                "query": "revenue growth",
                "ticker": "TSLA",
                "limit": 10,
            }
        )

        assert response.status_code == 200
        data = response.json()

        # All results should be for TSLA ticker
        for result in data["results"]:
            metadata = result.get("metadata", {})
            assert metadata.get("ticker") == "TSLA"

    def test_search_ticker_case_insensitive(self, seeded_client):
        """Test that ticker filter is case insensitive."""
        response = seeded_client.post(
            "/search",
            json={
                "query": "revenue growth",
                "ticker": "tsla",  # lowercase
                "limit": 10,
            }
        )

        assert response.status_code == 200
        data = response.json()

        for result in data["results"]:
            metadata = result.get("metadata", {})
            assert metadata.get("ticker") == "TSLA"

    def test_search_respects_limit(self, seeded_client):
        """Test that limit parameter is respected."""
        response = seeded_client.post(
            "/search",
            json={
                "query": "growth",
                "limit": 2,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 2

    def test_search_limit_maximum(self, client):
        """Test that limit has a maximum value (20)."""
        response = client.post(
            "/search",
            json={
                "query": "test query",
                "limit": 100,  # exceeds max of 20
            }
        )

        assert response.status_code == 422

    def test_search_limit_minimum(self, client):
        """Test that limit has a minimum value (1)."""
        response = client.post(
            "/search",
            json={
                "query": "test query",
                "limit": 0,  # below min of 1
            }
        )

        assert response.status_code == 422

    def test_search_default_limit(self, seeded_client):
        """Test that default limit is applied when not specified."""
        response = seeded_client.post(
            "/search",
            json={"query": "market analysis"}
        )

        assert response.status_code == 200
        data = response.json()
        # Default limit is 5, should not exceed that
        assert len(data["results"]) <= 5

    def test_search_results_sorted_by_score(self, seeded_client):
        """Test that results are sorted by similarity score descending."""
        response = seeded_client.post(
            "/search",
            json={"query": "growth analysis", "limit": 10}
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["results"]) > 1:
            scores = [r["score"] for r in data["results"]]
            assert scores == sorted(scores, reverse=True)

    def test_search_response_structure(self, client):
        """Test that response has correct structure."""
        response = client.post(
            "/search",
            json={"query": "test query"}
        )

        assert response.status_code == 200
        data = response.json()

        # Check top-level structure
        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["total_results"], int)

    def test_search_query_max_length(self, client):
        """Test that query has maximum length (500)."""
        long_query = "a" * 501  # exceeds max of 500

        response = client.post(
            "/search",
            json={"query": long_query}
        )

        assert response.status_code == 422

    def test_search_ticker_max_length(self, client):
        """Test that ticker has maximum length (10)."""
        response = client.post(
            "/search",
            json={
                "query": "test query",
                "ticker": "VERYLONGTICKER",  # exceeds max of 10
            }
        )

        assert response.status_code == 422
