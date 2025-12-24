"""
API endpoint tests.
"""

import pytest
import time
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client):
        """Test health check returns healthy."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "version" in data


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""

    def test_analyze_creates_job(self, client):
        """Test that /analyze creates a job."""
        response = client.post(
            "/analyze",
            json={"query": "Analyze the near-term prospects of Tesla, Inc. (TSLA)"}
        )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "PENDING"
        assert "status_url" in data

    def test_analyze_rejects_short_query(self, client):
        """Test that short queries are rejected."""
        response = client.post(
            "/analyze",
            json={"query": "short"}
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_requires_query(self, client):
        """Test that query is required."""
        response = client.post(
            "/analyze",
            json={}
        )

        assert response.status_code == 422


class TestStatusEndpoint:
    """Tests for /status/{job_id} endpoint."""

    def test_status_returns_pending(self, client):
        """Test status for new job."""
        # Create a job first
        create_response = client.post(
            "/analyze",
            json={"query": "Analyze Tesla, Inc. (TSLA) stock"}
        )
        job_id = create_response.json()["job_id"]

        # Check status
        status_response = client.get(f"/status/{job_id}")

        assert status_response.status_code == 200
        data = status_response.json()
        assert data["job_id"] == job_id
        assert data["status"] in ["PENDING", "RUNNING", "COMPLETED"]

    def test_status_not_found(self, client):
        """Test status for non-existent job."""
        response = client.get("/status/non-existent-job-id")

        assert response.status_code == 404

    def test_full_analysis_flow(self, client):
        """Test complete analysis workflow."""
        # 1. Create analysis job
        create_response = client.post(
            "/analyze",
            json={"query": "Analyze the near-term prospects of Tesla, Inc. (TSLA)"}
        )
        assert create_response.status_code == 202
        job_id = create_response.json()["job_id"]

        # 2. Poll for completion (with timeout)
        max_attempts = 30
        for _ in range(max_attempts):
            status_response = client.get(f"/status/{job_id}")
            assert status_response.status_code == 200
            data = status_response.json()

            if data["status"] == "COMPLETED":
                # Verify result structure
                assert "result" in data
                result = data["result"]
                assert result["company_ticker"] == "TSLA"
                assert "analysis_summary" in result
                assert "sentiment_score" in result
                assert "key_findings" in result
                return

            if data["status"] == "FAILED":
                pytest.fail(f"Analysis failed: {data.get('error')}")

            time.sleep(0.5)

        pytest.fail("Analysis did not complete in time")


class TestMonitoringEndpoint:
    """Tests for /monitor_start endpoint."""

    def test_monitor_start_creates_schedule(self, client):
        """Test starting monitoring."""
        response = client.post(
            "/monitor_start",
            json={"ticker": "GOOGL", "interval_hours": 24}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["ticker"] == "GOOGL"
        assert data["status"] == "MONITORING_STARTED"
        assert "next_check_at" in data

    def test_monitor_start_normalizes_ticker(self, client):
        """Test that ticker is normalized to uppercase."""
        response = client.post(
            "/monitor_start",
            json={"ticker": "aapl"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["ticker"] == "AAPL"

    def test_monitor_duplicate_rejected(self, client):
        """Test that duplicate monitoring is rejected."""
        # First request should succeed
        response1 = client.post(
            "/monitor_start",
            json={"ticker": "NVDA"}
        )
        assert response1.status_code == 201

        # Second request should fail
        response2 = client.post(
            "/monitor_start",
            json={"ticker": "NVDA"}
        )
        assert response2.status_code == 409


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
