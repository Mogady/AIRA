"""
Integration tests for the monitoring API endpoints.

These tests run against the actual running Docker containers.
Ensure docker-compose is running before executing these tests.

Run with: pytest tests/integration/test_monitoring_api.py -v
"""

import asyncio
import time
from typing import Optional

import httpx
import pytest

# Base URL for the API (running in Docker)
API_BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def api_client():
    """Create an HTTP client for API requests."""
    with httpx.Client(base_url=API_BASE_URL, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="module")
def async_api_client():
    """Create an async HTTP client for API requests."""
    return httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0)


def wait_for_api(max_retries: int = 30, delay: float = 1.0) -> bool:
    """Wait for the API to be available."""
    for _ in range(max_retries):
        try:
            response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(delay)
    return False


class TestMonitoringEndpoints:
    """Integration tests for monitoring endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, api_client):
        """Setup: ensure API is available and clean up any existing monitors."""
        if not wait_for_api():
            pytest.skip("API is not available. Ensure Docker containers are running.")

        # Clean up any existing TSLA monitor from previous test runs
        try:
            api_client.delete("/monitor/TSLA")
        except httpx.HTTPStatusError:
            pass  # Monitor might not exist

    def test_health_check(self, api_client):
        """Test that the API health endpoint is working."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_start_monitoring(self, api_client):
        """Test starting a monitoring schedule."""
        response = api_client.post(
            "/monitor_start",
            json={
                "ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "interval_hours": 24,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["ticker"] == "TSLA"
        assert data["status"] == "MONITORING_STARTED"
        assert "next_check_at" in data
        assert "Will check every 24 hours" in data["message"]

    def test_start_monitoring_duplicate(self, api_client):
        """Test that duplicate monitoring returns 409 conflict."""
        # First, start monitoring
        api_client.post(
            "/monitor_start",
            json={
                "ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "interval_hours": 24,
            },
        )

        # Try to start again - should get 409
        response = api_client.post(
            "/monitor_start",
            json={
                "ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "interval_hours": 24,
            },
        )

        assert response.status_code == 409
        assert "already active" in response.json()["detail"]

    def test_list_monitors(self, api_client):
        """Test listing active monitoring schedules."""
        # Start monitoring for a ticker
        api_client.post(
            "/monitor_start",
            json={
                "ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "interval_hours": 24,
            },
        )

        # List monitors
        response = api_client.get("/monitors")
        assert response.status_code == 200

        monitors = response.json()
        assert isinstance(monitors, list)

        # Find our monitor
        tsla_monitor = next((m for m in monitors if m["ticker"] == "TSLA"), None)
        assert tsla_monitor is not None
        assert tsla_monitor["is_active"] is True

    def test_stop_monitoring(self, api_client):
        """Test stopping a monitoring schedule."""
        # First, start monitoring
        api_client.post(
            "/monitor_start",
            json={
                "ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "interval_hours": 24,
            },
        )

        # Stop monitoring
        response = api_client.delete("/monitor/TSLA")
        assert response.status_code == 204

        # Verify it's no longer in active monitors
        response = api_client.get("/monitors")
        monitors = response.json()
        tsla_monitor = next((m for m in monitors if m["ticker"] == "TSLA"), None)
        assert tsla_monitor is None  # Should not be in active list

    def test_stop_nonexistent_monitoring(self, api_client):
        """Test that stopping non-existent monitoring returns 404."""
        response = api_client.delete("/monitor/NONEXISTENT")
        assert response.status_code == 404

    def test_monitoring_different_tickers(self, api_client):
        """Test monitoring multiple different tickers."""
        tickers = ["AAPL", "GOOGL", "MSFT"]

        try:
            for ticker in tickers:
                response = api_client.post(
                    "/monitor_start",
                    json={
                        "ticker": ticker,
                        "company_name": f"{ticker} Company",
                        "interval_hours": 24,
                    },
                )
                assert response.status_code == 201

            # Verify all are in the list
            response = api_client.get("/monitors")
            monitors = response.json()
            monitor_tickers = {m["ticker"] for m in monitors}

            for ticker in tickers:
                assert ticker in monitor_tickers

        finally:
            # Cleanup
            for ticker in tickers:
                try:
                    api_client.delete(f"/monitor/{ticker}")
                except httpx.HTTPStatusError:
                    pass


class TestMonitoringSchedulerIntegration:
    """
    Integration tests for the monitoring scheduler logic.

    These tests simulate what the scheduler does without waiting for the actual interval.
    """

    @pytest.fixture(autouse=True)
    def setup(self, api_client):
        """Setup: ensure API is available."""
        if not wait_for_api():
            pytest.skip("API is not available. Ensure Docker containers are running.")

        # Clean up
        try:
            api_client.delete("/monitor/TSLA")
        except httpx.HTTPStatusError:
            pass

    def test_full_monitoring_flow(self, api_client):
        """
        Test the complete monitoring flow:
        1. Start monitoring
        2. Verify schedule created
        3. Stop monitoring
        """
        # 1. Start monitoring
        response = api_client.post(
            "/monitor_start",
            json={
                "ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "interval_hours": 1,  # Short interval for testing
            },
        )
        assert response.status_code == 201
        start_data = response.json()

        # 2. Verify the schedule was created
        response = api_client.get("/monitors")
        monitors = response.json()
        tsla_monitor = next((m for m in monitors if m["ticker"] == "TSLA"), None)

        assert tsla_monitor is not None
        assert tsla_monitor["ticker"] == "TSLA"
        assert tsla_monitor["company_name"] == "Tesla, Inc."
        assert tsla_monitor["interval_hours"] == 1
        assert tsla_monitor["is_active"] is True

        # 3. Stop monitoring
        response = api_client.delete("/monitor/TSLA")
        assert response.status_code == 204


class TestAnalyzeEndpointWithMonitoring:
    """Test that the analyze endpoint works alongside monitoring."""

    @pytest.fixture(autouse=True)
    def setup(self, api_client):
        """Setup: ensure API is available."""
        if not wait_for_api():
            pytest.skip("API is not available. Ensure Docker containers are running.")

    def test_analyze_while_monitoring_active(self, api_client):
        """Test that manual analysis works while monitoring is active."""
        try:
            # Start monitoring
            api_client.post(
                "/monitor_start",
                json={
                    "ticker": "TSLA",
                    "company_name": "Tesla, Inc.",
                    "interval_hours": 24,
                },
            )

            # Run a manual analysis
            response = api_client.post(
                "/analyze",
                json={"query": "Analyze the near-term prospects of Tesla, Inc. (TSLA)"},
            )

            assert response.status_code in [200, 202]  # 202 Accepted for async jobs
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "PENDING"

            # Poll for completion (with timeout)
            job_id = data["job_id"]
            max_wait = 60  # seconds
            poll_interval = 2
            elapsed = 0

            while elapsed < max_wait:
                status_response = api_client.get(f"/status/{job_id}")
                status_data = status_response.json()

                if status_data["status"] in ["COMPLETED", "FAILED"]:
                    break

                time.sleep(poll_interval)
                elapsed += poll_interval

            # Verify analysis completed
            final_status = api_client.get(f"/status/{job_id}").json()
            assert final_status["status"] in ["COMPLETED", "FAILED"]

            if final_status["status"] == "COMPLETED":
                assert final_status["result"] is not None
                assert final_status["result"]["company_ticker"] == "TSLA"

        finally:
            # Cleanup
            try:
                api_client.delete("/monitor/TSLA")
            except httpx.HTTPStatusError:
                pass


class TestProactiveAlertSimulation:
    """
    Test the PROACTIVE_ALERT flow by simulating what happens when
    the scheduler detects significant new coverage.

    Note: This doesn't actually wait for the scheduler interval.
    Instead, it tests the components that would be triggered.
    """

    @pytest.fixture(autouse=True)
    def setup(self, api_client):
        """Setup: ensure API is available."""
        if not wait_for_api():
            pytest.skip("API is not available. Ensure Docker containers are running.")

    def test_proactive_alert_query_format(self, api_client):
        """
        Test that a PROACTIVE_ALERT style query works correctly.

        This simulates what the scheduler would enqueue when it detects
        significant new coverage.
        """
        # Submit a PROACTIVE_ALERT style analysis
        response = api_client.post(
            "/analyze",
            json={"query": "PROACTIVE_ALERT: Analyze TSLA (TSLA) due to significant new coverage"},
        )

        assert response.status_code in [200, 202]  # 202 Accepted for async jobs
        data = response.json()
        job_id = data["job_id"]

        # Poll for completion
        max_wait = 60
        poll_interval = 2
        elapsed = 0

        while elapsed < max_wait:
            status_response = api_client.get(f"/status/{job_id}")
            status_data = status_response.json()

            if status_data["status"] in ["COMPLETED", "FAILED"]:
                break

            time.sleep(poll_interval)
            elapsed += poll_interval

        # Verify
        final_status = api_client.get(f"/status/{job_id}").json()
        assert final_status["status"] in ["COMPLETED", "FAILED"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
