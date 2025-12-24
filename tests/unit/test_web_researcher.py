"""
Unit tests for the Web Researcher Tool.
"""

import pytest

from src.adapters.search.mock_search import MockSearch
from src.domain.models import WebResearchOutput, WebSearchResult
from src.tools.web_researcher import WebResearcherTool


@pytest.fixture
def mock_search():
    """Provide a mock search provider."""
    return MockSearch()


@pytest.fixture
def web_researcher(mock_search):
    """Provide a web researcher tool with mock provider."""
    return WebResearcherTool(search_provider=mock_search)


class TestWebResearcherTool:
    """Tests for WebResearcherTool."""

    @pytest.mark.asyncio
    async def test_execute_returns_output(self, web_researcher):
        """Test that web researcher returns WebResearchOutput."""
        result = await web_researcher.execute(
            company="Tesla, Inc.",
            ticker="TSLA",
            research_focus="general",
        )

        assert result is not None
        assert isinstance(result, WebResearchOutput)
        assert len(result.results) > 0
        assert all(isinstance(r, WebSearchResult) for r in result.results)

    @pytest.mark.asyncio
    async def test_execute_with_different_ticker(self, web_researcher):
        """Test web researcher with different ticker."""
        result = await web_researcher.execute(
            company="Apple Inc.",
            ticker="AAPL",
            research_focus="general",
        )

        assert result is not None
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_execute_analyst_ratings_focus(self, web_researcher):
        """Test research with analyst ratings focus."""
        result = await web_researcher.execute(
            company="Microsoft Corporation",
            ticker="MSFT",
            research_focus="analyst_ratings",
        )

        assert result is not None
        assert result.research_focus == "analyst_ratings"
        assert len(result.queries_used) > 0
        # Analyst focus should include rating-related queries
        assert any("analyst" in q.lower() or "rating" in q.lower() for q in result.queries_used)

    @pytest.mark.asyncio
    async def test_execute_competitive_focus(self, web_researcher):
        """Test research with competitive focus."""
        result = await web_researcher.execute(
            company="Tesla, Inc.",
            ticker="TSLA",
            research_focus="competitive",
        )

        assert result is not None
        assert result.research_focus == "competitive"
        # Competitive focus should include competitor-related queries
        assert any("competitor" in q.lower() or "vs" in q.lower() or "market share" in q.lower()
                   for q in result.queries_used)

    @pytest.mark.asyncio
    async def test_execute_earnings_focus(self, web_researcher):
        """Test research with earnings focus."""
        result = await web_researcher.execute(
            company="Apple Inc.",
            ticker="AAPL",
            research_focus="earnings",
        )

        assert result is not None
        assert result.research_focus == "earnings"
        # Earnings focus should include earnings-related queries
        assert any("earnings" in q.lower() or "revenue" in q.lower() for q in result.queries_used)

    @pytest.mark.asyncio
    async def test_execute_risks_focus(self, web_researcher):
        """Test research with risks focus."""
        result = await web_researcher.execute(
            company="Alphabet Inc.",
            ticker="GOOGL",
            research_focus="risks",
        )

        assert result is not None
        assert result.research_focus == "risks"
        # Risks focus should include risk-related queries
        assert any("risk" in q.lower() or "concern" in q.lower() or "bearish" in q.lower()
                   for q in result.queries_used)

    @pytest.mark.asyncio
    async def test_execute_default_focus_is_general(self, web_researcher):
        """Test that default research focus is general."""
        result = await web_researcher.execute(
            company="Tesla, Inc.",
            ticker="TSLA",
        )

        assert result.research_focus == "general"

    @pytest.mark.asyncio
    async def test_execute_from_dict(self, web_researcher):
        """Test execution from dictionary parameters."""
        result = await web_researcher.execute_from_dict({
            "company": "Microsoft",
            "ticker": "MSFT",
            "research_focus": "general",
        })

        assert result is not None
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_execute_from_dict_default_focus(self, web_researcher):
        """Test execution from dict uses default focus."""
        result = await web_researcher.execute_from_dict({
            "company": "Tesla",
            "ticker": "TSLA",
        })

        assert result.research_focus == "general"

    @pytest.mark.asyncio
    async def test_results_are_deduplicated(self, web_researcher, mock_search):
        """Test that duplicate URLs are removed."""
        result = await web_researcher.execute(
            company="Tesla, Inc.",
            ticker="TSLA",
            research_focus="general",
        )

        # Check for unique URLs
        urls = [r.url for r in result.results]
        assert len(urls) == len(set(urls)), "Duplicate URLs found in results"

    @pytest.mark.asyncio
    async def test_results_are_limited(self, web_researcher):
        """Test that results are limited to max count."""
        result = await web_researcher.execute(
            company="Tesla, Inc.",
            ticker="TSLA",
            research_focus="general",
        )

        # Should be limited to 15 results
        assert len(result.results) <= 15

    @pytest.mark.asyncio
    async def test_total_results_count(self, web_researcher):
        """Test that total_results matches actual results."""
        result = await web_researcher.execute(
            company="Apple Inc.",
            ticker="AAPL",
            research_focus="general",
        )

        assert result.total_results == len(result.results)

    def test_tool_definition(self, web_researcher):
        """Test tool definition for LLM."""
        definition = web_researcher.get_tool_definition()

        assert definition["name"] == "web_researcher"
        assert "description" in definition
        assert "parameters" in definition
        assert "company" in definition["parameters"]["properties"]
        assert "ticker" in definition["parameters"]["properties"]
        assert "research_focus" in definition["parameters"]["properties"]

    def test_tool_definition_research_focus_enum(self, web_researcher):
        """Test that research_focus has valid enum values."""
        definition = web_researcher.get_tool_definition()
        focus_props = definition["parameters"]["properties"]["research_focus"]

        assert "enum" in focus_props
        expected_focuses = ["general", "news", "analyst_ratings", "competitive", "earnings", "risks"]
        assert set(focus_props["enum"]) == set(expected_focuses)

    def test_tool_definition_required_fields(self, web_researcher):
        """Test that required fields are specified."""
        definition = web_researcher.get_tool_definition()

        assert "required" in definition["parameters"]
        assert "company" in definition["parameters"]["required"]
        assert "ticker" in definition["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_relevance_sorting(self, web_researcher):
        """Test that results are sorted by relevance."""
        result = await web_researcher.execute(
            company="Tesla",
            ticker="TSLA",
            research_focus="general",
        )

        # First results should be more relevant (contain ticker/company)
        if len(result.results) >= 2:
            first_result = result.results[0]
            # First result should likely contain TSLA or Tesla
            first_text = (first_result.title + first_result.snippet).lower()
            assert "tsla" in first_text or "tesla" in first_text

    @pytest.mark.asyncio
    async def test_queries_include_company_and_ticker(self, web_researcher):
        """Test that queries include both company and ticker."""
        result = await web_researcher.execute(
            company="Tesla, Inc.",
            ticker="TSLA",
            research_focus="general",
        )

        # At least one query should include both company and ticker
        combined_queries = " ".join(result.queries_used).lower()
        assert "tesla" in combined_queries
        assert "tsla" in combined_queries

    @pytest.mark.asyncio
    async def test_handles_search_errors_gracefully(self, mock_search):
        """Test that search errors are handled gracefully."""
        # Create a failing mock search
        class FailingSearch(MockSearch):
            async def search(self, query, num_results=10):
                raise Exception("Search failed")

        failing_search = FailingSearch()
        web_researcher = WebResearcherTool(search_provider=failing_search)

        # Should not raise, should return with empty or partial results
        result = await web_researcher.execute(
            company="Tesla",
            ticker="TSLA",
            research_focus="general",
        )

        assert result is not None
        assert isinstance(result, WebResearchOutput)
