"""
Unit tests for domain models.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.domain.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisReport,
    JobStatus,
    NewsArticle,
    NewsRetrieverInput,
    NewsRetrieverOutput,
    SentimentResult,
    FinancialData,
    QuarterlyRevenue,
    AgentState,
    WebSearchResult,
    WebResearchInput,
    WebResearchOutput,
)


class TestAnalyzeRequest:
    """Tests for AnalyzeRequest model."""

    def test_valid_request(self):
        """Test valid analysis request."""
        request = AnalyzeRequest(
            query="Analyze the near-term prospects of Tesla, Inc. (TSLA)"
        )
        assert request.query == "Analyze the near-term prospects of Tesla, Inc. (TSLA)"

    def test_query_too_short(self):
        """Test that short queries are rejected."""
        with pytest.raises(ValidationError):
            AnalyzeRequest(query="short")

    def test_query_too_long(self):
        """Test that long queries are rejected."""
        with pytest.raises(ValidationError):
            AnalyzeRequest(query="x" * 501)


class TestAnalyzeResponse:
    """Tests for AnalyzeResponse model."""

    def test_valid_response(self):
        """Test valid analysis response."""
        response = AnalyzeResponse(
            job_id="abc123",
            status="PENDING",
            message="Job created",
            status_url="/status/abc123",
        )
        assert response.job_id == "abc123"
        assert response.status == "PENDING"


class TestAnalysisReport:
    """Tests for AnalysisReport model."""

    def test_valid_report(self):
        """Test valid analysis report."""
        report = AnalysisReport(
            company_ticker="TSLA",
            company_name="Tesla, Inc.",
            analysis_summary="Tesla continues to demonstrate strong market position in the electric vehicle sector with recent positive momentum.",
            sentiment_score=0.45,
            key_findings=[
                "Strong Q3 deliveries",
                "Expanding production capacity",
                "Increased competition in EV market",
            ],
            tools_used=["news_retriever", "sentiment_analyzer", "data_fetcher"],
            citation_sources=["https://example.com/article1"],
        )
        assert report.company_ticker == "TSLA"
        assert report.sentiment_score == 0.45
        assert len(report.key_findings) == 3

    def test_sentiment_score_bounds(self):
        """Test sentiment score validation."""
        # Valid: exactly -1.0
        report = AnalysisReport(
            company_ticker="TSLA",
            company_name="Tesla",
            analysis_summary="x" * 50,
            sentiment_score=-1.0,
            key_findings=["a", "b", "c"],
            tools_used=["tool1"],
            citation_sources=[],
        )
        assert report.sentiment_score == -1.0

        # Invalid: below -1.0
        with pytest.raises(ValidationError):
            AnalysisReport(
                company_ticker="TSLA",
                company_name="Tesla",
                analysis_summary="x" * 50,
                sentiment_score=-1.5,
                key_findings=["a", "b", "c"],
                tools_used=["tool1"],
                citation_sources=[],
            )

    def test_minimum_key_findings(self):
        """Test minimum key findings requirement."""
        with pytest.raises(ValidationError):
            AnalysisReport(
                company_ticker="TSLA",
                company_name="Tesla",
                analysis_summary="x" * 50,
                sentiment_score=0.0,
                key_findings=["only", "two"],  # Need at least 3
                tools_used=["tool1"],
                citation_sources=[],
            )


class TestNewsArticle:
    """Tests for NewsArticle model."""

    def test_valid_article(self):
        """Test valid news article."""
        article = NewsArticle(
            title="Tesla Reports Record Deliveries",
            description="Electric vehicle maker exceeds expectations",
            url="https://example.com/news/tesla",
            source="Financial Times",
            published_at=datetime.now(timezone.utc),
            content="Full article content here",
        )
        assert article.title == "Tesla Reports Record Deliveries"
        assert article.source == "Financial Times"

    def test_optional_fields(self):
        """Test article with optional fields missing."""
        article = NewsArticle(
            title="Test Article",
            url="https://example.com",
            source="Test Source",
            published_at=datetime.now(timezone.utc),
        )
        assert article.description is None
        assert article.content is None


class TestSentimentResult:
    """Tests for SentimentResult model."""

    def test_valid_sentiment(self):
        """Test valid sentiment result."""
        result = SentimentResult(
            positive_count=3,
            negative_count=1,
            neutral_count=1,
            positive_ratio=0.6,
            negative_ratio=0.2,
            neutral_ratio=0.2,
            overall_sentiment="positive",
            sentiment_score=0.45,
            article_sentiments=[],
        )
        assert result.overall_sentiment == "positive"
        assert result.sentiment_score == 0.45

    def test_ratio_bounds(self):
        """Test ratio validation."""
        with pytest.raises(ValidationError):
            SentimentResult(
                positive_count=1,
                negative_count=1,
                neutral_count=1,
                positive_ratio=1.5,  # Invalid: > 1.0
                negative_ratio=0.0,
                neutral_ratio=0.0,
                overall_sentiment="neutral",
                sentiment_score=0.0,
            )


class TestFinancialData:
    """Tests for FinancialData model."""

    def test_valid_financial_data(self):
        """Test valid financial data."""
        data = FinancialData(
            ticker="TSLA",
            company_name="Tesla, Inc.",
            current_price=248.50,
            price_change_percent=2.35,
            market_cap=789_000_000_000,
            pe_ratio=62.5,
            quarterly_revenue=[
                QuarterlyRevenue(
                    quarter="Q3 2024",
                    revenue=25_180_000_000,
                    year_over_year_change=8.2,
                ),
            ],
            data_timestamp=datetime.now(timezone.utc),
        )
        assert data.ticker == "TSLA"
        assert data.current_price == 248.50
        assert len(data.quarterly_revenue) == 1


class TestAgentState:
    """Tests for AgentState model."""

    def test_initial_state(self):
        """Test initial agent state."""
        state = AgentState(
            job_id="test-123",
            query="Analyze TSLA",
        )
        assert state.job_id == "test-123"
        assert state.iteration == 0
        assert state.reflection_triggered is False
        assert state.status == "PENDING"

    def test_state_with_results(self):
        """Test agent state with tool results."""
        state = AgentState(
            job_id="test-123",
            query="Analyze TSLA",
            ticker="TSLA",
            company_name="Tesla, Inc.",
            tools_used=["news_retriever", "sentiment_analyzer"],
            iteration=3,
        )
        assert state.ticker == "TSLA"
        assert len(state.tools_used) == 2
        assert state.iteration == 3


class TestFinancialDataExtended:
    """Tests for new FinancialData fields."""

    def test_financial_data_with_profitability_metrics(self):
        """Test financial data with profitability metrics."""
        data = FinancialData(
            ticker="AAPL",
            company_name="Apple Inc.",
            gross_margin=0.438,
            operating_margin=0.297,
            profit_margin=0.253,
        )
        assert data.gross_margin == 0.438
        assert data.operating_margin == 0.297
        assert data.profit_margin == 0.253

    def test_financial_data_with_balance_sheet_metrics(self):
        """Test financial data with balance sheet metrics."""
        data = FinancialData(
            ticker="MSFT",
            company_name="Microsoft Corporation",
            total_debt=75_000_000_000,
            total_cash=80_000_000_000,
            debt_to_equity=0.42,
            current_ratio=1.77,
        )
        assert data.total_debt == 75_000_000_000
        assert data.total_cash == 80_000_000_000
        assert data.debt_to_equity == 0.42
        assert data.current_ratio == 1.77

    def test_financial_data_with_growth_metrics(self):
        """Test financial data with growth metrics."""
        data = FinancialData(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            revenue_growth=1.22,  # 122% growth
            earnings_growth=0.94,  # 94% growth
        )
        assert data.revenue_growth == 1.22
        assert data.earnings_growth == 0.94

    def test_financial_data_with_valuation_metrics(self):
        """Test financial data with valuation metrics."""
        data = FinancialData(
            ticker="GOOGL",
            company_name="Alphabet Inc.",
            forward_pe=22.5,
            peg_ratio=1.2,
            price_to_book=5.8,
            price_to_sales=5.5,
            enterprise_value=1_800_000_000_000,
            ev_to_ebitda=15.2,
        )
        assert data.forward_pe == 22.5
        assert data.peg_ratio == 1.2
        assert data.price_to_book == 5.8
        assert data.price_to_sales == 5.5
        assert data.enterprise_value == 1_800_000_000_000
        assert data.ev_to_ebitda == 15.2

    def test_financial_data_with_analyst_data(self):
        """Test financial data with analyst data."""
        data = FinancialData(
            ticker="TSLA",
            company_name="Tesla, Inc.",
            analyst_target_mean=275.00,
            analyst_target_low=150.00,
            analyst_target_high=400.00,
            analyst_recommendation="hold",
            number_of_analysts=45,
        )
        assert data.analyst_target_mean == 275.00
        assert data.analyst_target_low == 150.00
        assert data.analyst_target_high == 400.00
        assert data.analyst_recommendation == "hold"
        assert data.number_of_analysts == 45

    def test_financial_data_with_dividend_info(self):
        """Test financial data with dividend info."""
        data = FinancialData(
            ticker="JNJ",
            company_name="Johnson & Johnson",
            dividend_yield=0.03,  # 3%
            payout_ratio=0.45,  # 45%
        )
        assert data.dividend_yield == 0.03
        assert data.payout_ratio == 0.45

    def test_financial_data_with_context_fields(self):
        """Test financial data with sector/industry context."""
        data = FinancialData(
            ticker="META",
            company_name="Meta Platforms, Inc.",
            sector="Technology",
            industry="Internet Content & Information",
            beta=1.25,
            short_ratio=2.5,
        )
        assert data.sector == "Technology"
        assert data.industry == "Internet Content & Information"
        assert data.beta == 1.25
        assert data.short_ratio == 2.5

    def test_financial_data_all_fields(self):
        """Test financial data with all fields populated."""
        data = FinancialData(
            ticker="AAPL",
            company_name="Apple Inc.",
            current_price=178.25,
            price_change_percent=0.85,
            market_cap=2_780_000_000_000,
            pe_ratio=28.4,
            quarterly_revenue=[
                QuarterlyRevenue(quarter="Q4 2024", revenue=89_500_000_000, year_over_year_change=5.5),
            ],
            fifty_two_week_high=199.62,
            fifty_two_week_low=164.08,
            gross_margin=0.438,
            operating_margin=0.297,
            profit_margin=0.253,
            total_debt=110_000_000_000,
            total_cash=65_000_000_000,
            debt_to_equity=1.95,
            current_ratio=0.99,
            revenue_growth=0.05,
            earnings_growth=0.08,
            forward_pe=26.5,
            peg_ratio=2.5,
            price_to_book=47.8,
            price_to_sales=7.5,
            enterprise_value=2_850_000_000_000,
            ev_to_ebitda=21.5,
            analyst_target_mean=200.00,
            analyst_target_low=160.00,
            analyst_target_high=250.00,
            analyst_recommendation="buy",
            number_of_analysts=40,
            dividend_yield=0.005,
            payout_ratio=0.15,
            sector="Technology",
            industry="Consumer Electronics",
            beta=1.28,
            short_ratio=1.2,
        )
        assert data.ticker == "AAPL"
        assert data.gross_margin == 0.438
        assert data.analyst_recommendation == "buy"
        assert data.sector == "Technology"


class TestWebSearchResult:
    """Tests for WebSearchResult model."""

    def test_valid_search_result(self):
        """Test valid web search result."""
        result = WebSearchResult(
            title="Tesla Stock Analysis 2024",
            snippet="Comprehensive analysis of Tesla stock performance...",
            url="https://example.com/tesla-analysis",
            source="example.com",
        )
        assert result.title == "Tesla Stock Analysis 2024"
        assert result.url == "https://example.com/tesla-analysis"
        assert result.source == "example.com"

    def test_search_result_default_source(self):
        """Test search result with default source."""
        result = WebSearchResult(
            title="Test Result",
            snippet="Test snippet",
            url="https://test.com",
        )
        assert result.source == ""


class TestWebResearchInput:
    """Tests for WebResearchInput model."""

    def test_valid_research_input(self):
        """Test valid web research input."""
        input_data = WebResearchInput(
            company="Tesla, Inc.",
            ticker="TSLA",
            research_focus="analyst_ratings",
        )
        assert input_data.company == "Tesla, Inc."
        assert input_data.ticker == "TSLA"
        assert input_data.research_focus == "analyst_ratings"

    def test_research_input_default_focus(self):
        """Test research input default focus."""
        input_data = WebResearchInput(
            company="Apple Inc.",
            ticker="AAPL",
        )
        assert input_data.research_focus == "general"

    def test_research_input_ticker_validation(self):
        """Test ticker validation."""
        with pytest.raises(ValidationError):
            WebResearchInput(
                company="Test",
                ticker="TOOLONGTICKER123",  # > 10 chars
            )


class TestWebResearchOutput:
    """Tests for WebResearchOutput model."""

    def test_valid_research_output(self):
        """Test valid web research output."""
        output = WebResearchOutput(
            results=[
                WebSearchResult(
                    title="Test",
                    snippet="Test snippet",
                    url="https://test.com",
                    source="test.com",
                )
            ],
            queries_used=["TSLA stock analysis"],
            research_focus="general",
            total_results=1,
        )
        assert len(output.results) == 1
        assert output.research_focus == "general"
        assert output.total_results == 1

    def test_research_output_defaults(self):
        """Test research output defaults."""
        output = WebResearchOutput()
        assert output.results == []
        assert output.queries_used == []
        assert output.research_focus == "general"
        assert output.total_results == 0


class TestAnalysisReportExtended:
    """Tests for new AnalysisReport fields."""

    def test_report_with_investment_thesis(self):
        """Test analysis report with investment thesis."""
        report = AnalysisReport(
            company_ticker="TSLA",
            company_name="Tesla, Inc.",
            analysis_summary="Tesla continues to demonstrate strong market position in the EV sector.",
            sentiment_score=0.45,
            key_findings=["Strong deliveries", "Growing margins", "Market leader"],
            tools_used=["news_retriever"],
            citation_sources=[],
            investment_thesis="Bull case: EV market leader. Bear case: Competition increasing. Balanced: Hold with caution.",
        )
        assert report.investment_thesis is not None
        assert "Bull case" in report.investment_thesis

    def test_report_with_risk_factors(self):
        """Test analysis report with risk factors."""
        report = AnalysisReport(
            company_ticker="AAPL",
            company_name="Apple Inc.",
            analysis_summary="Apple maintains strong ecosystem with steady growth despite market headwinds.",
            sentiment_score=0.3,
            key_findings=["Strong services", "iPhone growth", "China concerns"],
            tools_used=["news_retriever"],
            citation_sources=[],
            risk_factors=[
                "China regulatory risk",
                "iPhone saturation in key markets",
                "Increased competition in services",
            ],
        )
        assert len(report.risk_factors) == 3
        assert "China regulatory risk" in report.risk_factors

    def test_report_with_analyst_consensus(self):
        """Test analysis report with analyst consensus."""
        report = AnalysisReport(
            company_ticker="MSFT",
            company_name="Microsoft Corporation",
            analysis_summary="Microsoft's cloud business continues to drive growth with AI integration.",
            sentiment_score=0.6,
            key_findings=["Azure growth", "AI integration", "Strong margins"],
            tools_used=["news_retriever", "data_fetcher"],
            citation_sources=[],
            analyst_consensus={
                "rating": "buy",
                "target_mean": 450.00,
                "target_low": 380.00,
                "target_high": 520.00,
                "analyst_count": 38,
            },
        )
        assert report.analyst_consensus is not None
        assert report.analyst_consensus["rating"] == "buy"

    def test_report_with_catalyst_events(self):
        """Test analysis report with catalyst events."""
        report = AnalysisReport(
            company_ticker="NVDA",
            company_name="NVIDIA Corporation",
            analysis_summary="NVIDIA's AI dominance continues with strong datacenter growth.",
            sentiment_score=0.75,
            key_findings=["AI leadership", "Datacenter growth", "Gaming recovery"],
            tools_used=["news_retriever"],
            citation_sources=[],
            catalyst_events=[
                "Q4 earnings release in February",
                "GTC conference in March",
                "New product announcements expected",
            ],
        )
        assert len(report.catalyst_events) == 3

    def test_report_with_financial_analysis(self):
        """Test analysis report with financial analysis."""
        report = AnalysisReport(
            company_ticker="GOOGL",
            company_name="Alphabet Inc.",
            analysis_summary="Alphabet's search dominance and cloud growth drive solid fundamentals.",
            sentiment_score=0.5,
            key_findings=["Search revenue", "YouTube growth", "Cloud momentum"],
            tools_used=["news_retriever", "data_fetcher"],
            citation_sources=[],
            financial_analysis="Trading at 22x forward earnings with 15% YoY revenue growth. Strong balance sheet with $100B+ cash position.",
        )
        assert report.financial_analysis is not None
        assert "22x forward earnings" in report.financial_analysis

    def test_report_with_competitive_context(self):
        """Test analysis report with competitive context."""
        report = AnalysisReport(
            company_ticker="META",
            company_name="Meta Platforms, Inc.",
            analysis_summary="Meta's advertising recovery and AI investments position it well for growth.",
            sentiment_score=0.55,
            key_findings=["Ad recovery", "Reels growth", "AI investment"],
            tools_used=["news_retriever", "web_researcher"],
            citation_sources=[],
            competitive_context="Competing with TikTok for short-form video. Strong position in VR/AR vs Apple Vision Pro.",
        )
        assert report.competitive_context is not None
        assert "TikTok" in report.competitive_context
