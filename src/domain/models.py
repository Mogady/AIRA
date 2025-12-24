"""
Domain Models - Pydantic models for ALL I/O boundaries.

This module contains all Pydantic models used throughout A.I.R.A.:
- Tool inputs and outputs
- API request/response schemas
- Agent state management
- Analysis reports

"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================================================================
# Tool Input/Output Models
# =============================================================================

class NewsArticle(BaseModel):
    """A single news article from the news retriever tool."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Article headline")
    description: Optional[str] = Field(None, description="Article summary/description")
    url: str = Field(..., description="URL to the full article")
    source: str = Field(..., description="News source name")
    published_at: datetime = Field(..., description="Publication timestamp")
    content: Optional[str] = Field(None, description="Article content (may be truncated)")

    @field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, v: Any) -> str:
        """Ensure URL is a non-empty string."""
        if not v:
            raise ValueError("URL cannot be empty or None")
        return str(v)


class NewsRetrieverInput(BaseModel):
    """Input schema for the news retriever tool."""

    company: str = Field(..., min_length=1, description="Company name to search for")
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    num_articles: int = Field(default=5, ge=1, le=20, description="Number of articles to retrieve")


class NewsRetrieverOutput(BaseModel):
    """Output schema for the news retriever tool."""

    articles: List[NewsArticle] = Field(default_factory=list, description="Retrieved articles")
    query_used: str = Field(..., description="The search query that was executed")
    total_results: int = Field(default=0, ge=0, description="Total results available from API")


class ArticleSentiment(BaseModel):
    """Sentiment analysis for a single article."""

    title: str = Field(..., description="Article title")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Sentiment classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    key_phrases: List[str] = Field(default_factory=list, description="Key phrases influencing sentiment")


class SentimentResult(BaseModel):
    """Output schema for the sentiment analyzer tool."""

    positive_count: int = Field(default=0, ge=0, description="Number of positive articles")
    negative_count: int = Field(default=0, ge=0, description="Number of negative articles")
    neutral_count: int = Field(default=0, ge=0, description="Number of neutral articles")
    positive_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of positive articles")
    negative_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of negative articles")
    neutral_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of neutral articles")
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        ..., description="Overall sentiment determination"
    )
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Aggregated sentiment score: -1.0 (very negative) to 1.0 (very positive)"
    )
    article_sentiments: List[ArticleSentiment] = Field(
        default_factory=list, description="Per-article sentiment analysis"
    )
    analysis_reasoning: Optional[str] = Field(None, description="LLM's reasoning for the sentiment analysis")


class SentimentAnalyzerInput(BaseModel):
    """Input schema for the sentiment analyzer tool."""

    articles: List[NewsArticle] = Field(..., min_length=1, description="Articles to analyze")
    ticker: str = Field(..., description="Stock ticker for context")
    company_name: str = Field(..., description="Company name for context")


class QuarterlyRevenue(BaseModel):
    """Revenue data for a single quarter."""

    quarter: str = Field(..., description="Quarter identifier, e.g., 'Q3 2024'")
    revenue: float = Field(..., description="Revenue in USD")
    year_over_year_change: Optional[float] = Field(None, description="YoY percentage change")


class FinancialData(BaseModel):
    """Output schema for the data fetcher tool."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    current_price: Optional[float] = Field(None, ge=0, description="Current stock price in USD")
    price_change_percent: Optional[float] = Field(None, description="Price change percentage (24h)")
    market_cap: Optional[float] = Field(None, ge=0, description="Market capitalization in USD")
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio (trailing)")
    quarterly_revenue: List[QuarterlyRevenue] = Field(
        default_factory=list, description="Last 4 quarters of revenue data"
    )
    fifty_two_week_high: Optional[float] = Field(None, description="52-week high price")
    fifty_two_week_low: Optional[float] = Field(None, description="52-week low price")
    data_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When data was fetched")

    # Profitability metrics
    gross_margin: Optional[float] = Field(None, description="Gross profit margin as decimal (e.g., 0.25 = 25%)")
    operating_margin: Optional[float] = Field(None, description="Operating margin as decimal")
    profit_margin: Optional[float] = Field(None, description="Net profit margin as decimal")

    # Balance sheet health
    total_debt: Optional[float] = Field(None, ge=0, description="Total debt in USD")
    total_cash: Optional[float] = Field(None, ge=0, description="Cash and equivalents in USD")
    debt_to_equity: Optional[float] = Field(None, description="Debt to equity ratio")
    current_ratio: Optional[float] = Field(None, description="Current ratio (current assets / current liabilities)")

    # Growth metrics
    revenue_growth: Optional[float] = Field(None, description="YoY revenue growth as decimal")
    earnings_growth: Optional[float] = Field(None, description="YoY earnings growth as decimal")

    # Valuation metrics
    forward_pe: Optional[float] = Field(None, description="Forward P/E ratio")
    peg_ratio: Optional[float] = Field(None, description="PEG ratio (P/E to growth)")
    price_to_book: Optional[float] = Field(None, description="Price to book ratio")
    price_to_sales: Optional[float] = Field(None, description="Price to sales ratio (trailing 12 months)")
    enterprise_value: Optional[float] = Field(None, ge=0, description="Enterprise value in USD")
    ev_to_ebitda: Optional[float] = Field(None, description="EV to EBITDA ratio")

    # Analyst data
    analyst_target_mean: Optional[float] = Field(None, description="Mean analyst price target")
    analyst_target_low: Optional[float] = Field(None, description="Low analyst price target")
    analyst_target_high: Optional[float] = Field(None, description="High analyst price target")
    analyst_recommendation: Optional[str] = Field(None, description="Consensus recommendation (buy, hold, sell, etc.)")
    number_of_analysts: Optional[int] = Field(None, ge=0, description="Number of analysts covering the stock")

    # Dividend info
    dividend_yield: Optional[float] = Field(None, description="Annual dividend yield as decimal")
    payout_ratio: Optional[float] = Field(None, description="Dividend payout ratio")

    # Additional context
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")
    beta: Optional[float] = Field(None, description="Stock beta (volatility vs market)")
    short_ratio: Optional[float] = Field(None, description="Short interest ratio")


class DataFetcherInput(BaseModel):
    """Input schema for the data fetcher tool."""

    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")


# =============================================================================
# Web Research Models
# =============================================================================

class WebSearchResult(BaseModel):
    """A single web search result."""

    title: str = Field(..., description="Result title")
    snippet: str = Field(..., description="Result snippet/description")
    url: str = Field(..., description="Result URL")
    source: str = Field(default="", description="Source domain")


class WebResearchInput(BaseModel):
    """Input schema for the web researcher tool."""

    company: str = Field(..., min_length=1, description="Company name")
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    research_focus: str = Field(
        default="general",
        description="Research focus: general, analyst_ratings, competitive, earnings, risks"
    )


class WebResearchOutput(BaseModel):
    """Output schema for the web researcher tool."""

    results: List[WebSearchResult] = Field(default_factory=list, description="Search results")
    queries_used: List[str] = Field(default_factory=list, description="Search queries executed")
    research_focus: str = Field(default="general", description="Focus of research")
    total_results: int = Field(default=0, ge=0, description="Total results found")


# =============================================================================
# API Request/Response Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request schema for POST /analyze endpoint."""

    query: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Analysis query, e.g., 'Analyze the near-term prospects of Tesla, Inc. (TSLA)'",
        json_schema_extra={"example": "Analyze the near-term prospects of Tesla, Inc. (TSLA)"}
    )


class AnalyzeResponse(BaseModel):
    """Response schema for POST /analyze endpoint."""

    job_id: str = Field(..., description="Unique identifier for tracking the analysis job")
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = Field(
        default="PENDING", description="Current job status"
    )
    message: str = Field(default="Analysis job created successfully", description="Status message")
    status_url: str = Field(..., description="URL to poll for job status and results")


class AnalysisReport(BaseModel):
    """
    The final structured analysis report - main deliverable.

    This is the core output returned when an analysis job completes.
    """

    company_ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    analysis_summary: str = Field(
        ...,
        min_length=50,
        description="Concise paragraph synthesis of all findings"
    )
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Derived sentiment score between -1.0 (very negative) and 1.0 (very positive)"
    )
    key_findings: List[str] = Field(
        ...,
        min_length=3,
        max_length=7,
        description="Top 3-7 actionable insights from the analysis"
    )
    tools_used: List[str] = Field(..., description="List of tools executed during analysis")
    citation_sources: List[str] = Field(..., description="URLs of sources used in analysis")

    # Optional detailed sections
    news_summary: Optional[str] = Field(None, description="Summary of news coverage")
    financial_snapshot: Optional[Dict[str, Any]] = Field(None, description="Key financial metrics")

    # NEW: Enhanced analysis sections
    investment_thesis: Optional[str] = Field(
        None,
        description="Investment thesis with bull case, bear case, and balanced view"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Key risk factors investors should monitor"
    )
    competitive_context: Optional[str] = Field(
        None,
        description="Competitive positioning and market dynamics"
    )
    analyst_consensus: Optional[Dict[str, Any]] = Field(
        None,
        description="Summary of analyst ratings, price targets, and recommendations"
    )
    catalyst_events: List[str] = Field(
        default_factory=list,
        description="Upcoming events that could move the stock"
    )
    financial_analysis: Optional[str] = Field(
        None,
        description="Detailed financial analysis including valuation and profitability"
    )
    web_research: Optional[Dict[str, Any]] = Field(
        None,
        description="Web research results including search queries and snippets"
    )

    # Reflection metadata
    reflection_notes: Optional[str] = Field(
        None,
        description="Notes from self-correction if reflection loop was triggered"
    )
    reflection_triggered: bool = Field(
        default=False,
        description="Whether the reflection loop was triggered"
    )

    # Metadata
    analysis_type: Literal["ON_DEMAND", "PROACTIVE_ALERT"] = Field(
        default="ON_DEMAND", description="Type of analysis"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when report was generated"
    )
    iteration_count: int = Field(default=1, ge=1, description="Number of agent iterations")


class JobStatus(BaseModel):
    """Response schema for GET /status/{job_id} endpoint."""

    job_id: str = Field(..., description="Unique job identifier")
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = Field(
        ..., description="Current job status"
    )
    progress: Optional[str] = Field(None, description="Human-readable progress description")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    result: Optional[AnalysisReport] = Field(None, description="Analysis report (when completed)")
    error: Optional[str] = Field(None, description="Error message (when failed)")


# =============================================================================
# Monitoring Models
# =============================================================================

class MonitorStartRequest(BaseModel):
    """Request schema for POST /monitor_start endpoint."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker to monitor",
        json_schema_extra={"example": "TSLA"}
    )
    company_name: Optional[str] = Field(
        None,
        description="Company name (will be auto-detected if not provided)"
    )
    interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Monitoring interval in hours (1 to 168)"
    )


class MonitorStartResponse(BaseModel):
    """Response schema for POST /monitor_start endpoint."""

    ticker: str = Field(..., description="Monitored ticker symbol")
    status: str = Field(default="MONITORING_STARTED", description="Operation status")
    next_check_at: datetime = Field(..., description="Scheduled time for next check")
    message: str = Field(..., description="Status message")


class MonitoringSchedule(BaseModel):
    """Internal model for monitoring schedule tracking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    ticker: str
    company_name: Optional[str] = None
    interval_hours: int = 24
    last_check_at: Optional[datetime] = None
    last_analysis_id: Optional[str] = None
    article_hashes: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Database Models (for SQLAlchemy mapping reference)
# =============================================================================

class AnalysisRecord(BaseModel):
    """Database record for an analysis job."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str
    company_ticker: str
    company_name: Optional[str] = None
    user_query: str
    analysis_type: Literal["ON_DEMAND", "PROACTIVE_ALERT"] = "ON_DEMAND"
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = "PENDING"
    progress: Optional[str] = None  # Human-readable progress description
    report: Optional[Dict[str, Any]] = None  # Serialized AnalysisReport
    error_message: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    iteration_count: int = 0
    reflection_triggered: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class ToolExecutionRecord(BaseModel):
    """Database record for tool execution logging."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    analysis_id: str
    tool_name: str
    input_params: Dict[str, Any]
    output_result: Optional[Dict[str, Any]] = None
    execution_time_ms: int
    success: bool = True
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentThoughtRecord(BaseModel):
    """Database record for agent thought logging."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    analysis_id: str
    step_number: int
    thought_type: Literal["planning", "tool_selection", "reflection", "synthesis", "error"]
    thought_content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Semantic Search Models
# =============================================================================

class SearchRequest(BaseModel):
    """Request schema for semantic search endpoint."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Search query for semantic similarity",
    )
    ticker: Optional[str] = Field(
        None,
        max_length=10,
        description="Optional ticker symbol to filter results",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return",
    )


class SearchResult(BaseModel):
    """A single semantic search result."""

    analysis_id: str = Field(..., description="ID of the parent analysis")
    content_type: Literal["summary", "key_finding"] = Field(
        ...,
        description="Type of content matched",
    )
    content_text: str = Field(..., description="The matched text content")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (0-1, higher is more similar)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (ticker, date, sentiment, etc.)",
    )


class SearchResponse(BaseModel):
    """Response schema for semantic search endpoint."""

    query: str = Field(..., description="The original search query")
    results: List[SearchResult] = Field(
        default_factory=list,
        description="List of search results",
    )
    total_results: int = Field(..., description="Total number of results returned")


class SentimentHistoryPoint(BaseModel):
    """A single point in sentiment history."""

    job_id: str = Field(..., description="Analysis job ID")
    date: Optional[datetime] = Field(None, description="Date of the analysis")
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score at this point",
    )
