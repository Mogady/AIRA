"""
Custom Exceptions for A.I.R.A.

Hierarchical exception structure for clean error handling.
"""

from typing import Any, Dict, Optional


class AIRAException(Exception):
    """
    Base exception for all A.I.R.A. errors.

    Provides structured error information for logging and API responses.
    """

    def __init__(
        self,
        message: str,
        code: str = "AIRA_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Analysis Errors
# =============================================================================

class AnalysisError(AIRAException):
    """Errors related to the analysis process."""

    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="ANALYSIS_ERROR",
            details={**(details or {}), "job_id": job_id} if job_id else details,
        )
        self.job_id = job_id


class AnalysisNotFoundError(AnalysisError):
    """Raised when an analysis job is not found."""

    def __init__(self, job_id: str):
        super().__init__(
            message=f"Analysis job not found: {job_id}",
            job_id=job_id,
        )
        self.code = "ANALYSIS_NOT_FOUND"


class AnalysisTimeoutError(AnalysisError):
    """Raised when analysis exceeds timeout."""

    def __init__(self, job_id: str, timeout_seconds: int):
        super().__init__(
            message=f"Analysis timed out after {timeout_seconds} seconds",
            job_id=job_id,
            details={"timeout_seconds": timeout_seconds},
        )
        self.code = "ANALYSIS_TIMEOUT"


class MaxIterationsError(AnalysisError):
    """Raised when agent exceeds maximum iterations."""

    def __init__(self, job_id: str, max_iterations: int):
        super().__init__(
            message=f"Agent exceeded maximum iterations ({max_iterations})",
            job_id=job_id,
            details={"max_iterations": max_iterations},
        )
        self.code = "MAX_ITERATIONS_EXCEEDED"


# =============================================================================
# Tool Execution Errors
# =============================================================================

class ToolExecutionError(AIRAException):
    """Errors related to tool execution."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="TOOL_EXECUTION_ERROR",
            details={**(details or {}), "tool_name": tool_name},
        )
        self.tool_name = tool_name


class NewsRetrievalError(ToolExecutionError):
    """Errors from the news retriever tool."""

    def __init__(self, message: str, query: Optional[str] = None):
        super().__init__(
            message=message,
            tool_name="news_retriever",
            details={"query": query} if query else None,
        )
        self.code = "NEWS_RETRIEVAL_ERROR"


class SentimentAnalysisError(ToolExecutionError):
    """Errors from the sentiment analyzer tool."""

    def __init__(self, message: str, article_count: Optional[int] = None):
        super().__init__(
            message=message,
            tool_name="sentiment_analyzer",
            details={"article_count": article_count} if article_count else None,
        )
        self.code = "SENTIMENT_ANALYSIS_ERROR"


class DataFetcherError(ToolExecutionError):
    """Errors from the data fetcher tool."""

    def __init__(self, message: str, ticker: Optional[str] = None):
        super().__init__(
            message=message,
            tool_name="data_fetcher",
            details={"ticker": ticker} if ticker else None,
        )
        self.code = "DATA_FETCHER_ERROR"


class FinancialDataError(ToolExecutionError):
    """Errors from financial data providers (yfinance, etc.)."""

    def __init__(self, message: str, ticker: Optional[str] = None):
        super().__init__(
            message=message,
            tool_name="financial_provider",
            details={"ticker": ticker} if ticker else None,
        )
        self.code = "FINANCIAL_DATA_ERROR"
        self.ticker = ticker


# =============================================================================
# LLM Provider Errors
# =============================================================================

class LLMProviderError(AIRAException):
    """Errors from LLM providers (Claude, OpenAI, etc.)."""

    def __init__(
        self,
        message: str,
        provider: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="LLM_PROVIDER_ERROR",
            details={**(details or {}), "provider": provider},
        )
        self.provider = provider


class LLMRateLimitError(LLMProviderError):
    """Raised when LLM rate limit is hit."""

    def __init__(self, provider: str, retry_after: Optional[int] = None):
        super().__init__(
            message=f"Rate limit exceeded for {provider}",
            provider=provider,
            details={"retry_after_seconds": retry_after} if retry_after else None,
        )
        self.code = "LLM_RATE_LIMIT"
        self.retry_after = retry_after


class LLMTimeoutError(LLMProviderError):
    """Raised when LLM request times out."""

    def __init__(self, provider: str, timeout_seconds: int):
        super().__init__(
            message=f"LLM request timed out after {timeout_seconds} seconds",
            provider=provider,
            details={"timeout_seconds": timeout_seconds},
        )
        self.code = "LLM_TIMEOUT"


class LLMResponseError(LLMProviderError):
    """Raised when LLM returns invalid response."""

    def __init__(self, provider: str, reason: str):
        super().__init__(
            message=f"Invalid LLM response: {reason}",
            provider=provider,
            details={"reason": reason},
        )
        self.code = "LLM_INVALID_RESPONSE"


# =============================================================================
# Storage Errors
# =============================================================================

class StorageError(AIRAException):
    """Errors related to data storage."""

    def __init__(
        self,
        message: str,
        storage_type: str = "database",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            details={**(details or {}), "storage_type": storage_type},
        )
        self.storage_type = storage_type


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails."""

    def __init__(self, message: str = "Failed to connect to database"):
        super().__init__(message=message, storage_type="postgresql")
        self.code = "DATABASE_CONNECTION_ERROR"


class RedisConnectionError(StorageError):
    """Raised when Redis connection fails."""

    def __init__(self, message: str = "Failed to connect to Redis"):
        super().__init__(message=message, storage_type="redis")
        self.code = "REDIS_CONNECTION_ERROR"


class RecordNotFoundError(StorageError):
    """Raised when a database record is not found."""

    def __init__(self, record_type: str, record_id: str):
        super().__init__(
            message=f"{record_type} not found: {record_id}",
            details={"record_type": record_type, "record_id": record_id},
        )
        self.code = "RECORD_NOT_FOUND"


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(AIRAException):
    """Errors related to input validation."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details={**(details or {}), "field": field} if field else details,
        )
        self.field = field


class InvalidTickerError(ValidationError):
    """Raised when ticker symbol is invalid."""

    def __init__(self, ticker: str):
        super().__init__(
            message=f"Invalid ticker symbol: {ticker}",
            field="ticker",
            details={"ticker": ticker},
        )
        self.code = "INVALID_TICKER"


class InvalidQueryError(ValidationError):
    """Raised when analysis query is invalid."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid query: {reason}",
            field="query",
            details={"reason": reason},
        )
        self.code = "INVALID_QUERY"


# =============================================================================
# Monitoring Errors
# =============================================================================

class MonitoringError(AIRAException):
    """Errors related to the monitoring system."""

    def __init__(
        self,
        message: str,
        ticker: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="MONITORING_ERROR",
            details={**(details or {}), "ticker": ticker} if ticker else details,
        )
        self.ticker = ticker


class MonitoringAlreadyExistsError(MonitoringError):
    """Raised when trying to create duplicate monitoring."""

    def __init__(self, ticker: str):
        super().__init__(
            message=f"Monitoring already exists for ticker: {ticker}",
            ticker=ticker,
        )
        self.code = "MONITORING_ALREADY_EXISTS"


class MonitoringNotFoundError(MonitoringError):
    """Raised when monitoring schedule is not found."""

    def __init__(self, ticker: str):
        super().__init__(
            message=f"Monitoring not found for ticker: {ticker}",
            ticker=ticker,
        )
        self.code = "MONITORING_NOT_FOUND"
