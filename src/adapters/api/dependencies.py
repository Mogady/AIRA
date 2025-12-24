"""
Dependency Injection for FastAPI.

This module provides dependency injection for all adapters,
allowing easy swapping between mock and real implementations
based on configuration.
"""

import asyncio
import threading
from typing import Optional

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from src.adapters.agents.langgraph_agent import AIRAAgent
from src.adapters.embeddings.mock_embeddings import MockEmbeddings
from src.adapters.financial.mock_financial import MockFinancial
from src.adapters.llm.mock_llm import MockLLM
from src.adapters.news.mock_news import MockNews
from src.adapters.storage.memory_repository import MemoryRepository
from src.application.ports.embeddings_port import EmbeddingsPort
from src.application.ports.financial_port import FinancialPort
from src.application.ports.llm_port import LLMPort
from src.application.ports.news_port import NewsPort
from src.application.ports.search_port import SearchPort
from src.application.ports.storage_port import StoragePort
from src.config.settings import get_settings
from src.tools.data_fetcher import DataFetcherTool
from src.tools.news_retriever import NewsRetrieverTool
from src.tools.sentiment_analyzer import SentimentAnalyzerTool
from src.tools.web_researcher import WebResearcherTool


# Thread-safe singleton management (RLock allows same thread to re-acquire)
_lock = threading.RLock()
_llm_provider: Optional[LLMPort] = None
_news_provider: Optional[NewsPort] = None
_financial_provider: Optional[FinancialPort] = None
_embeddings_provider: Optional[EmbeddingsPort] = None
_search_provider: Optional[SearchPort] = None
_storage: Optional[StoragePort] = None
_agent: Optional[AIRAAgent] = None
_arq_pool: Optional[ArqRedis] = None
_arq_pool_lock: Optional[asyncio.Lock] = None


def get_llm_provider() -> LLMPort:
    """Get the LLM provider based on configuration (thread-safe)."""
    global _llm_provider

    if _llm_provider is not None:
        return _llm_provider

    with _lock:
        # Double-check after acquiring lock
        if _llm_provider is not None:
            return _llm_provider

        settings = get_settings()

        if settings.llm.provider == "mock":
            _llm_provider = MockLLM()
        elif settings.llm.provider == "anthropic":
            from src.adapters.llm.claude_provider import ClaudeProvider
            _llm_provider = ClaudeProvider(
                api_key=settings.llm.api_key,
                model=settings.llm.model,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
                timeout=settings.llm.timeout,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")

        return _llm_provider


def get_news_provider() -> NewsPort:
    """Get the news provider based on configuration (thread-safe)."""
    global _news_provider

    if _news_provider is not None:
        return _news_provider

    with _lock:
        if _news_provider is not None:
            return _news_provider

        settings = get_settings()

        if settings.news.provider == "mock":
            _news_provider = MockNews()
        elif settings.news.provider == "newsapi":
            from src.adapters.news.newsapi_client import NewsAPIClient
            _news_provider = NewsAPIClient(
                api_key=settings.news.api_key,
                base_url=settings.news.base_url,
            )
        else:
            raise ValueError(f"Unknown news provider: {settings.news.provider}")

        return _news_provider


def get_financial_provider() -> FinancialPort:
    """Get the financial data provider based on configuration (thread-safe)."""
    global _financial_provider

    if _financial_provider is not None:
        return _financial_provider

    with _lock:
        if _financial_provider is not None:
            return _financial_provider

        settings = get_settings()

        if settings.financial.provider == "mock":
            _financial_provider = MockFinancial()
        elif settings.financial.provider == "yfinance":
            from src.adapters.financial.yfinance_client import YFinanceClient
            _financial_provider = YFinanceClient()
        else:
            raise ValueError(f"Unknown financial provider: {settings.financial.provider}")

        return _financial_provider


def get_embeddings_provider() -> EmbeddingsPort:
    """Get the embeddings provider based on configuration (thread-safe)."""
    global _embeddings_provider

    if _embeddings_provider is not None:
        return _embeddings_provider

    with _lock:
        if _embeddings_provider is not None:
            return _embeddings_provider

        settings = get_settings()

        if settings.embeddings.provider == "mock":
            _embeddings_provider = MockEmbeddings(dimensions=settings.embeddings.dimensions)
        elif settings.embeddings.provider == "openai":
            from src.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
            _embeddings_provider = OpenAIEmbeddings(
                api_key=settings.embeddings.api_key,
                model=settings.embeddings.model,
            )
        else:
            raise ValueError(f"Unknown embeddings provider: {settings.embeddings.provider}")

        return _embeddings_provider


def get_storage() -> StoragePort:
    """Get the storage provider based on configuration (thread-safe)."""
    global _storage

    if _storage is not None:
        return _storage

    with _lock:
        if _storage is not None:
            return _storage

        settings = get_settings()

        if settings.storage.provider == "memory":
            _storage = MemoryRepository()
        elif settings.storage.provider == "postgres":
            from src.adapters.storage.postgres_repository import PostgresRepository
            _storage = PostgresRepository(
                host=settings.database.host,
                port=settings.database.port,
                database=settings.database.name,
                user=settings.database.user,
                password=settings.database.password,
                min_connections=2,
                max_connections=settings.database.pool_size,
            )
        else:
            raise ValueError(f"Unknown storage provider: {settings.storage.provider}")

        return _storage


def get_search_provider() -> Optional[SearchPort]:
    """Get the web search provider (DuckDuckGo, thread-safe)."""
    global _search_provider

    if _search_provider is not None:
        return _search_provider

    with _lock:
        if _search_provider is not None:
            return _search_provider

        try:
            from src.adapters.search.duckduckgo_client import DuckDuckGoClient
            _search_provider = DuckDuckGoClient()
        except ImportError:
            # DuckDuckGo search not available, web research will be disabled
            _search_provider = None

        return _search_provider


def get_agent() -> AIRAAgent:
    """Get the A.I.R.A. agent with all dependencies (thread-safe)."""
    global _agent

    if _agent is not None:
        return _agent

    with _lock:
        if _agent is not None:
            return _agent

        llm = get_llm_provider()
        news = get_news_provider()
        financial = get_financial_provider()
        search = get_search_provider()
        storage = get_storage()
        embeddings = get_embeddings_provider()

        news_tool = NewsRetrieverTool(news_provider=news)
        sentiment_tool = SentimentAnalyzerTool(llm_provider=llm)
        data_tool = DataFetcherTool(financial_provider=financial)

        # Web researcher tool (optional - requires duckduckgo-search)
        web_research_tool = None
        if search is not None:
            web_research_tool = WebResearcherTool(search_provider=search)

        _agent = AIRAAgent(
            llm_provider=llm,
            news_tool=news_tool,
            sentiment_tool=sentiment_tool,
            data_tool=data_tool,
            web_research_tool=web_research_tool,
            storage=storage,
            embeddings_provider=embeddings,
        )

        return _agent


async def get_arq_pool() -> ArqRedis:
    """Get the Arq Redis pool for job enqueueing (async singleton with lock)."""
    global _arq_pool, _arq_pool_lock

    # Create lock lazily (must be created in async context)
    if _arq_pool_lock is None:
        _arq_pool_lock = asyncio.Lock()

    async with _arq_pool_lock:
        if _arq_pool is not None:
            return _arq_pool

        settings = get_settings()
        _arq_pool = await create_pool(
            RedisSettings(
                host=settings.redis.host,
                port=settings.redis.port,
                database=settings.redis.db,
                password=settings.redis.password or None,
            )
        )
        return _arq_pool


async def close_arq_pool() -> None:
    """Close the Arq Redis pool (call on shutdown)."""
    global _arq_pool

    if _arq_pool is not None:
        await _arq_pool.close()
        _arq_pool = None


def reset_dependencies() -> None:
    """Reset all singleton instances."""
    global _llm_provider, _news_provider, _financial_provider
    global _embeddings_provider, _search_provider, _storage, _agent, _arq_pool, _arq_pool_lock

    with _lock:
        _llm_provider = None
        _news_provider = None
        _financial_provider = None
        _embeddings_provider = None
        _search_provider = None
        _storage = None
        _agent = None
        _arq_pool = None
        _arq_pool_lock = None
