"""
Pytest configuration and shared fixtures.
"""

import pytest
from fastapi.testclient import TestClient

from src.adapters.api.dependencies import reset_dependencies
from src.adapters.embeddings.mock_embeddings import MockEmbeddings
from src.adapters.financial.mock_financial import MockFinancial
from src.adapters.llm.mock_llm import MockLLM
from src.adapters.news.mock_news import MockNews
from src.adapters.search.mock_search import MockSearch
from src.adapters.storage.memory_repository import MemoryRepository
from src.tools.data_fetcher import DataFetcherTool
from src.tools.news_retriever import NewsRetrieverTool
from src.tools.sentiment_analyzer import SentimentAnalyzerTool
from src.tools.web_researcher import WebResearcherTool


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset dependency singletons before each test."""
    reset_dependencies()
    yield
    reset_dependencies()


@pytest.fixture
def mock_llm():
    """Provide a mock LLM provider."""
    return MockLLM()


@pytest.fixture
def mock_news():
    """Provide a mock news provider."""
    return MockNews()


@pytest.fixture
def mock_financial():
    """Provide a mock financial data provider."""
    return MockFinancial()


@pytest.fixture
def mock_embeddings():
    """Provide a mock embeddings provider."""
    return MockEmbeddings()


@pytest.fixture
def memory_storage():
    """Provide an in-memory storage."""
    return MemoryRepository()


@pytest.fixture
def news_tool(mock_news):
    """Provide a news retriever tool with mock provider."""
    return NewsRetrieverTool(news_provider=mock_news)


@pytest.fixture
def sentiment_tool(mock_llm):
    """Provide a sentiment analyzer tool with mock LLM."""
    return SentimentAnalyzerTool(llm_provider=mock_llm)


@pytest.fixture
def data_tool(mock_financial):
    """Provide a data fetcher tool with mock provider."""
    return DataFetcherTool(financial_provider=mock_financial)


@pytest.fixture
def mock_search():
    """Provide a mock search provider."""
    return MockSearch()


@pytest.fixture
def web_research_tool(mock_search):
    """Provide a web researcher tool with mock provider."""
    return WebResearcherTool(search_provider=mock_search)


@pytest.fixture
def test_client():
    """Provide a FastAPI test client."""
    from main import app
    return TestClient(app)


@pytest.fixture
def embedding_service(mock_embeddings, memory_storage):
    """Provide an embedding service with mock dependencies."""
    from src.application.services.embedding_service import EmbeddingService
    return EmbeddingService(mock_embeddings, memory_storage)
