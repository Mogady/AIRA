# A.I.R.A. - Autonomous Investment Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.28-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

An AI-powered autonomous agent that monitors the financial world, performs deep research on specific companies, and generates highly structured, data-driven investment analysis reports.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              A.I.R.A. Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   Client    │────▶│              FastAPI REST API                    │   │
│  │  (curl/UI)  │◀────│  /analyze  /status/{id}  /monitor_start  /search │   │
│  └─────────────┘     └─────────────────┬───────────────────────────────┘   │
│                                        │                                    │
│                                        ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Arq Job Queue (Redis)                         │   │
│  │                    Async background job processing                   │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LangGraph Agent (ReAct Pattern)                  │   │
│  │                                                                      │   │
│  │  ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌─────────────────┐  │   │
│  │  │  Parse  │──▶│ Collect  │──▶│  Analyze   │──▶│    Reflect      │  │   │
│  │  │  Query  │   │   Data   │   │ Sentiment  │   │  (Self-Correct) │  │   │
│  │  └─────────┘   └──────────┘   └────────────┘   └────────┬────────┘  │   │
│  │                                                         │           │   │
│  │                 ┌───────────────────────────────────────┘           │   │
│  │                 │                                                   │   │
│  │         ┌──────┴──────┐                                             │   │
│  │         │  Sufficient? │                                            │   │
│  │         └──────┬──────┘                                             │   │
│  │           NO   │   YES                                              │   │
│  │    ┌───────────┴───────────┐                                        │   │
│  │    ▼                       ▼                                        │   │
│  │ ┌──────────┐       ┌─────────────┐                                  │   │
│  │ │ Re-plan  │       │  Synthesize │──▶ AnalysisReport (JSON)         │   │
│  │ │ (Different│       │   Report    │                                  │   │
│  │ │ Strategy) │       └─────────────┘                                  │   │
│  │ └────┬─────┘                                                        │   │
│  │      │                                                              │   │
│  │      └──────▶ Retry with different queries/focus                    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         ▼                          ▼                          ▼            │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐        │
│  │    News     │          │  Sentiment  │          │    Data     │        │
│  │  Retriever  │          │  Analyzer   │          │   Fetcher   │        │
│  │  (NewsAPI)  │          │  (Claude)   │          │  (yfinance) │        │
│  └─────────────┘          └─────────────┘          └─────────────┘        │
│         │                                                   │              │
│         │              ┌─────────────┐                      │              │
│         └─────────────▶│    Web      │◀─────────────────────┘              │
│                        │ Researcher  │                                     │
│                        │ (DuckDuckGo)│                                     │
│                        └─────────────┘                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Layer                                   │   │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐    │   │
│  │  │   PostgreSQL    │   │    pgvector     │   │      Redis      │    │   │
│  │  │  (Job Storage)  │   │ (Embeddings/RAG)│   │  (Queue/Cache)  │    │   │
│  │  └─────────────────┘   └─────────────────┘   └─────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Core Features
- **REST API** with async job processing (10 endpoints)
- **Streamlit Web UI** with real-time progress tracking
- **LangGraph Agent** with ReAct pattern for multi-step reasoning
- **Four Integrated Tools**:
  - 📰 **News Retriever** - Fetches recent news from NewsAPI.org
  - 📊 **Sentiment Analyzer** - AI-powered sentiment analysis via Claude
  - 💹 **Data Fetcher** - Real financial data from Yahoo Finance (yfinance)
  - 🔍 **Web Researcher** - DuckDuckGo search with multiple research focuses
- **Structured JSON Output** with Pydantic validation
- **Docker Deployment** with one-command startup

### Advanced Features
- **Dynamic Reflection Loop** - Self-correction with different strategies when data is insufficient:
  - `USE_WEB_FOR_NEWS` - When news API fails, uses web search with "news" focus
  - `EXPAND_RESEARCH_FOCUS` - Tries different research focuses (analyst_ratings, earnings, competitive, risks)
- **Long-Term Memory** - Vector embeddings with pgvector for semantic search
- **Scheduled Monitoring** - Automatic checks with PROACTIVE_ALERT generation
- **Semantic Search** - Query past analyses using natural language
- **Structured Logging** - Agent's "Thought" process visible in logs

### Reflection Mechanism (Self-Correction)

The agent implements intelligent self-correction when data quality issues are detected:

| Issue Detected | Strategy | Action |
|----------------|----------|--------|
| No news articles | `USE_WEB_FOR_NEWS` | Skip news API, use web search with "news" focus |
| News too old (>30 days) | `USE_WEB_FOR_NEWS` | Supplement with web search for recent coverage |
| No web research results | `EXPAND_RESEARCH_FOCUS` | Try different focus: analyst_ratings → earnings → competitive → risks |
| No financial data | Proceed | Continue with available data |

The agent will retry up to 2 times (configurable) with **different queries each time**, ensuring retries are meaningful rather than redundant.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Claude Sonnet 4.5 (Anthropic) |
| **Embeddings** | OpenAI text-embedding-ada-002 |
| **Agent Framework** | LangGraph |
| **Backend** | FastAPI + Arq |
| **Database** | PostgreSQL 16 + pgvector |
| **Cache/Queue** | Redis 7 |
| **News API** | NewsAPI.org |
| **Financial Data** | Yahoo Finance (yfinance) |
| **Web Search** | DuckDuckGo |
| **Architecture** | Hexagonal (Ports & Adapters) |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- API Keys:
  - Anthropic (Claude) API key
  - OpenAI API key (for embeddings)
  - NewsAPI.org API key

### 1. Clone and Configure

```bash
# Clone repository
git clone <repository-url>
cd AIRA

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 2. Start with Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Access:
# - Web UI at http://localhost:8501
# - API at http://localhost:8000
# - API docs at http://localhost:8000/docs
```

### 3. Start Locally (Development)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Start services (requires PostgreSQL and Redis running)
# Terminal 1: Start API
uvicorn main:app --reload --port 8000

# Terminal 2: Start worker
python -m arq src.adapters.jobs.arq_worker.WorkerSettings

# Terminal 3: Start Streamlit UI
streamlit run streamlit_app.py --server.port 8501
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Start a company analysis |
| `GET` | `/status/{job_id}` | Get analysis status and results |
| `POST` | `/monitor_start` | Start scheduled monitoring |
| `DELETE` | `/monitor/{ticker}` | Stop monitoring a ticker |
| `GET` | `/analyses` | List all analyses (with filters) |
| `GET` | `/monitors` | List active monitors |
| `GET` | `/status/{job_id}/thoughts` | Get agent's reasoning steps |
| `GET` | `/status/{job_id}/tools` | Get tool execution details |
| `POST` | `/search` | Semantic search across analyses |
| `GET` | `/health` | Health check |

## API Usage Examples

### Start an Analysis

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze the near-term prospects of Tesla, Inc. (TSLA)"}'
```

**Response:**
```json
{
  "job_id": "abc123-def456",
  "status": "PENDING",
  "message": "Analysis job created successfully",
  "status_url": "/status/abc123-def456"
}
```

### Check Status

```bash
curl "http://localhost:8000/status/abc123-def456"
```

**Response (when complete):**
```json
{
  "job_id": "abc123-def456",
  "status": "COMPLETED",
  "result": {
    "company_ticker": "TSLA",
    "company_name": "Tesla, Inc.",
    "analysis_summary": "Tesla continues to demonstrate strong market position...",
    "sentiment_score": 0.45,
    "key_findings": [
      "Strong Q3 delivery numbers exceeded analyst expectations",
      "Expansion of Gigafactory production capacity on track",
      "Increased competition in EV market requires continued innovation"
    ],
    "tools_used": ["news_retriever", "sentiment_analyzer", "data_fetcher", "web_researcher"],
    "citation_sources": ["https://..."],
    "financial_snapshot": {
      "current_price": 248.50,
      "market_cap": 789000000000,
      "pe_ratio": 62.5,
      "analyst_target_mean": 210.00
    },
    "reflection_triggered": false,
    "reflection_notes": null,
    "generated_at": "2024-12-23T10:30:00Z"
  }
}
```

### Start Monitoring

```bash
curl -X POST "http://localhost:8000/monitor_start" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "interval_hours": 24}'
```

### Semantic Search

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "companies with positive sentiment about AI", "limit": 5}'
```

### List Analyses

```bash
# List all analyses
curl "http://localhost:8000/analyses"

# Filter by ticker
curl "http://localhost:8000/analyses?ticker=TSLA"

# Filter by status
curl "http://localhost:8000/analyses?status=COMPLETED&limit=10"
```

### View Agent Thoughts

```bash
curl "http://localhost:8000/status/abc123-def456/thoughts"
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

## Web UI

A.I.R.A. includes a full-featured Streamlit web interface:

### Features
- **New Analysis** - Submit queries with real-time progress tracking
- **Dashboard** - Overview of all analyses and monitoring status
- **Monitoring** - Configure automated stock monitoring
- **History** - Browse and filter past analyses

### Access
- **Local**: http://localhost:8501
- **Docker**: http://localhost:8501

## Configuration

All settings are configurable via `config.yaml`:

```yaml
llm:
  provider: "anthropic"  # or "mock" for testing
  model: "claude-sonnet-4-5-20250514"
  max_tokens: 4096

news:
  provider: "newsapi"  # or "mock" for testing
  articles_per_request: 10
  max_age_days: 30

financial:
  provider: "yfinance"  # or "mock" for testing

search:
  provider: "duckduckgo"  # or "mock" for testing

agent:
  max_iterations: 10
  max_reflection_cycles: 2  # How many times to retry with different strategies
  reflection_enabled: true

monitoring:
  enabled: true
  check_interval_hours: 24
  min_new_articles: 5  # Trigger analysis if >= 5 new articles
```

## Project Structure

```
AIRA/
├── src/
│   ├── domain/           # Core business models & exceptions
│   │   ├── models.py     # Pydantic models (30+ data models)
│   │   └── exceptions.py # Custom exception hierarchy
│   ├── application/      # Ports (interfaces) & services
│   │   ├── ports/        # Abstract interfaces for all providers
│   │   └── services/     # Embedding service
│   ├── adapters/         # Infrastructure implementations
│   │   ├── api/          # FastAPI routes & dependencies
│   │   ├── agents/       # LangGraph agent (1300+ lines)
│   │   ├── llm/          # Claude & mock providers
│   │   ├── news/         # NewsAPI & mock providers
│   │   ├── financial/    # yfinance & mock providers
│   │   ├── search/       # DuckDuckGo & mock providers
│   │   ├── embeddings/   # OpenAI & mock embeddings
│   │   ├── storage/      # PostgreSQL & memory repos
│   │   └── jobs/         # Arq worker & scheduler
│   ├── tools/            # Agent tools (4 tools)
│   │   ├── news_retriever.py
│   │   ├── sentiment_analyzer.py
│   │   ├── data_fetcher.py
│   │   └── web_researcher.py
│   └── config/           # Settings & logging
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests (including reflection tests)
│   └── api/              # API endpoint tests
├── main.py               # FastAPI entry point
├── worker.py             # Arq worker entry point
├── streamlit_app.py      # Streamlit UI application
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # 5-service orchestration
├── config.yaml           # Application configuration
└── pyproject.toml        # Dependencies (uv/pip)
```

## Testing

```bash
# Run all tests
.venv/bin/python -m pytest

# Run with coverage
.venv/bin/python -m pytest --cov=src --cov-report=html

# Run specific test categories
.venv/bin/python -m pytest tests/unit/              # Unit tests
.venv/bin/python -m pytest tests/integration/       # Integration tests
.venv/bin/python -m pytest tests/api/               # API tests

# Run reflection mechanism tests
.venv/bin/python -m pytest tests/integration/test_reflection.py -v
```

### Test Coverage

- **Unit tests**: Models, tools, services
- **Integration tests**: Full agent flow, reflection mechanism, providers
- **API tests**: All endpoints, error handling

## Design Decisions

### Why LangGraph?
- Native support for reflection loops (cyclic graphs)
- Explicit state machine makes agent reasoning transparent
- Production-ready with checkpointing support
- Better debugging compared to LangChain agents

### Why Hexagonal Architecture?
- Easy to swap implementations (mock ↔ real)
- Clean separation enables comprehensive testing
- Demonstrates senior-level software engineering
- Future-proof for new providers

### Why Arq over Celery?
- Async-native (perfect for FastAPI)
- Simpler setup (no RabbitMQ needed)
- Sufficient for the use case
- Lower operational overhead

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | Yes (for real LLM) |
| `OPENAI_API_KEY` | OpenAI API key | Yes (for embeddings) |
| `NEWS_API_KEY` | NewsAPI.org key | Yes (for real news) |
| `POSTGRES_HOST` | PostgreSQL host | Yes |
| `POSTGRES_PASSWORD` | Database password | Yes |
| `REDIS_HOST` | Redis host | Yes |
| `REDIS_PASSWORD` | Redis password | No |

## Output Schema

The analysis report includes:

```json
{
  "company_ticker": "TSLA",
  "company_name": "Tesla, Inc.",
  "analysis_summary": "Executive summary with specific numbers...",
  "sentiment_score": 0.45,
  "key_findings": ["Finding 1", "Finding 2", "..."],
  "tools_used": ["news_retriever", "sentiment_analyzer", "data_fetcher", "web_researcher"],
  "citation_sources": ["https://..."],
  "news_summary": "Summary of news coverage...",
  "financial_snapshot": {
    "current_price": 248.50,
    "market_cap": 789000000000,
    "pe_ratio": 62.5,
    "analyst_target_mean": 210.00,
    "...": "50+ financial metrics"
  },
  "investment_thesis": "Bull case, bear case, balanced view...",
  "risk_factors": ["Risk 1", "Risk 2"],
  "competitive_context": "Market positioning...",
  "analyst_consensus": {
    "target_mean": 210.00,
    "recommendation": "hold",
    "number_of_analysts": 45
  },
  "catalyst_events": ["Upcoming catalyst 1", "..."],
  "web_research": {
    "results": [...],
    "queries_used": [...]
  },
  "reflection_triggered": true,
  "reflection_notes": "No news articles retrieved",
  "analysis_type": "ON_DEMAND",
  "generated_at": "2024-12-24T10:30:00Z",
  "iteration_count": 2
}
