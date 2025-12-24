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
│  │  (curl/UI)  │◀────│  /analyze  /status/{id}  /monitor_start  /health │   │
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
│  │  ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌─────────────────┐  │   │
│  │  │  Parse  │──▶│   Plan   │──▶│  Execute   │──▶│    Reflect      │  │   │
│  │  │  Query  │   │          │   │   Tools    │   │  (Self-Correct) │  │   │
│  │  └─────────┘   └──────────┘   └────────────┘   └────────┬────────┘  │   │
│  │                                                         │           │   │
│  │                      ┌───────────────────────────────────┘           │   │
│  │                      │ Sufficient?                                   │   │
│  │                      ▼ YES                                           │   │
│  │              ┌─────────────┐                                         │   │
│  │              │  Synthesize │──▶ AnalysisReport (JSON)                │   │
│  │              │   Report    │                                         │   │
│  │              └─────────────┘                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         ▼                          ▼                          ▼            │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐        │
│  │    News     │          │  Sentiment  │          │    Data     │        │
│  │  Retriever  │          │  Analyzer   │          │   Fetcher   │        │
│  │  (NewsAPI)  │          │  (Claude)   │          │   (Mock)    │        │
│  └─────────────┘          └─────────────┘          └─────────────┘        │
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
- **REST API** with async job processing (`/analyze`, `/status/{job_id}`)
- **Streamlit Web UI** with real-time progress tracking
- **LangGraph Agent** with ReAct pattern for multi-step reasoning
- **Three Integrated Tools**:
  - 📰 **News Retriever** - Fetches recent news from NewsAPI.org
  - 📊 **Sentiment Analyzer** - AI-powered sentiment analysis via Claude
  - 💹 **Data Fetcher** - Company financial data retrieval
- **Structured JSON Output** with Pydantic validation
- **Docker Deployment** with one-command startup

### Advanced Features
- **Reflection Loop** - Self-correction when data is insufficient
- **Long-Term Memory** - Vector embeddings with pgvector for semantic search
- **Scheduled Monitoring** - Automatic checks with PROACTIVE_ALERT generation
- **Structured Logging** - Agent's "Thought" process visible in logs

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

# Install dependencies
pip install -e .

# Start services (requires PostgreSQL and Redis running)
# Terminal 1: Start API
uvicorn main:app --reload --port 8000

# Terminal 2: Start worker
python -m arq src.adapters.jobs.arq_worker.WorkerSettings

# Terminal 3: Start Streamlit UI
streamlit run streamlit_app.py --server.port 8501
```

## Web UI

A.I.R.A. includes a full-featured Streamlit web interface:

### Features
- **New Analysis** - Submit queries with real-time progress tracking
- **Dashboard** - Overview of all analyses and monitoring status
- **Monitoring** - Configure automated stock monitoring
- **History** - Browse and filter past analyses

### Screenshots

```
+----------------------------------+----------------------------------------+
|         SIDEBAR                  |              MAIN CONTENT              |
|                                  |                                        |
|  A.I.R.A.                        |  NEW ANALYSIS                          |
|                                  |                                        |
|  > New Analysis                  |  [Enter query: Analyze Tesla (TSLA)]   |
|    Dashboard                     |  [Analyze Button]                      |
|    Monitoring                    |                                        |
|    History                       |  Progress:                             |
|                                  |  [====>     ] Fetching news...         |
|  Backend Connected               |                                        |
+----------------------------------+----------------------------------------+
```

### Access
- **Local**: http://localhost:8501
- **Docker**: http://localhost:8501

## API Usage

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
    "tools_used": ["news_retriever", "sentiment_analyzer", "data_fetcher"],
    "citation_sources": ["https://..."],
    "reflection_triggered": false,
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

### Health Check

```bash
curl "http://localhost:8000/health"
```

## Configuration

All settings are configurable via `config.yaml`:

```yaml
llm:
  provider: "anthropic"  # or "mock" for testing
  model: "claude-sonnet-4-5-20250514"
  max_tokens: 4096

news:
  provider: "newsapi"  # or "mock" for testing
  articles_per_request: 5
  max_age_days: 30

agent:
  max_iterations: 10
  max_reflection_cycles: 2
  reflection_enabled: true
```

## Project Structure

```
aira/
├── src/
│   ├── domain/           # Core business models & exceptions
│   ├── application/      # Ports (interfaces) & use cases
│   ├── adapters/         # Infrastructure implementations
│   │   ├── api/          # FastAPI routes
│   │   ├── agents/       # LangGraph agent
│   │   ├── llm/          # Claude & mock providers
│   │   ├── news/         # NewsAPI & mock providers
│   │   ├── financial/    # Financial data providers
│   │   ├── embeddings/   # OpenAI & mock embeddings
│   │   ├── storage/      # PostgreSQL & memory repos
│   │   └── jobs/         # Arq worker & scheduler
│   ├── tools/            # Agent tools
│   └── config/           # Settings & logging
├── tests/                # Test suite
├── main.py               # FastAPI entry point
├── worker.py             # Arq worker entry point
├── streamlit_app.py      # Streamlit UI application
├── Dockerfile
├── docker-compose.yml
└── config.yaml
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py
```

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
| `POSTGRES_PASSWORD` | Database password | Yes |
| `REDIS_PASSWORD` | Redis password | No |

## License

This project was created for the InVitro Capital case study assessment.

---

**Built with Claude Sonnet 4.5, LangGraph, and FastAPI**
