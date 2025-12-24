"""
LangGraph Agent - Core agent implementation with ReAct pattern and reflection.

This module implements the A.I.R.A. agent using LangGraph for:
- Multi-step planning and execution
- Tool orchestration (news, sentiment, financial data, web research)
- Reflection loop for self-correction
- LLM-powered professional synthesis
- Structured logging of agent thoughts
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.application.ports.embeddings_port import EmbeddingsPort
from src.application.ports.llm_port import LLMPort, LLMMessage
from src.application.ports.storage_port import StoragePort
from src.config.logging import get_agent_logger, get_logger
from src.config.settings import get_settings
from src.domain.models import (
    AnalysisReport,
    NewsRetrieverOutput,
    SentimentResult,
    FinancialData,
    WebResearchOutput,
)
from src.tools.news_retriever import NewsRetrieverTool
from src.tools.sentiment_analyzer import SentimentAnalyzerTool
from src.tools.data_fetcher import DataFetcherTool
from src.tools.web_researcher import WebResearcherTool

logger = get_logger(__name__)


# =============================================================================
# Professional Investment Analyst Prompts
# =============================================================================

INVESTMENT_ANALYST_SYSTEM_PROMPT = """You are a senior equity research analyst at a top-tier investment bank with 15+ years of experience.

Your analysis is known for being:
- Data-driven and precise with specific numbers and percentages
- Balanced, acknowledging both opportunities and risks
- Forward-looking with clear investment implications
- Professional but accessible to sophisticated investors

When analyzing a company, you always:
1. Lead with the most important insight that moves the investment thesis
2. Connect financial metrics to business fundamentals and competitive dynamics
3. Compare current valuation to analyst consensus when available
4. Identify specific catalysts and quantifiable risk factors
5. Provide actionable investment perspective with price context

You never use vague language like "further monitoring recommended" - you always provide specific, actionable insights."""

SYNTHESIS_PROMPT_TEMPLATE = """Analyze {company} ({ticker}) using the following comprehensive data:

## FINANCIAL METRICS
{financial_data}

## RECENT NEWS ({news_count} articles)
{news_headlines}

## MARKET SENTIMENT
Sentiment Score: {sentiment_score:.2f} ({sentiment_label})
Distribution: {positive_count} positive, {negative_count} negative, {neutral_count} neutral
{sentiment_themes}

## WEB RESEARCH FINDINGS
{web_research}

## HISTORICAL CONTEXT
{historical_context}

---

Provide a professional investment analysis in JSON format with these exact fields:

{{
    "analysis_summary": "2-3 sentence executive summary leading with the most important insight. Include specific numbers. If historical context shows a trend, mention it.",
    "key_findings": ["Finding 1 with specific data", "Finding 2 with numbers", "Finding 3", "Finding 4", "Finding 5"],
    "financial_analysis": "Detailed paragraph on valuation (P/E, price vs targets), profitability (margins), balance sheet health, and revenue trajectory. Use actual numbers.",
    "investment_thesis": "Bull case, bear case, and your balanced view. Reference analyst targets if available. Consider historical sentiment trend if available.",
    "risk_factors": ["Specific risk 1", "Specific risk 2", "Specific risk 3"],
    "competitive_context": "Brief competitive positioning if data available, otherwise null",
    "catalyst_events": ["Upcoming catalyst 1", "Catalyst 2"]
}}

IMPORTANT:
- Use actual numbers from the data provided
- Be specific - avoid generic statements
- If analyst price targets are available, compare current price to targets
- Reference specific metrics like P/E, margins, growth rates
- Each key finding should contain at least one number or specific fact
- If historical context shows sentiment improving or declining, factor this into your analysis
- Return ONLY valid JSON, no other text"""


class ReflectionReason(Enum):
    """Structured reasons for triggering reflection."""
    NEWS_TOO_OLD = "news_too_old"
    NO_NEWS = "no_news"
    NO_SENTIMENT = "no_sentiment"
    NO_FINANCIAL_DATA = "no_financial_data"
    INSUFFICIENT_RESEARCH = "insufficient_research"


class ReflectionStrategy(Enum):
    """Strategies for handling reflection retries."""
    USE_WEB_FOR_NEWS = "use_web_for_news"  # When news API fails, use web search for news
    EXPAND_RESEARCH_FOCUS = "expand_research_focus"  # Try different research focus areas
    NONE = "none"  # No special strategy (proceed with available data)


class GraphState(TypedDict):
    """State for the LangGraph state machine."""

    # Job identification
    job_id: str
    query: str
    ticker: str
    company_name: str

    # Tool results
    news_result: Optional[NewsRetrieverOutput]
    sentiment_result: Optional[SentimentResult]
    financial_data: Optional[FinancialData]
    web_research: Optional[WebResearchOutput]

    # Historical context from embeddings
    historical_context: Optional[Dict[str, Any]]

    # Execution tracking
    iteration: int
    max_iterations: int
    reflection_count: int
    max_reflection_cycles: int
    reflection_triggered: bool
    reflection_reason: Optional[str]
    reflection_reasons: List[ReflectionReason]

    # Reflection strategy for retries
    reflection_strategy: Optional[str]  # ReflectionStrategy value
    attempted_research_focuses: List[str]  # Track which focuses we've tried

    # Planning
    current_plan: List[str]
    completed_steps: List[str]
    tools_used: List[str]
    thoughts: List[str]

    # Output
    final_report: Optional[Dict[str, Any]]
    error: Optional[str]
    status: str


class AIRAAgent:
    """
    Autonomous Investment Research Agent implemented with LangGraph.

    Features:
    - Parallel data collection for efficiency
    - Web research for broader context
    - LLM-powered professional synthesis
    - Reflection loop for self-correction
    - Structured logging of agent thoughts
    """

    def __init__(
        self,
        llm_provider: LLMPort,
        news_tool: NewsRetrieverTool,
        sentiment_tool: SentimentAnalyzerTool,
        data_tool: DataFetcherTool,
        web_research_tool: Optional[WebResearcherTool] = None,
        storage: Optional[StoragePort] = None,
        embeddings_provider: Optional[EmbeddingsPort] = None,
    ):
        """
        Initialize the A.I.R.A. agent.

        Args:
            llm_provider: LLM for reasoning and synthesis
            news_tool: Tool for fetching news
            sentiment_tool: Tool for sentiment analysis
            data_tool: Tool for financial data
            web_research_tool: Tool for web research (optional)
            storage: Optional storage for logging
            embeddings_provider: Optional embeddings provider for historical context
        """
        self._llm = llm_provider
        self._news_tool = news_tool
        self._sentiment_tool = sentiment_tool
        self._data_tool = data_tool
        self._web_research_tool = web_research_tool
        self._storage = storage
        self._embeddings = embeddings_provider

        self._settings = get_settings()
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine with parallel data collection."""

        graph = StateGraph(GraphState)

        graph.add_node("parse_query", self._parse_query_node)
        graph.add_node("collect_data", self._collect_data_node)
        graph.add_node("analyze_sentiment", self._analyze_sentiment_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("replan", self._replan_node)
        graph.add_node("retrieve_history", self._retrieve_historical_context_node)
        graph.add_node("synthesize", self._synthesize_node)

        # Set entry point
        graph.set_entry_point("parse_query")

        # flow: parse → collect (parallel) → sentiment → reflect → retrieve_history → synthesize
        graph.add_edge("parse_query", "collect_data")
        graph.add_edge("collect_data", "analyze_sentiment")
        graph.add_edge("analyze_sentiment", "reflect")

        graph.add_conditional_edges(
            "reflect",
            self._should_continue_or_replan,
            {
                "replan": "replan",
                "retrieve_history": "retrieve_history",
            }
        )

        # Replan leads back to appropriate node
        graph.add_conditional_edges(
            "replan",
            self._route_after_replan,
            {
                "collect_data": "collect_data",
                "analyze_sentiment": "analyze_sentiment",
                "retrieve_history": "retrieve_history",
            }
        )

        # After retrieving historical context, proceed to synthesis
        graph.add_edge("retrieve_history", "synthesize")

        graph.add_edge("synthesize", END)

        return graph.compile()

    async def analyze(
        self,
        job_id: str,
        query: str,
    ) -> AnalysisReport:
        """
        Run the full analysis for a company.

        Args:
            job_id: Unique job identifier
            query: User's analysis query

        Returns:
            AnalysisReport with complete analysis
        """
        start_time = time.time()
        agent_logger = get_agent_logger(job_id)

        agent_logger.thought(
            thought_type="initialization",
            content=f"Starting analysis for query: {query}",
        )

        # Initialize state
        initial_state: GraphState = {
            "job_id": job_id,
            "query": query,
            "ticker": "",
            "company_name": "",
            "news_result": None,
            "sentiment_result": None,
            "financial_data": None,
            "web_research": None,
            "historical_context": None,
            "iteration": 0,
            "max_iterations": self._settings.agent.max_iterations,
            "reflection_count": 0,
            "max_reflection_cycles": self._settings.agent.max_reflection_cycles,
            "reflection_triggered": False,
            "reflection_reason": None,
            "reflection_reasons": [],
            "reflection_strategy": None,
            "attempted_research_focuses": [],
            "current_plan": [],
            "completed_steps": [],
            "tools_used": [],
            "thoughts": [],
            "final_report": None,
            "error": None,
            "status": "RUNNING",
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)

            if final_state.get("final_report"):
                report = AnalysisReport(**final_state["final_report"])
            else:
                raise ValueError("No report generated")

            duration_ms = int((time.time() - start_time) * 1000)

            agent_logger.analysis_complete(
                ticker=report.company_ticker,
                duration_ms=duration_ms,
                tools_used=report.tools_used,
                reflection_triggered=report.reflection_triggered,
            )

            # Store embeddings for future retrieval
            await self._store_embeddings(job_id, report, agent_logger)

            return report

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            agent_logger.analysis_failed(
                error=str(e),
                duration_ms=duration_ms,
            )
            raise

    async def _store_embeddings(
        self,
        job_id: str,
        report: AnalysisReport,
        agent_logger,
    ) -> None:
        """
        Store embeddings for the completed analysis.

        This enables future semantic search and historical context retrieval.
        """
        if not self._embeddings or not self._storage:
            logger.debug("Skipping embedding storage (no embeddings provider)")
            return

        try:
            from src.application.services.embedding_service import EmbeddingService

            service = EmbeddingService(self._embeddings, self._storage)
            embedding_ids = await service.embed_analysis(job_id, report)

            agent_logger.thought(
                thought_type="memory",
                content=f"Stored {len(embedding_ids)} embeddings for future retrieval",
            )

            logger.info(
                "embeddings_stored",
                job_id=job_id,
                ticker=report.company_ticker,
                num_embeddings=len(embedding_ids),
            )

        except Exception as e:
            # Don't fail the analysis if embedding storage fails
            logger.warning(
                "embedding_storage_failed",
                job_id=job_id,
                error=str(e),
            )

    # =========================================================================
    # Graph Nodes
    # =========================================================================

    async def _parse_query_node(self, state: GraphState) -> GraphState:
        """Parse the user query to extract ticker and company name."""
        agent_logger = get_agent_logger(state["job_id"])

        query = state["query"]

        # Extract ticker from parentheses
        ticker_match = re.search(r'\(([A-Z]{1,5})\)', query)
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"

        # Extract company name
        company_match = re.search(r'(\b[A-Za-z][A-Za-z]*(?:\s+[A-Za-z]+)*)\s*\([A-Z]{1,5}\)', query)
        if company_match:
            company_name = company_match.group(1).strip()
            company_name = re.sub(r'^[Aa]nalyze\s+', '', company_name)
        else:
            company_name = ticker

        agent_logger.thought(
            thought_type="parsing",
            content=f"Extracted ticker: {ticker}, company: {company_name}",
            ticker=ticker,
            company_name=company_name,
        )

        state["ticker"] = ticker
        state["company_name"] = company_name
        state["iteration"] = state["iteration"] + 1

        return state

    async def _collect_data_node(self, state: GraphState) -> GraphState:
        """
        Collect all data in parallel for efficiency.

        Runs news retrieval, financial data fetch, and web research concurrently.
        On reflection retries, uses alternative strategies based on reflection_strategy.
        """
        agent_logger = get_agent_logger(state["job_id"])

        # Check if we're in a reflection retry with a specific strategy
        reflection_strategy = state.get("reflection_strategy")
        is_retry = reflection_strategy is not None

        if is_retry:
            agent_logger.thought(
                thought_type="planning",
                content=f"Retry data collection with strategy: {reflection_strategy}",
            )
        else:
            agent_logger.thought(
                thought_type="planning",
                content=f"Starting parallel data collection for {state['company_name']} ({state['ticker']})",
            )

        # Determine what to fetch based on strategy
        should_fetch_news = not is_retry or reflection_strategy != ReflectionStrategy.USE_WEB_FOR_NEWS.value
        should_fetch_financial = not is_retry  # Only fetch financial on first pass

        # Determine research focus for web research
        if reflection_strategy == ReflectionStrategy.USE_WEB_FOR_NEWS.value:
            # Use web search to find news with a news-specific focus
            research_focus = "news"
            should_do_web_research = True
        elif reflection_strategy == ReflectionStrategy.EXPAND_RESEARCH_FOCUS.value:
            # Try a different research focus
            attempted = state.get("attempted_research_focuses", [])
            available_focuses = ["analyst_ratings", "earnings", "competitive", "risks"]
            research_focus = next((f for f in available_focuses if f not in attempted), "general")
            should_do_web_research = True
        else:
            research_focus = "general"
            should_do_web_research = not is_retry  # Only on first pass or specific strategies

        # Define async tasks for parallel execution
        async def fetch_news():
            if not should_fetch_news:
                return ("news", None)
            try:
                result = await self._news_tool.execute(
                    company=state["company_name"],
                    ticker=state["ticker"],
                    num_articles=self._settings.news.articles_per_request,
                )
                return ("news", result)
            except Exception as e:
                logger.warning(f"News fetch failed: {e}")
                return ("news", None)

        async def fetch_financial():
            if not should_fetch_financial:
                return ("financial", None)
            try:
                result = await self._data_tool.execute(ticker=state["ticker"])
                return ("financial", result)
            except Exception as e:
                logger.warning(f"Financial fetch failed: {e}")
                return ("financial", None)

        async def fetch_web_research():
            if not should_do_web_research or not self._web_research_tool:
                return ("web_research", None)
            try:
                result = await self._web_research_tool.execute(
                    company=state["company_name"],
                    ticker=state["ticker"],
                    research_focus=research_focus,
                )
                return ("web_research", result)
            except Exception as e:
                logger.warning(f"Web research failed: {e}")
                return ("web_research", None)

        # Run all data collection in parallel
        start_time = time.time()
        results = await asyncio.gather(
            fetch_news(),
            fetch_financial(),
            fetch_web_research(),
            return_exceptions=True
        )
        duration_ms = int((time.time() - start_time) * 1000)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Data collection error: {result}")
                continue

            result_type, data = result

            if result_type == "news" and data:
                state["news_result"] = data
                if "news_retriever" not in state["tools_used"]:
                    state["tools_used"].append("news_retriever")
                state["thoughts"].append(
                    f"Retrieved {len(data.articles)} news articles about {state['ticker']}."
                )

            elif result_type == "financial" and data:
                state["financial_data"] = data
                if "data_fetcher" not in state["tools_used"]:
                    state["tools_used"].append("data_fetcher")
                analyst_info = ""
                if data.analyst_target_mean:
                    analyst_info = f", analyst target: ${data.analyst_target_mean:.2f}"
                state["thoughts"].append(
                    f"Financial data retrieved: price=${data.current_price}{analyst_info}"
                )

            elif result_type == "web_research" and data:
                # On retry, merge new web research with existing
                if state.get("web_research") and is_retry:
                    # Combine results, avoiding duplicates
                    existing_urls = {r.url for r in state["web_research"].results}
                    new_results = [r for r in data.results if r.url not in existing_urls]
                    state["web_research"].results.extend(new_results)
                    state["web_research"].queries_used.extend(data.queries_used)
                    state["web_research"].total_results += len(new_results)
                    state["thoughts"].append(
                        f"Additional web research ({research_focus}): {len(new_results)} new results."
                    )
                else:
                    state["web_research"] = data
                    state["thoughts"].append(
                        f"Web research ({research_focus}): {len(data.results)} results from {len(data.queries_used)} queries."
                    )

                if "web_researcher" not in state["tools_used"]:
                    state["tools_used"].append("web_researcher")

                # Track the research focus we've attempted
                if research_focus not in state.get("attempted_research_focuses", []):
                    state["attempted_research_focuses"].append(research_focus)

        agent_logger.tool_complete(
            tool_name="parallel_data_collection" if not is_retry else f"retry_data_collection_{reflection_strategy}",
            duration_ms=duration_ms,
            result_summary=f"News: {len(state['news_result'].articles) if state.get('news_result') else 0}, "
                          f"Financial: {'Yes' if state.get('financial_data') else 'No'}, "
                          f"Web: {len(state['web_research'].results) if state.get('web_research') else 0}",
        )

        # Clear the reflection strategy after processing (so next iteration starts fresh)
        state["reflection_strategy"] = None
        state["completed_steps"].append("collect_data")
        return state

    async def _analyze_sentiment_node(self, state: GraphState) -> GraphState:
        """Analyze sentiment from news articles."""
        agent_logger = get_agent_logger(state["job_id"])

        if not state.get("news_result") or not state["news_result"].articles:
            agent_logger.thought(
                thought_type="skip",
                content="Skipping sentiment analysis - no articles available",
            )
            return state

        agent_logger.tool_start(
            tool_name="sentiment_analyzer",
            input_params={
                "ticker": state["ticker"],
                "article_count": len(state["news_result"].articles),
            },
        )

        start_time = time.time()

        try:
            result = await self._sentiment_tool.execute(
                articles=state["news_result"].articles,
                ticker=state["ticker"],
                company_name=state["company_name"],
            )

            duration_ms = int((time.time() - start_time) * 1000)

            agent_logger.tool_complete(
                tool_name="sentiment_analyzer",
                duration_ms=duration_ms,
                result_summary=f"Sentiment: {result.overall_sentiment} (score: {result.sentiment_score:.2f})",
            )

            state["sentiment_result"] = result
            state["tools_used"].append("sentiment_analyzer")
            state["completed_steps"].append("analyze_sentiment")
            state["thoughts"].append(
                f"Sentiment analysis: {result.overall_sentiment} "
                f"(score: {result.sentiment_score:.2f}, "
                f"+{result.positive_count}/-{result.negative_count}/~{result.neutral_count})"
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            agent_logger.tool_error(
                tool_name="sentiment_analyzer",
                error=str(e),
                duration_ms=duration_ms,
            )
            state["error"] = f"Sentiment analysis failed: {e}"

        return state

    async def _reflect_node(self, state: GraphState) -> GraphState:
        """Reflect on gathered data and decide if more research is needed."""
        agent_logger = get_agent_logger(state["job_id"])

        issues: List[tuple[ReflectionReason, str]] = []

        # Check news recency
        if state.get("news_result") and state["news_result"].articles:
            max_age = timedelta(days=self._settings.news.max_age_days)
            now = datetime.now(timezone.utc)
            old_articles = [
                a for a in state["news_result"].articles
                if (now - a.published_at.replace(tzinfo=timezone.utc) if a.published_at.tzinfo is None else now - a.published_at) > max_age
            ]
            if len(old_articles) == len(state["news_result"].articles):
                issues.append((ReflectionReason.NEWS_TOO_OLD, "All news articles are older than 30 days"))
        elif not state.get("news_result") or not state["news_result"].articles:
            issues.append((ReflectionReason.NO_NEWS, "No news articles retrieved"))

        # Check sentiment - only flag if analysis failed, not if result is neutral
        # Neutral sentiment is a valid result, not an error condition
        if not state.get("sentiment_result"):
            if state.get("news_result") and state["news_result"].articles:
                issues.append((ReflectionReason.NO_SENTIMENT, "Sentiment analysis not completed"))

        # Check financial data
        if not state.get("financial_data"):
            issues.append((ReflectionReason.NO_FINANCIAL_DATA, "Financial data not retrieved"))

        # Check web research (if web_research tool was used but returned no results)
        if state.get("web_research") is not None and len(state["web_research"].results) == 0:
            issues.append((ReflectionReason.INSUFFICIENT_RESEARCH, "Web research returned no results"))

        # Decide if reflection should trigger re-planning
        reflection_count = state.get("reflection_count", 0)
        max_reflections = state.get("max_reflection_cycles", 2)

        if issues and reflection_count < max_reflections and self._settings.agent.reflection_enabled:
            state["reflection_triggered"] = True
            state["reflection_reasons"] = [reason for reason, _ in issues]
            state["reflection_reason"] = "; ".join(desc for _, desc in issues)
            state["reflection_count"] = reflection_count + 1

            agent_logger.reflection(
                triggered=True,
                reason=state["reflection_reason"],
                action="Will re-plan to gather more data",
                cycle=state["reflection_count"],
            )

            state["thoughts"].append(
                f"Reflection triggered (cycle {state['reflection_count']}): {state['reflection_reason']}"
            )
        else:
            # Important: set reflection_triggered to False so we proceed to synthesis
            state["reflection_triggered"] = False
            reason = "Data quality sufficient for synthesis" if not issues else f"Max reflections reached: {'; '.join(desc for _, desc in issues)}"
            agent_logger.reflection(
                triggered=False,
                reason=reason,
                cycle=state.get("reflection_count", 0),
            )
            state["thoughts"].append(f"Reflection: {reason}")

        return state

    async def _replan_node(self, state: GraphState) -> GraphState:
        """Re-plan after reflection identifies issues with a concrete strategy."""
        agent_logger = get_agent_logger(state["job_id"])

        reasons = state.get("reflection_reasons", [])

        # Determine strategy and next step based on issues
        strategy = ReflectionStrategy.NONE
        next_step = "retrieve_history"
        strategy_description = ""

        if ReflectionReason.NO_NEWS in reasons or ReflectionReason.NEWS_TOO_OLD in reasons:
            # News API failed or returned stale data - use web search to find news instead
            strategy = ReflectionStrategy.USE_WEB_FOR_NEWS
            next_step = "collect_data"
            strategy_description = "Will use web search to find recent news coverage"
            state["thoughts"].append(f"Re-planning: {strategy_description}")

        elif ReflectionReason.INSUFFICIENT_RESEARCH in reasons:
            # Web research returned no results - try a different research focus
            attempted = state.get("attempted_research_focuses", [])
            available_focuses = ["analyst_ratings", "earnings", "competitive", "risks"]
            # Find a focus we haven't tried yet
            for focus in available_focuses:
                if focus not in attempted:
                    strategy = ReflectionStrategy.EXPAND_RESEARCH_FOCUS
                    next_step = "collect_data"
                    strategy_description = f"Will try web research with '{focus}' focus"
                    state["thoughts"].append(f"Re-planning: {strategy_description}")
                    break
            else:
                # All focuses exhausted, proceed with what we have
                strategy_description = "All research focuses attempted, proceeding with available data"
                state["thoughts"].append(f"Re-planning: {strategy_description}")

        elif ReflectionReason.NO_SENTIMENT in reasons:
            # Sentiment analysis failed - retry it
            next_step = "analyze_sentiment"
            strategy_description = "Will retry sentiment analysis"
            state["thoughts"].append(f"Re-planning: {strategy_description}")

        elif ReflectionReason.NO_FINANCIAL_DATA in reasons:
            # Financial data fetch failed - not much we can do differently, proceed
            strategy_description = "Financial data unavailable, proceeding with available data"
            state["thoughts"].append(f"Re-planning: {strategy_description}")

        state["reflection_strategy"] = strategy.value

        agent_logger.planning(
            plan=[next_step],
            reasoning=f"Re-planning ({strategy.value}): {strategy_description}" if strategy != ReflectionStrategy.NONE else "Proceeding to synthesis",
        )

        state["current_plan"] = [next_step]
        return state

    async def _retrieve_historical_context_node(self, state: GraphState) -> GraphState:
        """
        Retrieve historical context from past analyses using embeddings.

        This node queries for:
        - Similar past analyses for the same ticker
        - Sentiment history to identify trends
        """
        agent_logger = get_agent_logger(state["job_id"])

        # Skip if no embeddings provider or storage
        if not self._embeddings or not self._storage:
            agent_logger.thought(
                thought_type="memory",
                content="Historical context retrieval skipped (no embeddings provider)",
            )
            state["historical_context"] = None
            return state

        agent_logger.thought(
            thought_type="memory",
            content=f"Retrieving historical context for {state['ticker']}",
        )

        try:
            from src.application.services.embedding_service import EmbeddingService

            service = EmbeddingService(self._embeddings, self._storage)

            # Build a preliminary summary for similarity search
            preliminary_summary = f"{state['company_name']} ({state['ticker']}) analysis"
            if state.get("sentiment_result"):
                preliminary_summary += f" - sentiment: {state['sentiment_result'].overall_sentiment}"

            context = await service.get_historical_context(
                ticker=state["ticker"],
                current_summary=preliminary_summary,
                limit=3,
            )

            state["historical_context"] = context

            num_similar = len(context.get("similar_analyses", []))
            num_history = len(context.get("sentiment_history", []))

            agent_logger.thought(
                thought_type="memory",
                content=f"Retrieved {num_similar} similar analyses and {num_history} sentiment history points",
            )

        except Exception as e:
            logger.warning(f"Failed to retrieve historical context: {e}")
            agent_logger.thought(
                thought_type="error",
                content=f"Historical context retrieval failed: {str(e)}",
            )
            state["historical_context"] = None

        return state

    async def _synthesize_node(self, state: GraphState) -> GraphState:
        """
        Synthesize all gathered data into a professional investment report.

        Uses LLM for intelligent synthesis rather than template-based generation.
        """
        agent_logger = get_agent_logger(state["job_id"])
        agent_logger.synthesis_start()

        context = self._build_synthesis_context(state)

        try:
            synthesis = await self._llm_synthesis(context, state)
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}, falling back to template")
            synthesis = self._fallback_synthesis(state)

        # Build citation sources
        citation_sources = []
        if state.get("news_result"):
            citation_sources.extend([a.url for a in state["news_result"].articles[:5]])
        if state.get("web_research"):
            citation_sources.extend([r.url for r in state["web_research"].results[:5]])
        citation_sources = list(dict.fromkeys(citation_sources))[:10]  # Dedupe, limit to 10

        # Build financial snapshot
        financial_snapshot = None
        if state.get("financial_data"):
            financial_snapshot = self._build_enhanced_financial_snapshot(state["financial_data"])

        # Build analyst consensus
        analyst_consensus = None
        if state.get("financial_data"):
            fd = state["financial_data"]
            if fd.analyst_target_mean or fd.analyst_recommendation:
                analyst_consensus = {
                    "target_mean": fd.analyst_target_mean,
                    "target_low": fd.analyst_target_low,
                    "target_high": fd.analyst_target_high,
                    "recommendation": fd.analyst_recommendation,
                    "number_of_analysts": fd.number_of_analysts,
                }

        # Get sentiment score
        sentiment_score = state["sentiment_result"].sentiment_score if state.get("sentiment_result") else 0.0

        # Build the report
        report_dict = {
            "company_ticker": state["ticker"],
            "company_name": state["company_name"],
            "analysis_summary": synthesis.get("analysis_summary", self._fallback_summary(state)),
            "sentiment_score": sentiment_score,
            "key_findings": synthesis.get("key_findings", self._extract_key_findings(state)),
            "tools_used": list(set(state.get("tools_used", []))),
            "citation_sources": citation_sources,
            "news_summary": self._build_news_summary(state),
            "financial_snapshot": financial_snapshot,
            "investment_thesis": synthesis.get("investment_thesis"),
            "risk_factors": synthesis.get("risk_factors", []),
            "competitive_context": synthesis.get("competitive_context"),
            "analyst_consensus": analyst_consensus,
            "catalyst_events": synthesis.get("catalyst_events", []),
            "financial_analysis": synthesis.get("financial_analysis"),
            "web_research": self._build_web_research_snapshot(state),
            # reflection_triggered should be True if any reflection occurred (reflection_count > 0)
            # reflection_notes captures the reason(s) for reflection
            "reflection_notes": state.get("reflection_reason") if state.get("reflection_count", 0) > 0 else None,
            "reflection_triggered": state.get("reflection_count", 0) > 0,
            "analysis_type": "ON_DEMAND",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "iteration_count": state.get("iteration", 1),
        }

        agent_logger.synthesis_complete(
            summary_length=len(report_dict["analysis_summary"]),
            findings_count=len(report_dict["key_findings"]),
            sentiment_score=sentiment_score,
        )

        state["final_report"] = report_dict
        state["status"] = "COMPLETED"
        state["completed_steps"].append("synthesize")

        return state

    # =========================================================================
    # Synthesis Helper Methods
    # =========================================================================

    def _build_synthesis_context(self, state: GraphState) -> Dict[str, Any]:
        """Build comprehensive context for LLM synthesis."""
        context = {
            "company": state["company_name"],
            "ticker": state["ticker"],
        }

        # Financial data context
        if state.get("financial_data"):
            fd = state["financial_data"]
            context["financial_data"] = self._format_financial_for_prompt(fd)
        else:
            context["financial_data"] = "No financial data available."

        # News context
        if state.get("news_result") and state["news_result"].articles:
            articles = state["news_result"].articles
            context["news_count"] = len(articles)
            context["news_headlines"] = "\n".join([
                f"- {a.title} ({a.source}, {a.published_at.strftime('%Y-%m-%d')})"
                for a in articles[:10]
            ])
        else:
            context["news_count"] = 0
            context["news_headlines"] = "No recent news available."

        # Sentiment context
        if state.get("sentiment_result"):
            sr = state["sentiment_result"]
            context["sentiment_score"] = sr.sentiment_score
            context["sentiment_label"] = sr.overall_sentiment
            context["positive_count"] = sr.positive_count
            context["negative_count"] = sr.negative_count
            context["neutral_count"] = sr.neutral_count
            # Extract key themes from article sentiments
            themes = []
            for article_sentiment in sr.article_sentiments[:5]:
                themes.extend(article_sentiment.key_phrases[:2])
            context["sentiment_themes"] = f"Key themes: {', '.join(themes[:8])}" if themes else ""
        else:
            context["sentiment_score"] = 0.0
            context["sentiment_label"] = "unknown"
            context["positive_count"] = 0
            context["negative_count"] = 0
            context["neutral_count"] = 0
            context["sentiment_themes"] = ""

        # Web research context
        if state.get("web_research") and state["web_research"].results:
            results = state["web_research"].results
            context["web_research"] = "\n".join([
                f"- {r.title}: {r.snippet[:150]}..." if len(r.snippet) > 150 else f"- {r.title}: {r.snippet}"
                for r in results[:8]
            ])
        else:
            context["web_research"] = "No additional web research available."

        # Historical context from embeddings
        if state.get("historical_context"):
            from src.application.services.embedding_service import EmbeddingService
            # Create a temporary service just for formatting (doesn't need real providers)
            context["historical_context"] = self._format_historical_context(state["historical_context"])
        else:
            context["historical_context"] = "No historical data available (first analysis for this ticker)."

        return context

    def _format_historical_context(self, historical_context: Dict[str, Any]) -> str:
        """Format historical context for the synthesis prompt."""
        parts: List[str] = []

        # Format similar past analyses
        similar = historical_context.get("similar_analyses", [])
        if similar:
            parts.append("### Similar Past Analyses")
            for i, analysis in enumerate(similar[:3], 1):
                content = analysis.get("content_text", "")[:200]
                score = analysis.get("score", 0)
                metadata = analysis.get("metadata", {})
                date = metadata.get("generated_at", "")[:10] if metadata.get("generated_at") else "Unknown"
                parts.append(
                    f"{i}. [{date}] (relevance: {score:.0%}): \"{content}...\""
                )
        else:
            parts.append("No similar past analyses found.")

        # Format sentiment trend
        sentiment_history = historical_context.get("sentiment_history", [])
        if sentiment_history:
            parts.append("\n### Sentiment Trend")

            # Calculate trend direction
            if len(sentiment_history) >= 2:
                recent = sentiment_history[0].get("sentiment_score", 0)
                oldest = sentiment_history[-1].get("sentiment_score", 0)
                diff = recent - oldest

                if diff > 0.1:
                    trend = "IMPROVING"
                elif diff < -0.1:
                    trend = "DECLINING"
                else:
                    trend = "STABLE"

                parts.append(f"Overall trend: {trend} ({diff:+.2f} change)")

            # Show recent history points
            for point in sentiment_history[:5]:
                date = point.get("date", "")[:10] if point.get("date") else "Unknown"
                score = point.get("sentiment_score", 0)
                parts.append(f"- {date}: sentiment {score:+.2f}")
        else:
            parts.append("\nNo sentiment history available.")

        return "\n".join(parts)

    def _format_financial_for_prompt(self, fd: FinancialData) -> str:
        """Format financial data for the synthesis prompt."""
        lines = []

        # Price and valuation
        if fd.current_price:
            lines.append(f"Current Price: ${fd.current_price:.2f}")
        if fd.price_change_percent:
            lines.append(f"Price Change (24h): {fd.price_change_percent:+.2f}%")
        if fd.market_cap:
            lines.append(f"Market Cap: ${fd.market_cap / 1e9:.2f}B")
        if fd.pe_ratio:
            lines.append(f"P/E Ratio (TTM): {fd.pe_ratio:.2f}")
        if fd.forward_pe:
            lines.append(f"Forward P/E: {fd.forward_pe:.2f}")
        if fd.peg_ratio:
            lines.append(f"PEG Ratio: {fd.peg_ratio:.2f}")
        if fd.price_to_book:
            lines.append(f"Price/Book: {fd.price_to_book:.2f}")
        if fd.price_to_sales:
            lines.append(f"Price/Sales: {fd.price_to_sales:.2f}")

        # 52-week range
        if fd.fifty_two_week_low and fd.fifty_two_week_high:
            lines.append(f"52-Week Range: ${fd.fifty_two_week_low:.2f} - ${fd.fifty_two_week_high:.2f}")

        # Profitability
        if fd.gross_margin:
            lines.append(f"Gross Margin: {fd.gross_margin * 100:.1f}%")
        if fd.operating_margin:
            lines.append(f"Operating Margin: {fd.operating_margin * 100:.1f}%")
        if fd.profit_margin:
            lines.append(f"Net Profit Margin: {fd.profit_margin * 100:.1f}%")

        # Growth
        if fd.revenue_growth:
            lines.append(f"Revenue Growth (YoY): {fd.revenue_growth * 100:.1f}%")
        if fd.earnings_growth:
            lines.append(f"Earnings Growth (YoY): {fd.earnings_growth * 100:.1f}%")

        # Balance sheet
        if fd.total_cash:
            lines.append(f"Cash: ${fd.total_cash / 1e9:.2f}B")
        if fd.total_debt:
            lines.append(f"Total Debt: ${fd.total_debt / 1e9:.2f}B")
        if fd.debt_to_equity:
            lines.append(f"Debt/Equity: {fd.debt_to_equity:.2f}")
        if fd.current_ratio:
            lines.append(f"Current Ratio: {fd.current_ratio:.2f}")

        # Analyst data
        if fd.analyst_target_mean:
            lines.append(f"Analyst Mean Target: ${fd.analyst_target_mean:.2f}")
        if fd.analyst_target_low and fd.analyst_target_high:
            lines.append(f"Analyst Range: ${fd.analyst_target_low:.2f} - ${fd.analyst_target_high:.2f}")
        if fd.analyst_recommendation:
            lines.append(f"Analyst Consensus: {fd.analyst_recommendation.upper()}")
        if fd.number_of_analysts:
            lines.append(f"Number of Analysts: {fd.number_of_analysts}")

        # Quarterly revenue
        if fd.quarterly_revenue:
            lines.append("Quarterly Revenue:")
            for q in fd.quarterly_revenue[:4]:
                yoy = f" ({q.year_over_year_change:+.1f}% YoY)" if q.year_over_year_change else ""
                lines.append(f"  {q.quarter}: ${q.revenue / 1e9:.2f}B{yoy}")

        # Other
        if fd.dividend_yield:
            lines.append(f"Dividend Yield: {fd.dividend_yield * 100:.2f}%")
        if fd.beta:
            lines.append(f"Beta: {fd.beta:.2f}")
        if fd.sector:
            lines.append(f"Sector: {fd.sector}")
        if fd.industry:
            lines.append(f"Industry: {fd.industry}")

        return "\n".join(lines) if lines else "Limited financial data available."

    async def _llm_synthesis(self, context: Dict[str, Any], state: GraphState) -> Dict[str, Any]:
        """Use LLM for intelligent report synthesis."""
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(**context)

        response = await self._llm.complete(
            messages=[
                LLMMessage(role="system", content=INVESTMENT_ANALYST_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        # Parse JSON response
        content = response.content.strip()

        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            synthesis = json.loads(content)
            return synthesis

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM synthesis JSON: {e}")
            # Try to extract key parts with regex as fallback
            return self._parse_synthesis_fallback(content)

    def _parse_synthesis_fallback(self, content: str) -> Dict[str, Any]:
        """Fallback parsing if JSON fails."""
        result = {}

        # Try to extract analysis_summary
        if '"analysis_summary"' in content:
            match = re.search(r'"analysis_summary"\s*:\s*"([^"]+)"', content)
            if match:
                result["analysis_summary"] = match.group(1)

        # Try to extract key_findings
        if '"key_findings"' in content:
            match = re.search(r'"key_findings"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            if match:
                findings = re.findall(r'"([^"]+)"', match.group(1))
                result["key_findings"] = findings[:7]

        return result

    def _fallback_synthesis(self, state: GraphState) -> Dict[str, Any]:
        """Fallback template-based synthesis if LLM fails."""
        return {
            "analysis_summary": self._fallback_summary(state),
            "key_findings": self._extract_key_findings(state),
            "risk_factors": self._extract_risk_factors(state),
            "investment_thesis": None,
            "competitive_context": None,
            "catalyst_events": [],
            "financial_analysis": None,
        }

    def _fallback_summary(self, state: GraphState) -> str:
        """Build fallback analysis summary."""
        parts = [f"{state['company_name']} ({state['ticker']}) analysis based on recent data."]

        if state.get("sentiment_result"):
            sr = state["sentiment_result"]
            sentiment_desc = {
                "positive": "positive", "negative": "negative",
                "neutral": "neutral", "mixed": "mixed"
            }.get(sr.overall_sentiment, "mixed")
            parts.append(f"Market sentiment is {sentiment_desc} (score: {sr.sentiment_score:.2f}).")

        if state.get("financial_data"):
            fd = state["financial_data"]
            if fd.current_price:
                parts.append(f"Trading at ${fd.current_price:.2f}")
                if fd.analyst_target_mean:
                    upside = ((fd.analyst_target_mean - fd.current_price) / fd.current_price) * 100
                    parts.append(f"with analyst target of ${fd.analyst_target_mean:.2f} ({upside:+.1f}% upside).")

        return " ".join(parts)

    def _build_enhanced_financial_snapshot(self, fd: FinancialData) -> Dict[str, Any]:
        """Build enhanced financial snapshot for the report."""
        snapshot = {
            # Price & basics
            "current_price": fd.current_price,
            "price_change_percent": fd.price_change_percent,
            "market_cap": fd.market_cap,
            "market_cap_formatted": f"${fd.market_cap / 1e9:.2f}B" if fd.market_cap else None,
            # 52-week range
            "52_week_high": fd.fifty_two_week_high,
            "52_week_low": fd.fifty_two_week_low,
            # Valuation
            "pe_ratio": fd.pe_ratio,
            "forward_pe": fd.forward_pe,
            "peg_ratio": fd.peg_ratio,
            "price_to_book": fd.price_to_book,
            "price_to_sales": fd.price_to_sales,
            "enterprise_value": fd.enterprise_value,
            "ev_to_ebitda": fd.ev_to_ebitda,
            # Profitability (raw values for UI to format)
            "gross_margin": fd.gross_margin,
            "operating_margin": fd.operating_margin,
            "profit_margin": fd.profit_margin,
            # Growth (raw values for UI to format)
            "revenue_growth": fd.revenue_growth,
            "earnings_growth": fd.earnings_growth,
            # Balance sheet
            "total_cash": fd.total_cash,
            "total_debt": fd.total_debt,
            "debt_to_equity": fd.debt_to_equity,
            "current_ratio": fd.current_ratio,
            # Analyst data
            "analyst_target_mean": fd.analyst_target_mean,
            "analyst_target_low": fd.analyst_target_low,
            "analyst_target_high": fd.analyst_target_high,
            "analyst_recommendation": fd.analyst_recommendation,
            "number_of_analysts": fd.number_of_analysts,
            # Other
            "dividend_yield": fd.dividend_yield,
            "beta": fd.beta,
            "sector": fd.sector,
            "industry": fd.industry,
        }

        # Add quarterly revenue if available
        if fd.quarterly_revenue:
            snapshot["recent_revenue"] = [
                {
                    "quarter": q.quarter,
                    "revenue": q.revenue,
                    "revenue_formatted": f"${q.revenue / 1e9:.2f}B" if q.revenue else None,
                    "yoy_change": q.year_over_year_change,
                }
                for q in fd.quarterly_revenue[:4]
            ]

        return snapshot

    def _build_web_research_snapshot(self, state: GraphState) -> Optional[Dict[str, Any]]:
        """Build web research snapshot for the report."""
        web_research = state.get("web_research")
        if not web_research or not web_research.results:
            return None

        return {
            "results": [
                {
                    "title": r.title,
                    "snippet": r.snippet,
                    "url": r.url,
                    "source": r.source,
                    "published_date": getattr(r, "published_date", None),
                }
                for r in web_research.results[:15]  # Limit to 15 results
            ],
            "queries_used": web_research.queries_used,
            "research_focus": web_research.research_focus,
            "total_results": web_research.total_results,
        }

    # =========================================================================
    # Routing Functions
    # =========================================================================

    def _should_continue_or_replan(self, state: GraphState) -> str:
        """Decide whether to re-plan or proceed to historical context retrieval."""
        if state.get("reflection_triggered"):
            return "replan"
        return "retrieve_history"

    def _route_after_replan(self, state: GraphState) -> str:
        """Route to the appropriate node after re-planning."""
        plan = state.get("current_plan", [])
        if plan:
            return plan[0] if plan[0] in ["collect_data", "analyze_sentiment"] else "retrieve_history"
        return "retrieve_history"

    # =========================================================================
    # Legacy Helper Methods (kept for fallback)
    # =========================================================================

    def _extract_key_findings(self, state: GraphState) -> List[str]:
        """Extract key findings from the analysis (fallback method)."""
        findings = []

        # Sentiment findings
        if state.get("sentiment_result"):
            sr = state["sentiment_result"]
            if sr.sentiment_score > 0.3:
                findings.append(
                    f"Strong positive market sentiment (score: {sr.sentiment_score:.2f}) "
                    f"with {sr.positive_count} of {sr.positive_count + sr.negative_count + sr.neutral_count} articles positive"
                )
            elif sr.sentiment_score < -0.3:
                findings.append(
                    f"Negative market sentiment (score: {sr.sentiment_score:.2f}) suggests caution"
                )
            else:
                findings.append(
                    f"Mixed sentiment (score: {sr.sentiment_score:.2f}) indicates market uncertainty"
                )

        # Financial findings
        if state.get("financial_data"):
            fd = state["financial_data"]

            # Analyst target comparison
            if fd.analyst_target_mean and fd.current_price:
                upside = ((fd.analyst_target_mean - fd.current_price) / fd.current_price) * 100
                if upside > 15:
                    findings.append(
                        f"Trading {abs(upside):.1f}% below analyst mean target of ${fd.analyst_target_mean:.2f}"
                    )
                elif upside < -15:
                    findings.append(
                        f"Trading {abs(upside):.1f}% above analyst mean target of ${fd.analyst_target_mean:.2f}"
                    )

            # Revenue growth
            if fd.revenue_growth:
                if fd.revenue_growth > 0.15:
                    findings.append(f"Strong revenue growth of {fd.revenue_growth * 100:.1f}% YoY")
                elif fd.revenue_growth < -0.05:
                    findings.append(f"Revenue declining {abs(fd.revenue_growth * 100):.1f}% YoY")

            # Profitability
            if fd.profit_margin:
                if fd.profit_margin > 0.15:
                    findings.append(f"Healthy profit margin of {fd.profit_margin * 100:.1f}%")
                elif fd.profit_margin < 0:
                    findings.append(f"Currently unprofitable with {fd.profit_margin * 100:.1f}% margin")

            # Valuation
            if fd.pe_ratio:
                if fd.pe_ratio > 50:
                    findings.append(f"Premium valuation with P/E of {fd.pe_ratio:.1f}")
                elif fd.pe_ratio < 15 and fd.pe_ratio > 0:
                    findings.append(f"Value territory with P/E of {fd.pe_ratio:.1f}")

        # Ensure minimum findings
        while len(findings) < 3:
            if state.get("financial_data") and state["financial_data"].sector:
                findings.append(f"Operating in {state['financial_data'].sector} sector")
            else:
                findings.append(f"Additional research on {state['company_name']} recommended")

        return findings

    def _extract_risk_factors(self, state: GraphState) -> List[str]:
        """Extract risk factors from the data."""
        risks = []

        if state.get("financial_data"):
            fd = state["financial_data"]

            if fd.debt_to_equity and fd.debt_to_equity > 2:
                risks.append(f"High leverage with debt/equity of {fd.debt_to_equity:.1f}")

            if fd.current_ratio and fd.current_ratio < 1:
                risks.append(f"Liquidity concern with current ratio of {fd.current_ratio:.2f}")

            if fd.beta and fd.beta > 1.5:
                risks.append(f"High volatility stock with beta of {fd.beta:.2f}")

            if fd.short_ratio and fd.short_ratio > 10:
                risks.append(f"Elevated short interest (ratio: {fd.short_ratio:.1f})")

        if state.get("sentiment_result"):
            sr = state["sentiment_result"]
            if sr.negative_count > sr.positive_count:
                risks.append("Negative news flow may pressure stock price")

        return risks

    def _build_news_summary(self, state: GraphState) -> Optional[str]:
        """Build a summary of news coverage."""
        if not state.get("news_result") or not state["news_result"].articles:
            return None

        articles = state["news_result"].articles
        sources = list(set(a.source for a in articles))

        summary_parts = [
            f"Analyzed {len(articles)} recent news articles from {', '.join(sources[:3])}."
        ]

        if articles:
            summary_parts.append(f"Key headline: {articles[0].title}")

        return " ".join(summary_parts)
