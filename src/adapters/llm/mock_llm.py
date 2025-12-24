"""
Mock LLM Provider - For testing without external API calls.

Provides scripted responses for testing the agent flow.
Supports both text completion and structured output.
"""

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from src.application.ports.llm_port import (
    LLMMessage,
    LLMPort,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
)
from src.config.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class MockLLM(LLMPort):
    """
    Mock LLM implementation for testing.

    Provides realistic scripted responses that allow testing
    the full agent flow without external API dependencies.
    """

    def __init__(self):
        self._call_count = 0
        self._responses: Dict[str, Any] = {}

    async def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[LLMToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Generate a mock completion."""
        self._call_count += 1

        # Log the call for debugging
        logger.debug(
            "mock_llm_complete",
            call_count=self._call_count,
            message_count=len(messages),
            has_tools=bool(tools),
        )

        # Extract context from messages
        last_message = messages[-1].content if messages else ""

        # If tools are provided and tool_choice suggests using them
        if tools and tool_choice != "none":
            tool_call = self._determine_tool_call(last_message, tools)
            if tool_call:
                return LLMResponse(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                )

        # Generate text response
        content = self._generate_response(last_message)

        return LLMResponse(
            content=content,
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 150},
        )

    async def complete_structured(
        self,
        messages: List[LLMMessage],
        response_model: Type[T],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> T:
        """Generate a structured mock response."""
        self._call_count += 1

        # Get the model name to determine what response to generate
        model_name = response_model.__name__

        logger.debug(
            "mock_llm_structured",
            call_count=self._call_count,
            response_model=model_name,
        )

        # Generate appropriate mock data based on model type
        mock_data = self._get_mock_structured_response(model_name, messages)

        return response_model(**mock_data)

    async def analyze_sentiment(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        self._call_count += 1

        # Simple mock sentiment analysis
        text_lower = text.lower()

        # Count sentiment indicators
        positive_words = ["good", "great", "excellent", "strong", "growth", "positive", "up", "surge", "profit", "success"]
        negative_words = ["bad", "poor", "weak", "decline", "negative", "down", "loss", "fail", "concern", "risk"]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            sentiment = "neutral"
            score = 0.0
            confidence = 0.5
        elif positive_count > negative_count:
            sentiment = "positive"
            score = min(0.9, positive_count / max(total, 1))
            confidence = 0.8
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(-0.9, -negative_count / max(total, 1))
            confidence = 0.8
        else:
            # Equal positive and negative - classify as neutral
            sentiment = "neutral"
            score = 0.0
            confidence = 0.6

        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
        }

    async def health_check(self) -> bool:
        """Mock is always healthy."""
        return True

    def _determine_tool_call(
        self,
        message: str,
        tools: List[LLMToolDefinition],
    ) -> Optional[LLMToolCall]:
        """Determine which tool to call based on message content."""
        message_lower = message.lower()

        # Extract ticker if present
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', message)
        ticker = ticker_match.group(1) if ticker_match else "TSLA"

        # Extract company name
        company_patterns = [
            r"(?:analyze|research|about)\s+(.+?)(?:\s*\(|\s*stock|\s*$)",
            r"prospects?\s+of\s+(.+?)(?:\s*\(|\s*$)",
        ]
        company_name = "Tesla, Inc."
        for pattern in company_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                company_name = match.group(1).strip()
                break

        # Determine tool based on context
        for tool in tools:
            if tool.name == "news_retriever" and any(
                kw in message_lower for kw in ["news", "articles", "recent", "analyze", "research"]
            ):
                return LLMToolCall(
                    id=f"call_{self._call_count}",
                    name="news_retriever",
                    arguments={
                        "company": company_name,
                        "ticker": ticker,
                        "num_articles": 5,
                    },
                )
            elif tool.name == "sentiment_analyzer" and any(
                kw in message_lower for kw in ["sentiment", "feeling", "opinion", "analyze news"]
            ):
                return LLMToolCall(
                    id=f"call_{self._call_count}",
                    name="sentiment_analyzer",
                    arguments={
                        "ticker": ticker,
                        "company_name": company_name,
                    },
                )
            elif tool.name == "data_fetcher" and any(
                kw in message_lower for kw in ["financial", "stock", "price", "data", "revenue"]
            ):
                return LLMToolCall(
                    id=f"call_{self._call_count}",
                    name="data_fetcher",
                    arguments={"ticker": ticker},
                )

        return None

    def _generate_response(self, message: str) -> str:
        """Generate a contextual text response."""
        message_lower = message.lower()

        if "plan" in message_lower or "analyze" in message_lower:
            return (
                "I'll analyze this company by:\n"
                "1. Fetching recent news articles\n"
                "2. Analyzing sentiment from the news\n"
                "3. Getting financial data\n"
                "4. Synthesizing findings into a report"
            )
        elif "reflect" in message_lower:
            return (
                "Reviewing the gathered information:\n"
                "- News coverage is recent and relevant\n"
                "- Sentiment analysis shows mixed signals\n"
                "- Financial data is complete\n"
                "The information is sufficient for analysis."
            )
        elif "synthesize" in message_lower or "report" in message_lower:
            return (
                "Based on the analysis:\n"
                "The company shows strong fundamentals with recent positive news coverage. "
                "Market sentiment is cautiously optimistic. Key risks include market volatility "
                "and competitive pressures."
            )
        else:
            return "I understand. Let me proceed with the analysis."

    def _get_mock_structured_response(
        self,
        model_name: str,
        messages: List[LLMMessage],
    ) -> Dict[str, Any]:
        """Generate mock structured response based on model type."""
        if model_name == "SentimentResult":
            return {
                "positive_count": 3,
                "negative_count": 1,
                "neutral_count": 1,
                "positive_ratio": 0.6,
                "negative_ratio": 0.2,
                "neutral_ratio": 0.2,
                "overall_sentiment": "positive",
                "sentiment_score": 0.45,
                "article_sentiments": [
                    {
                        "title": "Company Reports Strong Q3 Earnings",
                        "sentiment": "positive",
                        "confidence": 0.85,
                        "key_phrases": ["strong earnings", "beat expectations"],
                    },
                    {
                        "title": "Market Concerns Over Competition",
                        "sentiment": "negative",
                        "confidence": 0.75,
                        "key_phrases": ["competition", "market share concerns"],
                    },
                ],
                "analysis_reasoning": "Based on the news articles, sentiment is predominantly positive due to strong financial performance.",
            }
        elif model_name == "AnalysisReport":
            return {
                "company_ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "analysis_summary": (
                    "Tesla continues to demonstrate strong market position in the EV sector. "
                    "Recent news indicates positive momentum with strong Q3 deliveries and "
                    "expanding production capacity. However, increased competition and "
                    "supply chain considerations remain key factors to monitor."
                ),
                "sentiment_score": 0.45,
                "key_findings": [
                    "Strong Q3 delivery numbers exceeded analyst expectations",
                    "Expansion of Gigafactory production capacity on track",
                    "Increased competition in EV market requires continued innovation",
                ],
                "tools_used": ["news_retriever", "sentiment_analyzer", "data_fetcher"],
                "citation_sources": [
                    "https://example.com/article1",
                    "https://example.com/article2",
                ],
                "news_summary": "Recent coverage focuses on Q3 performance and expansion plans.",
                "financial_snapshot": {
                    "current_price": 250.00,
                    "market_cap": 800000000000,
                    "pe_ratio": 65.5,
                },
                "reflection_notes": None,
                "reflection_triggered": False,
                "analysis_type": "ON_DEMAND",
                "iteration_count": 1,
            }
        else:
            # Default mock data
            return {}

    def set_response(self, key: str, response: Any) -> None:
        """Set a custom response for testing."""
        self._responses[key] = response

    def get_call_count(self) -> int:
        """Get the number of calls made."""
        return self._call_count

    def reset(self) -> None:
        """Reset the mock state."""
        self._call_count = 0
        self._responses = {}
