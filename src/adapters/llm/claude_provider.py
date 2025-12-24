"""
Claude LLM Provider - Anthropic Claude Sonnet 4.5 integration.

This module implements the LLM port using Claude for:
- Text completion with tool use
- Structured output generation
- Sentiment analysis
"""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar

import anthropic
from pydantic import BaseModel

from src.application.ports.llm_port import (
    LLMMessage,
    LLMPort,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
)
from src.config.logging import get_logger
from src.domain.exceptions import LLMProviderError, LLMRateLimitError, LLMTimeoutError

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeProvider(LLMPort):
    """
    Claude Sonnet 4.5 LLM provider via Anthropic API.

    Supports:
    - Tool use (function calling)
    - Structured output with JSON schema
    - Sentiment analysis
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 120,
    ):
        """
        Initialize the Claude provider.

        Args:
            api_key: Anthropic API key
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    async def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[LLMToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Generate a completion using Claude."""
        try:
            # Convert messages to Claude format
            claude_messages = self._convert_messages(messages)

            # Build request kwargs
            kwargs: Dict[str, Any] = {
                "model": self._model,
                "max_tokens": max_tokens or self._max_tokens,
                "messages": claude_messages,
            }

            # Add system message if present
            system_messages = [m for m in messages if m.role == "system"]
            if system_messages:
                kwargs["system"] = system_messages[0].content

            # Add temperature if not using tools
            if not tools:
                kwargs["temperature"] = temperature or self._temperature

            # Add tools if provided
            if tools:
                kwargs["tools"] = self._convert_tools(tools)
                if tool_choice:
                    if tool_choice == "required":
                        kwargs["tool_choice"] = {"type": "any"}
                    elif tool_choice == "none":
                        kwargs["tool_choice"] = {"type": "none"}
                    elif tool_choice != "auto":
                        kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

            logger.debug(
                "claude_request",
                model=self._model,
                message_count=len(claude_messages),
                has_tools=bool(tools),
            )

            response = await self._client.messages.create(**kwargs)

            return self._parse_response(response)

        except anthropic.RateLimitError as e:
            logger.error("claude_rate_limit", error=str(e))
            raise LLMRateLimitError(provider="anthropic")
        except anthropic.APITimeoutError as e:
            logger.error("claude_timeout", error=str(e))
            raise LLMTimeoutError(provider="anthropic", timeout_seconds=self._timeout)
        except anthropic.APIError as e:
            logger.error("claude_api_error", error=str(e))
            raise LLMProviderError(
                message=f"Claude API error: {e}",
                provider="anthropic",
            )

    async def complete_structured(
        self,
        messages: List[LLMMessage],
        response_model: Type[T],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> T:
        """Generate a structured response matching a Pydantic model."""

        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        structured_prompt = f"""
Please respond with a JSON object that matches this schema:

```json
{schema_str}
```

Return ONLY the JSON object, no additional text.
"""

        enhanced_messages = list(messages) + [
            LLMMessage(role="user", content=structured_prompt)
        ]

        response = await self.complete(
            messages=enhanced_messages,
            max_tokens=max_tokens,
            temperature=temperature or 0.3,
        )

        # Parse JSON from response
        if response.content:
            try:
                # Try to extract JSON from response
                content = response.content.strip()

                # Handle markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                data = json.loads(content)
                return response_model(**data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    "claude_structured_parse_error",
                    error=str(e),
                    content=response.content[:200],
                )
                raise LLMProviderError(
                    message=f"Failed to parse structured response: {e}",
                    provider="anthropic",
                )

        raise LLMProviderError(
            message="Empty response from Claude",
            provider="anthropic",
        )

    async def analyze_sentiment(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text using Claude with market-aware understanding.

        This prompt is specifically designed for financial news analysis,
        focusing on market impact rather than just emotional tone.
        """
        context_str = f" about {context}" if context else ""

        prompt = f"""You are a financial sentiment analyst specializing in market-moving news.

Analyze the MARKET IMPACT of this news{context_str}. Focus on how this news would likely affect the stock price, not just the emotional tone of the text.

News text:
---
{text}
---

Consider these market-relevant factors:
- Earnings beats/misses and guidance changes
- Analyst upgrades/downgrades or price target changes
- Product launches, partnerships, or competitive developments
- Regulatory approvals or legal developments
- Management changes, insider activity, or strategic shifts
- Macro factors affecting the company or sector

Classify the expected MARKET REACTION:
- "positive": News likely to drive stock higher (beat expectations, positive surprise, growth catalysts)
- "negative": News likely to drive stock lower (miss expectations, negative surprise, headwinds)
- "neutral": News unlikely to significantly move the stock (already priced in, routine updates)

Respond with ONLY a JSON object:
{{
    "sentiment": "positive" | "negative" | "neutral",
    "score": -1.0 to 1.0 (magnitude of expected market reaction: 0.8+ is very significant, 0.3-0.7 is moderate, <0.3 is minor),
    "confidence": 0.0 to 1.0 (how confident are you in this assessment),
    "key_phrases": ["specific phrase 1", "specific phrase 2", "specific phrase 3"],
    "market_implication": "One sentence on expected market reaction"
}}"""

        response = await self.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.2,  # Lower for more consistent analysis
        )

        if response.content:
            try:
                content = response.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                result = json.loads(content)
                # Ensure required fields exist
                return {
                    "sentiment": result.get("sentiment", "neutral"),
                    "score": float(result.get("score", 0.0)),
                    "confidence": float(result.get("confidence", 0.5)),
                    "key_phrases": result.get("key_phrases", []),
                    "market_implication": result.get("market_implication", ""),
                }
            except (json.JSONDecodeError, ValueError):
                # Fall back to simple analysis
                return {
                    "sentiment": "neutral",
                    "score": 0.0,
                    "confidence": 0.5,
                    "key_phrases": [],
                    "market_implication": "",
                }

        return {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.5,
            "key_phrases": [],
            "market_implication": "",
        }

    async def health_check(self) -> bool:
        """Check if Claude is available."""
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return response.content is not None
        except Exception as e:
            logger.error("claude_health_check_failed", error=str(e))
            return False

    def _convert_messages(
        self,
        messages: List[LLMMessage],
    ) -> List[Dict[str, Any]]:
        """Convert internal message format to Claude format."""
        claude_messages = []

        for msg in messages:
            if msg.role == "system":
                continue

            claude_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        return claude_messages

    def _convert_tools(
        self,
        tools: List[LLMToolDefinition],
    ) -> List[Dict[str, Any]]:
        """Convert internal tool format to Claude format."""
        claude_tools = []

        for tool in tools:
            claude_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })

        return claude_tools

    def _parse_response(
        self,
        response: anthropic.types.Message,
    ) -> LLMResponse:
        """Parse Claude response to internal format."""
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    LLMToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        # Determine finish reason
        if response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        )
