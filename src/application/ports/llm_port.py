"""
LLM Provider Port - Abstract interface for LLM operations.

This port defines the contract for LLM providers (Claude, OpenAI, Mock).
Implementations must support both text completion and structured output.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMMessage(BaseModel):
    """A single message in the conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class LLMToolCall(BaseModel):
    """A tool call made by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


class LLMResponse(BaseModel):
    """Response from the LLM."""

    content: Optional[str] = None
    tool_calls: List[LLMToolCall] = []
    finish_reason: str = "stop"  # "stop", "tool_calls", "length"
    usage: Dict[str, int] = {}  # tokens used


class LLMToolDefinition(BaseModel):
    """Definition of a tool for the LLM."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema


class LLMPort(ABC):
    """
    Abstract interface for LLM providers.

    Implementations:
    - ClaudeProvider: Claude Sonnet 4.5 via Anthropic API
    - MockLLM: Mock implementation for testing
    """

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[LLMToolDefinition]] = None,
        tool_choice: Optional[str] = None,  # "auto", "required", "none", or specific tool name
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: Conversation history
            tools: Available tools for function calling
            tool_choice: How to handle tool calls
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLM response with content and/or tool calls
        """
        pass

    @abstractmethod
    async def complete_structured(
        self,
        messages: List[LLMMessage],
        response_model: Type[T],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> T:
        """
        Generate a structured response matching a Pydantic model.

        Args:
            messages: Conversation history
            response_model: Pydantic model class for response
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Parsed response as the specified Pydantic model
        """
        pass

    @abstractmethod
    async def analyze_sentiment(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            context: Optional context (e.g., company name)

        Returns:
            Sentiment analysis result
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM provider is healthy.

        Returns:
            True if provider is available
        """
        pass
