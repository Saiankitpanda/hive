"""Kimi (Moonshot AI) provider for enhanced reasoning capabilities.

Kimi provides OpenAI-compatible API with additional reasoning features:
- reasoning_content field for transparent thinking process
- Thinking mode for complex multi-step tasks
- Faster convergence with reduced API calls
- Support for fine-tuning

Models:
- kimi-k2.5: Advanced multimodal with thinking mode (default enabled)
- kimi-k2-thinking: Forced thinking mode for deep reasoning
- kimi-k2-instruct: Fast reflex-grade responses

See: https://platform.moonshot.ai/docs
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]

from framework.llm.provider import LLMProvider, LLMResponse, Tool, ToolResult, ToolUse


@dataclass
class KimiResponse(LLMResponse):
    """Extended response with Kimi reasoning content."""

    reasoning_content: str = ""
    """The model's internal reasoning process (thinking mode)."""

    thinking_tokens: int = 0
    """Tokens used for reasoning/thinking."""


# Model mappings for Kimi
KIMI_MODELS = {
    # Thinking models (with reasoning_content)
    "kimi-k2.5": "moonshot/kimi-k2.5",
    "kimi-k2-thinking": "moonshot/kimi-k2-thinking",
    # Fast instruct model (no thinking)
    "kimi-k2-instruct": "moonshot/kimi-k2-instruct",
    # Legacy models
    "moonshot-v1-8k": "moonshot/moonshot-v1-8k",
    "moonshot-v1-32k": "moonshot/moonshot-v1-32k",
    "moonshot-v1-128k": "moonshot/moonshot-v1-128k",
}


class KimiProvider(LLMProvider):
    """
    Kimi (Moonshot AI) provider with reasoning capabilities.

    Extends LiteLLM to support Kimi's unique features:
    - reasoning_content: Access model's thinking process
    - Thinking mode: Deep reasoning for complex tasks
    - Efficient caching: Reduce redundant API calls

    Usage:
        # Default thinking model
        provider = KimiProvider()

        # Specific model
        provider = KimiProvider(model="kimi-k2-thinking")

        # Fast responses (no thinking)
        provider = KimiProvider(model="kimi-k2-instruct")

        # Access reasoning
        response = provider.complete(messages)
        if isinstance(response, KimiResponse):
            print(f"Reasoning: {response.reasoning_content}")
    """

    # Moonshot API base URL
    API_BASE = "https://api.moonshot.ai/v1"

    def __init__(
        self,
        model: str = "kimi-k2.5",
        api_key: str | None = None,
        thinking_mode: bool = True,
        cache_reasoning: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the Kimi provider.

        Args:
            model: Kimi model to use (kimi-k2.5, kimi-k2-thinking, etc.)
            api_key: Moonshot API key. If not provided, uses MOONSHOT_API_KEY
                     or KIMI_API_KEY environment variable.
            thinking_mode: Enable thinking mode for reasoning_content access.
                          Note: kimi-k2-thinking has forced thinking mode.
            cache_reasoning: Cache reasoning patterns to reduce API calls.
            **kwargs: Additional arguments passed to LiteLLM.
        """
        # Resolve model name
        self.model = KIMI_MODELS.get(model, model)
        if not self.model.startswith("moonshot/"):
            self.model = f"moonshot/{model}"

        # Resolve API key
        self.api_key = api_key or os.environ.get("MOONSHOT_API_KEY", os.environ.get("KIMI_API_KEY"))

        self.thinking_mode = thinking_mode
        self.cache_reasoning = cache_reasoning
        self.extra_kwargs = kwargs

        # Reasoning cache for similar error patterns
        self._reasoning_cache: dict[str, str] = {}

        if litellm is None:
            raise ImportError("LiteLLM required for Kimi provider. Install: pip install litellm")

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[Tool] | None = None,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        json_mode: bool = False,
    ) -> KimiResponse:
        """Generate a completion with optional reasoning content."""
        # Prepare messages
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # Add JSON mode instruction
        if json_mode:
            json_instruction = "\n\nRespond with valid JSON only."
            if full_messages and full_messages[0]["role"] == "system":
                full_messages[0]["content"] += json_instruction
            else:
                full_messages.insert(0, {"role": "system", "content": json_instruction.strip()})

        # Build kwargs for LiteLLM
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "api_base": self.API_BASE,
            **self.extra_kwargs,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key

        if tools:
            kwargs["tools"] = [self._tool_to_openai_format(t) for t in tools]

        if response_format:
            kwargs["response_format"] = response_format

        # Make the call
        response = litellm.completion(**kwargs)  # type: ignore[union-attr]

        # Extract content
        message = response.choices[0].message
        content = message.content or ""

        # Extract reasoning_content if available (Kimi thinking mode)
        reasoning_content = ""
        thinking_tokens = 0

        if hasattr(message, "reasoning_content"):
            reasoning_content = message.reasoning_content or ""
        elif hasattr(response.choices[0], "reasoning_content"):
            reasoning_content = response.choices[0].reasoning_content or ""

        # Cache reasoning for similar patterns
        if self.cache_reasoning and reasoning_content:
            cache_key = self._get_cache_key(messages)
            self._reasoning_cache[cache_key] = reasoning_content

        # Get usage info
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Some Kimi models report thinking tokens separately
        if hasattr(usage, "reasoning_tokens"):
            thinking_tokens = usage.reasoning_tokens

        return KimiResponse(
            content=content,
            model=response.model or self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=response.choices[0].finish_reason or "",
            raw_response=response,
            reasoning_content=reasoning_content,
            thinking_tokens=thinking_tokens,
        )

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[Tool],
        tool_executor: Callable[[ToolUse], ToolResult],
        max_iterations: int = 10,
        max_tokens: int = 4096,
    ) -> KimiResponse:
        """Run tool-use loop with reasoning preservation."""
        current_messages = []
        if system:
            current_messages.append({"role": "system", "content": system})
        current_messages.extend(messages)

        total_input_tokens = 0
        total_output_tokens = 0
        total_thinking_tokens = 0
        accumulated_reasoning = []

        openai_tools = [self._tool_to_openai_format(t) for t in tools]

        for iteration in range(max_iterations):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": current_messages,
                "max_tokens": max_tokens,
                "tools": openai_tools,
                "api_base": self.API_BASE,
                **self.extra_kwargs,
            }

            if self.api_key:
                kwargs["api_key"] = self.api_key

            response = litellm.completion(**kwargs)  # type: ignore[union-attr]

            # Track tokens
            usage = response.usage
            if usage:
                total_input_tokens += usage.prompt_tokens
                total_output_tokens += usage.completion_tokens
                if hasattr(usage, "reasoning_tokens"):
                    total_thinking_tokens += usage.reasoning_tokens

            choice = response.choices[0]
            message = choice.message

            # Capture reasoning content from this iteration
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                accumulated_reasoning.append(f"[Step {iteration + 1}] {message.reasoning_content}")

            # Check if done
            if choice.finish_reason == "stop" or not message.tool_calls:
                return KimiResponse(
                    content=message.content or "",
                    model=response.model or self.model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    stop_reason=choice.finish_reason or "stop",
                    raw_response=response,
                    reasoning_content="\n\n".join(accumulated_reasoning),
                    thinking_tokens=total_thinking_tokens,
                )

            # Process tool calls
            current_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            # Execute tools
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_use = ToolUse(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=args,
                )

                result = tool_executor(tool_use)

                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_use_id,
                        "content": result.content,
                    }
                )

        # Max iterations reached
        return KimiResponse(
            content="Max tool iterations reached",
            model=self.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            stop_reason="max_iterations",
            raw_response=None,
            reasoning_content="\n\n".join(accumulated_reasoning),
            thinking_tokens=total_thinking_tokens,
        )

    def get_cached_reasoning(self, messages: list[dict[str, Any]]) -> str | None:
        """Get cached reasoning for similar message patterns."""
        cache_key = self._get_cache_key(messages)
        return self._reasoning_cache.get(cache_key)

    def clear_reasoning_cache(self) -> None:
        """Clear the reasoning cache."""
        self._reasoning_cache.clear()

    def _get_cache_key(self, messages: list[dict[str, Any]]) -> str:
        """Generate cache key from messages (error patterns)."""
        # Use last user message as key for similar error caching
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Extract error signature if present
                if "error" in content.lower():
                    return content[:200]  # First 200 chars as key
        return ""

    def _tool_to_openai_format(self, tool: Tool) -> dict[str, Any]:
        """Convert Tool to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters.get("properties", {}),
                    "required": tool.parameters.get("required", []),
                },
            },
        }
