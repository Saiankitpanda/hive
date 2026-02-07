# Kimi (Moonshot AI) Provider Integration

This document describes the integration of Kimi/Moonshot AI as a first-class LLM provider in the Hive Agent Framework.

## Overview

Kimi is Moonshot AI's flagship model with unique reasoning capabilities. This integration provides:

- **reasoning_content**: Access to model's thinking process
- **Thinking Mode**: Deep reasoning for complex multi-step tasks
- **Faster convergence**: Cached reasoning patterns reduce redundant API calls
- **Future-ready**: Support for fine-tuning models

## Installation

```bash
# Kimi uses LiteLLM under the hood
pip install litellm
```

Set your API key:
```bash
export MOONSHOT_API_KEY="your-key-here"
# or
export KIMI_API_KEY="your-key-here"
```

Get your API key at: https://platform.moonshot.cn/

## Available Models

| Model | Description | Thinking Mode |
|-------|-------------|---------------|
| `kimi-k2.5` | Advanced multimodal, vision + text | Enabled by default |
| `kimi-k2-thinking` | Deep reasoning, forced thinking | Always enabled |
| `kimi-k2-instruct` | Fast reflex-grade responses | Disabled |
| `moonshot-v1-8k` | Legacy 8K context | N/A |
| `moonshot-v1-32k` | Legacy 32K context | N/A |
| `moonshot-v1-128k` | Legacy 128K context | N/A |

## Usage

### Basic Usage

```python
from framework.llm import KimiProvider

# Default model (kimi-k2.5)
provider = KimiProvider()

# Generate completion
response = provider.complete(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    system="You are a helpful AI assistant."
)

print(response.content)
```

### Accessing Reasoning Content

```python
from framework.llm import KimiProvider, KimiResponse

provider = KimiProvider(model="kimi-k2-thinking")

response = provider.complete(
    messages=[{"role": "user", "content": "Solve: 2x + 5 = 15"}]
)

# Access the model's thinking process
if isinstance(response, KimiResponse):
    print("Reasoning:", response.reasoning_content)
    print("Thinking tokens:", response.thinking_tokens)
    print("Answer:", response.content)
```

### Tool Use with Reasoning

```python
from framework.llm import KimiProvider, Tool, ToolUse, ToolResult

provider = KimiProvider()

tools = [
    Tool(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    )
]

def execute_tool(tool_use: ToolUse) -> ToolResult:
    if tool_use.name == "calculate":
        result = eval(tool_use.input["expression"])
        return ToolResult(tool_use_id=tool_use.id, content=str(result))
    return ToolResult(tool_use_id=tool_use.id, content="Unknown tool", is_error=True)

response = provider.complete_with_tools(
    messages=[{"role": "user", "content": "What is 25 * 47?"}],
    system="Use the calculate tool for math.",
    tools=tools,
    tool_executor=execute_tool
)

# Reasoning is preserved across tool iterations
print("Full reasoning:", response.reasoning_content)
```

### Fast Mode (No Thinking)

```python
# Use kimi-k2-instruct for quick responses
provider = KimiProvider(model="kimi-k2-instruct")

response = provider.complete(
    messages=[{"role": "user", "content": "Hello!"}]
)
# No reasoning_content in instruct mode
```

### Caching for Similar Errors

```python
provider = KimiProvider(cache_reasoning=True)

# First call - generates reasoning
response1 = provider.complete(
    messages=[{"role": "user", "content": "Error: TypeError in line 42"}]
)

# Later, check cache for similar patterns
cached = provider.get_cached_reasoning(
    messages=[{"role": "user", "content": "Error: TypeError in line 42"}]
)
if cached:
    print("Using cached reasoning:", cached)
```

## Benefits for Hive Agents

### 1. Improved Agent Building
Kimi's reasoning transparency helps the coding agent:
- Understand complex goal requirements
- Generate better test cases
- Debug issues with visible thought process

### 2. Reduced API Calls
Cached reasoning patterns mean:
- Similar errors don't need full re-reasoning
- Faster iteration during agent development
- Lower costs in production

### 3. Better Tool Usage
Deep reasoning ensures:
- Correct tool selection
- Accurate parameter extraction
- Multi-step task orchestration

## Configuration Options

```python
provider = KimiProvider(
    model="kimi-k2.5",           # Model selection
    api_key="your-key",          # Optional, uses env var if not set
    thinking_mode=True,          # Enable thinking (default: True)
    cache_reasoning=True,        # Cache reasoning patterns (default: True)
)
```

## Comparison with Other Providers

| Feature | KimiProvider | LiteLLMProvider | AnthropicProvider |
|---------|-------------|-----------------|-------------------|
| Reasoning content | ✅ | ❌ | ❌ |
| Thinking tokens | ✅ | ❌ | ❌ |
| Reasoning cache | ✅ | ❌ | ❌ |
| OpenAI compatible | ✅ | ✅ | ❌ |
| Tool use | ✅ | ✅ | ✅ |

## Future Enhancements

- **Fine-tuning support**: Custom model training on your data
- **Vision integration**: Kimi K2.5 multimodal capabilities
- **Agent swarm**: Parallel sub-agent orchestration

## Related Resources

- [Moonshot AI Platform](https://platform.moonshot.cn/)
- [Kimi API Documentation](https://platform.moonshot.ai/docs)
- [LiteLLM Moonshot Integration](https://docs.litellm.ai/docs/providers/moonshot)
