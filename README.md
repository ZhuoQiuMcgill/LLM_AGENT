# LLM Agent Module

A simple, reliable, and easy-to-use Python module for building and managing agents that interact with different Large Language Models (LLMs).

This module provides a unified Agent abstraction that encapsulates core LLM interaction logic, allowing developers to create agents powered by different LLMs through dependency injection.

## Features

- **Unified Agent Interface**: Create agents that work with any supported LLM
- **Multiple LLM Support**: 
  - OpenAI GPT models (GPT-4, GPT-3.5-turbo, etc.)
  - Google Gemini models (gemini-pro, etc.)
- **Conversation History Management**: Built-in support for maintaining conversation context
- **Async/Await Support**: Modern asynchronous API for non-blocking operations
- **Simple Configuration**: Easy API key management through environment variables
- **Comprehensive Error Handling**: Clear and specific error messages

## Installation

```bash
pip install llm-agent-module
```

## Quick Start

```python
import asyncio
from agent_module import Agent, GPTModel, load_env_file

# Load environment variables from .env file (API keys)
load_env_file()

# Create an LLM implementation (OpenAI GPT-4)
llm = GPTModel(model="gpt-4")

# Create an agent with the LLM implementation
agent = Agent(
    llm_interface=llm,
    name="GPT Assistant",
    system_prompt="You are a helpful assistant."
)

# Use the agent
async def chat():
    response = await agent.process("Hello! Can you help me with a question?")
    print(f'Agent: {response}')
    
    # Continue the conversation
    response = await agent.process("What's the capital of France?")
    print(f'Agent: {response}')

# Run the async function
asyncio.run(chat())
```

## Environment Setup

Create a `.env` file in your project root with your API keys:

```
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORGANIZATION=your_organization_id_here  # Optional

# Google API Key (for Gemini models)
GOOGLE_API_KEY=your_google_api_key_here
```

## Supported LLM Implementations

### OpenAI GPT Models

```python
from agent_module import GPTModel

# Create a GPT model implementation
gpt = GPTModel(
    model="gpt-4",  # or "gpt-3.5-turbo", etc.
    temperature=0.7,
    max_tokens=1000,
    default_system_prompt="You are a helpful assistant."
)
```

### Google Gemini Models

```python
from agent_module import GeminiModel

# Create a Gemini model implementation
gemini = GeminiModel(
    model="gemini-pro",
    temperature=0.7,
    max_output_tokens=1000,
    default_system_prompt="You are a helpful assistant."
)
```

## Managing Conversation History

By default, the Agent will create an InMemoryHistoryManager instance. You can also create one explicitly:

```python
from agent_module import InMemoryHistoryManager, Agent

# Create a history manager
history = InMemoryHistoryManager(max_messages=50)

# Create an agent with the history manager
agent = Agent(
    llm_interface=llm,
    history_manager=history,
    name="Assistant"
)

# Reset conversation history
agent.reset_history()
```

## Advanced Configuration

```python
from agent_module import load_config, create_llm_config

# Load configuration from environment variables or .env file
config = load_config(env_file=".env")

# Create a model-specific configuration
openai_config = create_llm_config(config, "openai")
google_config = create_llm_config(config, "google")
```

## Connecting to Different LLMs

You can easily switch between different LLM providers:

```python
# OpenAI GPT model
gpt_agent = Agent(
    llm_interface=GPTModel(),
    name="GPT Assistant"
)

# Google Gemini model
gemini_agent = Agent(
    llm_interface=GeminiModel(),
    name="Gemini Assistant"
)
```

## Testing Connection

Before starting a conversation, you can test the connection to the LLM:

```python
async def test_connection():
    try:
        result = await agent.connection_test()
        print(f"Connection test result: {result}")
    except Exception as e:
        print(f"Connection test failed: {str(e)}")

asyncio.run(test_connection())
```

## Error Handling

The module provides custom exceptions for different error scenarios:

```python
from agent_module import AgentError, LLMAPIError, ConfigurationError, HistoryError

try:
    response = await agent.process("Hello")
except LLMAPIError as e:
    print(f"API Error: {str(e)}")
except ConfigurationError as e:
    print(f"Configuration Error: {str(e)}")
except AgentError as e:
    print(f"General Agent Error: {str(e)}")
```

## Logging

The module includes built-in logging:

```python
from agent_module import setup_logging
import logging

# Set up detailed logging
setup_logging(level=logging.DEBUG, log_file="agent.log")
```