# LLM Agent Module

A simple, reliable, and easy-to-use Python module for building and managing agents that interact with different Large
Language Models (LLMs).

This module provides a unified Agent abstraction that encapsulates core LLM interaction logic, allowing developers to
create agents powered by different LLMs through dependency injection.

## Features

- **Unified Agent Interface**: Create agents that work with any supported LLM
- **Multiple LLM Support**:
    - OpenAI GPT models (GPT-4, GPT-3.5-turbo, etc.)
    - Google Gemini models (gemini-pro, etc.)
- **Conversation History Management**: Built-in support for maintaining conversation context
- **Async/Await Support**: Modern asynchronous API for non-blocking operations
- **Simple Configuration**: Easy API key management through environment variables
- **Comprehensive Error Handling**: Clear and specific error messages
- **Vision Capabilities**: Support for processing images with compatible models
- **System Prompts**: Define agent personality and behavior with customizable system prompts

## Installation

Clone the repository:

```bash
git clone https://github.com/ZhuoQiuMcgill/LLM_AGENT.git
cd LLM_AGENT
```

Install dependencies:

```bash
pip install -r requirements.txt
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

## Agent Class: Detailed Usage

The `Agent` class is the central component of this module. It wraps different LLM implementations with a unified
interface and manages conversation history.

### Creating an Agent

```python
from agent_module import Agent, GPTModel, InMemoryHistoryManager

# Create a custom history manager (optional)
history_manager = InMemoryHistoryManager(max_messages=50)

# Create an agent with the GPT model
agent = Agent(
    llm_interface=GPTModel(model="gpt-4o"),
    history_manager=history_manager,  # Optional
    name="GPT Assistant",  # Optional
    system_prompt="You are a helpful assistant specialized in explaining complex topics in simple terms.",
    image_system_prompt="You are a visual assistant that can analyze images and provide detailed explanations.",
    vision_model_override="gpt-4o",  # Override model for image processing
    max_image_size=(1024, 1024),  # Resize images before processing
)
```

### Processing Regular Text Inputs

```python
# Process a single text message
response = await agent.process("Can you explain quantum computing in simple terms?")

# Alternative method name (alias for process)
response = await agent.run("What is the difference between AI, ML, and deep learning?")

# Add configuration overrides for a specific request
response = await agent.process(
    "Write a creative story about space exploration.",
    config={
        "temperature": 0.9,  # Increase creativity
        "max_output_tokens": 2000,  # Request longer response
    }
)
```

### Processing Images with Text

The module supports processing images along with text prompts:

```python
# Process text with an image file
response = await agent.process_with_image(
    "What can you tell me about this image?",
    image_path="path/to/your/image.jpg"
)

# Process with custom config
response = await agent.process_with_image(
    "Explain what's happening in this diagram.",
    image_path="path/to/diagram.png",
    config={
        "temperature": 0.2,  # More precise responses
        "max_output_tokens": 1000,  # Control response length
    }
)

# Process with binary image data (e.g., from web uploads)
with open("path/to/image.jpg", "rb") as img_file:
    image_data = img_file.read()

response = await agent.process_with_image_bin(
    "Describe the objects in this image.",
    image_data=image_data,
    mime_type="image/jpeg"  # Optional, will try to detect if not provided
)
```

### System Prompts and Image Processing

The system prompt is used to define the agent's behavior and capabilities. When processing both text and images, the
system prompt applies to the combined input:

```python
# Create an agent with a system prompt that works for text and images
agent = Agent(
    llm_interface=GPTModel(model="gpt-4o"),
    system_prompt="You are a helpful assistant. When analyzing images, be detailed and focus on key elements."
)

# The system prompt will apply to this text+image request as a single unit
response = await agent.process_with_image(
    "What's in this image and what's significant about it?",
    image_path="path/to/image.jpg"
)
```

For specialized image processing needs, you can provide a separate image system prompt:

```python
agent = Agent(
    llm_interface=GPTModel(model="gpt-4o"),
    system_prompt="You are a helpful assistant for general questions.",
    image_system_prompt="You are a visual analysis expert. When examining images, describe all visual elements in detail."
)
```

### Example: Same System Instruction for Text and Image

This example demonstrates how the same system instruction influences both text-only and image+text interactions:

```python
import asyncio
from agent_module import Agent, GPTModel, load_env_file

# Load environment variables
load_env_file()


async def main():
    # Create an agent with a specific persona via system instruction
    agent = Agent(
        llm_interface=GPTModel(model="gpt-4o"),
        system_prompt="You are a marine biology expert. Always relate your answers to ocean life and ecosystems."
    )

    # Process text-only input - system instruction applies
    text_response = await agent.process(
        "What makes ecosystems resilient to change?"
    )
    print("Text-only response (influenced by marine biology persona):")
    print(text_response)
    print("\n" + "-" * 50 + "\n")

    # Process text+image input - same system instruction applies
    image_response = await agent.process_with_image(
        "What do you observe in this image?",
        image_path="coral_reef.jpg"
    )
    print("Image+text response (influenced by same marine biology persona):")
    print(image_response)

    # Both responses will reflect the marine biology expertise defined in the system prompt,
    # showing that the same system instruction applies to both text and image processing


asyncio.run(main())
```

Example output would show that both responses incorporate marine biology expertise, with the text-only response
discussing ecosystem resilience in ocean contexts, and the image response analyzing the coral reef image from a marine
biology perspective - all from the same system instruction.

The implementation ensures that this system instruction is applied as a unified context for both text and image inputs,
without requiring separate configuration.

### Managing Conversation History

```python
# Reset conversation history
agent.reset_history()

# Create an agent that limits history
from agent_module import InMemoryHistoryManager

history = InMemoryHistoryManager(max_messages=10)
agent = Agent(llm_interface=llm, history_manager=history)

# History is maintained automatically across multiple calls
response1 = await agent.process("Hello, I'm researching renewable energy.")
response2 = await agent.process("What are the most promising technologies?")
# The agent will remember the context from the first message
```

### Testing Connection

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

## LLM Implementations

### OpenAI GPT Models

```python
from agent_module import GPTModel

# Create a GPT model implementation
gpt = GPTModel(
    model="gpt-4o",  # Optional: The GPT model to use
    api_key=None,  # Optional: OpenAI API key (if None, loads from env)
    organization=None,  # Optional: OpenAI organization ID
    max_output_tokens=8192,  # Optional: Maximum number of tokens to generate
    temperature=1.0,  # Optional: Controls randomness (0.0 to 2.0)
    default_system_prompt=None,  # Optional: Default system prompt
)
```

### Google Gemini Models

```python
from agent_module import GeminiModel

# Create a Gemini model implementation
gemini = GeminiModel(
    model="gemini-pro",  # Optional: The Gemini model to use
    api_key=None,  # Optional: Google AI API key (if None, loads from env)
    temperature=0.7,  # Optional: Controls randomness (0.0 to 1.0)
    max_output_tokens=65536,  # Optional: Maximum number of tokens to generate
    top_p=0.95,  # Optional: Nucleus sampling parameter
    top_k=40,  # Optional: Top-k sampling parameter
    default_system_prompt=None,  # Optional: Default system prompt
)
```

## Advanced Usage

### Using Different Models for Text and Images

You can configure the Agent to use different models for text-only vs. image+text processing:

```python
from agent_module import Agent, GPTModel

# Create an agent that uses different models for different tasks
agent = Agent(
    llm_interface=GPTModel(model="gpt-3.5-turbo"),  # Lower cost for text-only
    vision_model_override="gpt-4o",  # Higher capability for vision
)

# Text-only uses the base model (gpt-3.5-turbo)
await agent.process("What's the capital of France?")

# Image processing uses the override model (gpt-4o)
await agent.process_with_image("What's in this image?", "image.jpg")
```

### Handling Images from Web Applications

For web applications receiving file uploads:

```python
async def handle_upload(file_data, prompt):
    # file_data is the binary data from the upload
    response = await agent.process_with_image_bin(
        prompt,
        image_data=file_data,
        mime_type="image/jpeg"  # Set based on upload info
    )
    return response
```

### Error Handling

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

## Utility Functions

### Configuration Loading

```python
from agent_module import load_config, create_llm_config

# Load configuration from environment variables or .env file
config = load_config(
    env_file=None,  # Optional: Path to .env file
    include_api_keys=True  # Optional: Whether to include API keys in config
)

# Create a model-specific configuration
openai_config = create_llm_config(
    config,  # Required: The main configuration dictionary
    "openai"  # Required: LLM type ("openai" or "google")
)
```

### Environment File Loading

```python
from agent_module import load_env_file

# Load environment variables from .env file
load_env_file(
    env_file=None  # Optional: Path to .env file (default: ".env")
)
```

### API Key Retrieval

```python
from agent_module import get_api_key

# Get API key from environment variables
api_key = get_api_key(
    env_var_name,  # Required: Name of the environment variable
    env_file=None  # Optional: Path to .env file
)
```

### Logging Setup

```python
from agent_module import setup_logging
import logging

# Set up logging
setup_logging(
    level=logging.INFO,  # Optional: Logging level
    format_string=None,  # Optional: Log format string
    log_file=None  # Optional: Path to log file
)
```

## Real-World Examples

### Chat Application Example

```python
import asyncio
from agent_module import Agent, GPTModel, load_env_file

# Load API keys
load_env_file()

# Set up the agent
agent = Agent(
    llm_interface=GPTModel(model="gpt-4"),
    system_prompt="You are a friendly and helpful assistant named Alex."
)


# Simple chat loop
async def chat_loop():
    print("Chat started. Type 'exit' to end the conversation.")
    print("Assistant: Hi! I'm Alex. How can I help you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Assistant: Goodbye! Have a great day!")
            break

        response = await agent.process(user_input)
        print(f"Assistant: {response}")


# Run the chat loop
asyncio.run(chat_loop())
```

### Image Analysis Application

```python
import asyncio
import os
from agent_module import Agent, GPTModel, load_env_file

# Load API keys
load_env_file()

# Set up the agent with vision capabilities
agent = Agent(
    llm_interface=GPTModel(model="gpt-4o"),
    system_prompt="You are a visual analysis assistant that specializes in describing images in detail."
)


# Function to analyze images in a directory
async def analyze_images(directory, query="Describe this image in detail."):
    results = {}

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_path = os.path.join(directory, filename)
            print(f"Analyzing {filename}...")

            try:
                response = await agent.process_with_image(query, file_path)
                results[filename] = response
                print(f"✓ Completed analysis of {filename}")
            except Exception as e:
                print(f"✗ Error analyzing {filename}: {str(e)}")
                results[filename] = f"Error: {str(e)}"

    return results


# Run the analysis
async def main():
    results = await analyze_images("./images", "What objects do you see in this image?")

    # Print or save results
    for filename, analysis in results.items():
        print(f"\n--- {filename} ---\n{analysis}\n")


asyncio.run(main())
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.