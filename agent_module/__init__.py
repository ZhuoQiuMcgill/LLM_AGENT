"""
LLM Agent Module

A simple, reliable, and easy-to-use Python module for building and managing
agents that interact with different Large Language Models (LLMs).

This module provides a unified Agent abstraction that encapsulates core LLM
interaction logic, allowing developers to create agents powered by different
LLMs through dependency injection.
"""

import logging
import sys

# Set up basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core classes
from .agent import Agent
from .llm.base import LLMInterface
from .llm.openai_llm import GPTModel

# Import GeminiModel conditionally to avoid crashing on missing dependencies
try:
    from .llm.google_llm import GeminiModel
except ImportError:
    logger.warning("Google Gemini support is not available. Install required dependencies with: "
                   "pip install google-generativeai")

from .history.memory import HistoryManager, InMemoryHistoryManager
from .exceptions import AgentError, LLMAPIError, ConfigurationError, HistoryError
from .config import load_config, create_llm_config
from .utils.helpers import load_env_file, get_api_key, setup_logging

# Export public API
__all__ = [
    # Core classes
    "Agent",
    "LLMInterface",
    "GPTModel",
    "GeminiModel",
    "HistoryManager",
    "InMemoryHistoryManager",

    # Exceptions
    "AgentError",
    "LLMAPIError",
    "ConfigurationError",
    "HistoryError",

    # Configuration
    "load_config",
    "create_llm_config",

    # Utilities
    "load_env_file",
    "get_api_key",
    "setup_logging",
]

# Set up package version
__version__ = "1.0.0"
