"""
LLM package for the agent module.

This package provides the interfaces and implementations for different
Large Language Models (LLMs) that can be used with the Agent.
"""

import importlib.util
import logging

logger = logging.getLogger(__name__)

# Always import the base interface
from .base import LLMInterface

# Always import the OpenAI implementation
from .openai_llm import GPTModel

# Conditionally import the Google implementation
try:
    from .google_llm import GeminiModel, MissingDependencyError

    _has_gemini = True
except ImportError as e:
    logger.warning(f"Google Gemini implementation not available: {str(e)}")
    logger.warning("To use GeminiModel, install the required package: pip install google-generativeai")
    _has_gemini = False


    # Create a placeholder class for easier imports
    class GeminiModel:
        """Placeholder for GeminiModel when the dependency is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The 'google-generativeai' package is required to use GeminiModel. "
                "Please install it using: pip install google-generativeai"
            )

# Define the public API
__all__ = [
    "LLMInterface",
    "GPTModel",
    "GeminiModel",
]
