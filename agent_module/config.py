"""
Configuration management for the agent module.
"""

import os
import logging
from typing import Dict, Any, Optional

from .utils.helpers import load_env_file, get_api_key

logger = logging.getLogger(__name__)


def load_config(
        env_file: Optional[str] = None,
        include_api_keys: bool = True
) -> Dict[str, Any]:
    """
    Load configuration settings from environment variables and/or .env file.

    Args:
        env_file: Optional path to a .env file.
        include_api_keys: Whether to include API keys in the returned configuration.

    Returns:
        A dictionary with configuration settings.
    """
    # Load environment variables from .env file if provided
    if env_file:
        load_env_file(env_file)

    config = {}

    # Load API keys if requested
    if include_api_keys:
        # OpenAI API key
        openai_api_key = get_api_key("OPENAI_API_KEY")
        if openai_api_key:
            config["openai_api_key"] = openai_api_key

        # Google API key
        google_api_key = get_api_key("GOOGLE_API_KEY")
        if google_api_key:
            config["google_api_key"] = google_api_key

    # Load other configuration settings from environment variables
    # Example: Default models
    config["default_openai_model"] = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4")
    config["default_google_model"] = os.getenv("DEFAULT_GOOGLE_MODEL", "gemini-pro")

    # Example: Default system prompt
    config["default_system_prompt"] = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a helpful assistant."
    )

    # Example: History settings
    try:
        max_history = int(os.getenv("MAX_HISTORY_MESSAGES", "50"))
        config["max_history_messages"] = max_history
    except (ValueError, TypeError):
        config["max_history_messages"] = 50
        logger.warning("Invalid MAX_HISTORY_MESSAGES, using default value of 50")

    # Example: Logging settings
    config["log_level"] = os.getenv("LOG_LEVEL", "INFO")
    config["log_file"] = os.getenv("LOG_FILE")

    logger.debug(f"Loaded configuration with {len(config)} settings")

    return config


def create_llm_config(config: Dict[str, Any], llm_type: str) -> Dict[str, Any]:
    """
    Create a configuration dictionary for a specific LLM type.

    Args:
        config: The main configuration dictionary.
        llm_type: The type of LLM (e.g., "openai", "google").

    Returns:
        A dictionary with LLM-specific configuration settings.
    """
    llm_config = {}

    if llm_type == "openai":
        llm_config["api_key"] = config.get("openai_api_key")
        llm_config["model"] = config.get("default_openai_model", "gpt-4")
        llm_config["organization"] = config.get("openai_organization")

        # Example: Add other OpenAI-specific settings
        llm_config["temperature"] = float(config.get("openai_temperature", "0.7"))
        llm_config["max_tokens"] = int(config.get("openai_max_tokens", "1000"))

    elif llm_type == "google":
        llm_config["api_key"] = config.get("google_api_key")
        llm_config["model"] = config.get("default_google_model", "gemini-pro")

        # Example: Add other Google-specific settings
        llm_config["temperature"] = float(config.get("google_temperature", "0.7"))
        llm_config["max_output_tokens"] = int(config.get("google_max_tokens", "1000"))

    # Add common settings
    llm_config["system_prompt"] = config.get("default_system_prompt")

    return llm_config