"""
Utility functions for the agent module.
"""

import os
import logging
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_env_file(env_file: Optional[str] = None) -> bool:
    """
    Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file. If None, will search in current directory.

    Returns:
        True if the .env file was loaded successfully, False otherwise.
    """
    # If no env_file provided, look for .env in current directory
    if env_file is None:
        env_file = ".env"

    # Convert to Path object
    env_path = Path(env_file)

    # Check if the file exists
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path.absolute()}")
        return False

    # Load the environment variables
    try:
        load_dotenv(env_path)
        logger.debug(f"Loaded environment variables from {env_path.absolute()}")
        return True
    except Exception as e:
        logger.error(f"Error loading .env file: {str(e)}")
        return False


def get_api_key(env_var_name: str, env_file: Optional[str] = None) -> Optional[str]:
    """
    Get an API key from environment variables.

    This function will:
    1. Try to get the API key from environment variables
    2. If not found and env_file is provided, try to load from .env file
    3. Try again from environment variables

    Args:
        env_var_name: Name of the environment variable containing the API key.
        env_file: Optional path to a .env file.

    Returns:
        The API key if found, None otherwise.
    """
    # First try to get from current environment
    api_key = os.getenv(env_var_name)

    # If not found and env_file provided, try to load from .env
    if api_key is None and env_file is not None:
        logger.debug(f"API key {env_var_name} not found in environment, trying to load from .env file")
        load_env_file(env_file)
        api_key = os.getenv(env_var_name)

    # Log success or failure
    if api_key:
        logger.debug(f"Found API key for {env_var_name}")
    else:
        logger.warning(f"API key {env_var_name} not found")

    return api_key


def setup_logging(
        level: int = logging.INFO,
        format_string: Optional[str] = None,
        log_file: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: The logging level (default: INFO).
        format_string: The log format string. If None, a default format will be used.
        log_file: Optional path to a log file.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = []

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)

    # Add file handler if log_file is provided
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_string))
            handlers.append(file_handler)
        except Exception as e:
            logger.error(f"Error setting up log file: {str(e)}")

    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )

    logger.debug("Logging setup complete")
