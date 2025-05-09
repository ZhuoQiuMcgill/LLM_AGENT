"""
Utility functions for the agent module.
"""

import os
import logging
from typing import Optional, List, Union, Tuple
from pathlib import Path
import json

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


def validate_file_path(filepath: str) -> str:
    """
    Validate and normalize file path.

    Args:
        filepath: Path to validate

    Returns:
        Absolute path
    """
    return os.path.abspath(filepath)


def validate_readable_file(filepath: str, max_size_mb: int = 100) -> None:
    """
    Validate that a file exists, is readable, and not too large.

    Args:
        filepath: Path to the file
        max_size_mb: Maximum allowed file size in MB

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If not a regular file or too large
        PermissionError: If no read permission
    """
    filepath = validate_file_path(filepath)

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check if it's a regular file (not a directory or device)
    if not os.path.isfile(filepath):
        raise ValueError(f"Not a regular file: {filepath}")

    # Check read permissions
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"No read permission for {filepath}")

    # Check file size to prevent memory issues
    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = os.path.getsize(filepath)
    if file_size > max_size_bytes:
        raise ValueError(f"File too large ({file_size} bytes). Maximum size is {max_size_mb}MB")


def read_file_content(filepath: str, as_lines: bool = False) -> Union[str, List[str]]:
    """
    Read content from a file with proper encoding and error handling.

    Args:
        filepath: Path to the file
        as_lines: If True, return a list of lines, else return the entire content

    Returns:
        File content as string or list of lines

    Raises:
        Various IOError exceptions if reading fails
    """
    # Validate the file
    validate_readable_file(filepath)

    # Try reading with UTF-8 encoding first
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
            if as_lines:
                lines = file.readlines()
                return [line.rstrip('\n') for line in lines]
            else:
                return file.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding if utf-8 fails
        try:
            with open(filepath, 'r', encoding='latin-1') as file:
                if as_lines:
                    lines = file.readlines()
                    return [line.rstrip('\n') for line in lines]
                else:
                    return file.read()
        except Exception as e:
            raise IOError(f"Failed to read file with multiple encodings: {e}")
    except Exception as e:
        raise IOError(f"Failed to read file: {e}")


def validate_writable_path(filepath: str, create_dirs: bool = True) -> Tuple[str, str]:
    """
    Validate that a file path is writable and optionally create directories.

    Args:
        filepath: Path to validate
        create_dirs: Whether to create missing directories

    Returns:
        Tuple of (normalized filepath, directory)

    Raises:
        IOError: If directory creation fails
        PermissionError: If writing is not allowed
    """
    filepath = validate_file_path(filepath)
    directory = os.path.dirname(filepath)

    # Create directories if they don't exist
    if create_dirs and directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except (OSError, PermissionError) as e:
            raise IOError(f"Failed to create directory: {e}")

    # Check write permissions
    try:
        # Check if file exists and we can write to it
        if os.path.exists(filepath):
            if not os.access(filepath, os.W_OK):
                raise PermissionError(f"No write permission for {filepath}")
        # Check if we can write to the directory
        elif directory and not os.access(directory, os.W_OK):
            raise PermissionError(f"No write permission for directory {directory}")
    except Exception as e:
        raise IOError(f"Permission check failed: {e}")

    return filepath, directory


def get_sentences(filepath: str) -> List[str]:
    """
    Read a file and return its content as a list of lines.

    Args:
        filepath: Path to the file

    Returns:
        List of strings, one per line

    Raises:
        Various exceptions if reading fails
    """
    return read_file_content(filepath, as_lines=True)


def get_prompt(filepath: str) -> str:
    """
    Read a file and return its entire content as a string.

    Args:
        filepath: Path to the file

    Returns:
        File content as a string

    Raises:
        Various exceptions if reading fails
    """
    return read_file_content(filepath, as_lines=False)


def read_json_to_dict(file_path):
    """
    Read a JSON file and return its contents as a Python dictionary.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        dict: The contents of the JSON file as a dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' contains invalid JSON.")
        raise


def save_content(content: str, filename: str, filetype: str = None) -> str:
    """
    Save content to a file with the specified file type extension.

    Args:
        content: Content to save
        filename: Name of the file without extension (e.g., "my_document" or "data/report")
        filetype: File extension to use (without the dot). If None, will
                  default to 'txt'. Supported types: 'txt', 'md', 'html'

    Returns:
        Absolute path to the saved file

    Raises:
        ValueError: If an unsupported file type is provided
        Various IOError exceptions if writing fails
    """
    # Determine file type
    supported_types = ['txt', 'md', 'html']

    if filetype is None:
        filetype = 'txt'  # Default to txt if no filetype specified

    # Validate filetype
    if filetype not in supported_types:
        raise ValueError(f"Unsupported file type: {filetype}. Supported types: {', '.join(supported_types)}")

    # Remove any existing extension from the filename
    base_filename = os.path.splitext(filename)[0]

    # Create filepath with the correct extension
    filepath = f"{base_filename}.{filetype}"

    # Validate the path is writable
    filepath, _ = validate_writable_path(filepath)

    # Write file with proper encoding and error handling
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        logger.debug(f"Content successfully saved to: {filepath}")
        return filepath
    except Exception as e:
        error_msg = f"Failed to write file: {e}"
        logger.error(error_msg)
        raise IOError(error_msg)
