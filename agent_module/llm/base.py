"""
Base module defining the abstract interface for LLM integrations.
This module provides the core LLMInterface that all LLM implementations must follow.
"""

import abc
from typing import Dict, List, Optional, Union, Any, Tuple


class LLMInterface(abc.ABC):
    """
    Abstract base class defining the interface for all LLM implementations.

    This interface ensures all LLM implementations provide consistent methods
    for generating responses regardless of the underlying LLM API being used.
    """

    @abc.abstractmethod
    async def generate_response(
            self,
            prompt: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the LLM based on the provided prompt and history.

        Args:
            prompt: The text prompt to send to the LLM.
            history: Optional conversation history for context.
            config: Optional configuration parameters for the specific request.

        Returns:
            A string containing the LLM's response.

        Raises:
            LLMAPIError: If there's an error with the LLM API call.
            ConfigurationError: If there's a configuration issue.
        """
        pass

    @abc.abstractmethod
    async def process_with_image(
            self,
            text: str,
            image_path: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the LLM using both text and image inputs.

        Args:
            text: The text prompt to send to the LLM.
            image_path: Path to the image file to include with the prompt.
            history: Optional conversation history for context.
            config: Optional configuration parameters for the specific request.

        Returns:
            A string containing the LLM's response.

        Raises:
            LLMAPIError: If there's an error with the LLM API call.
            ConfigurationError: If there's a configuration issue.
        """
        pass

    @abc.abstractmethod
    async def process_with_image_bin(
            self,
            text: str,
            image_data: bytes,
            mime_type: Optional[str] = None,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the LLM using both text and binary image data.

        This method is useful for processing images directly from memory without saving to disk,
        such as handling file uploads in web applications.

        Args:
            text: The text prompt to send to the LLM.
            image_data: Binary image data.
            mime_type: Optional MIME type of the image (e.g., "image/jpeg", "image/png").
                       If None, will attempt to detect or use a default.
            history: Optional conversation history for context.
            config: Optional configuration parameters for the specific request.

        Returns:
            A string containing the LLM's response.

        Raises:
            LLMAPIError: If there's an error with the LLM API call.
            ConfigurationError: If there's a configuration issue.
        """
        pass

    @abc.abstractmethod
    def model_name(self):
        pass
