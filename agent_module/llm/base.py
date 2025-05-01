"""
Base module defining the abstract interface for LLM integrations.
This module provides the core LLMInterface that all LLM implementations must follow.
"""

import abc
from typing import Dict, List, Optional, Union, Any


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
