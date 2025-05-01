"""
OpenAI GPT implementation of the LLM interface.
"""

import os
import logging
from typing import Dict, List, Optional, Any

import openai
from openai import AsyncOpenAI

from ..llm.base import LLMInterface
from ..exceptions import LLMAPIError, ConfigurationError
from ..utils.helpers import get_api_key

logger = logging.getLogger(__name__)


class GPTModel(LLMInterface):
    """
    Implementation of LLMInterface for OpenAI's GPT models.

    This class handles authentication, API calls, and response parsing for
    OpenAI's GPT models (such as GPT-4, GPT-3.5-turbo).
    """

    def __init__(
            self,
            model: str = "gpt-4",
            api_key: Optional[str] = None,
            organization: Optional[str] = None,
            max_tokens: int = 1000,
            temperature: float = 0.7,
            default_system_prompt: Optional[str] = None,
    ):
        """
        Initialize the GPT model.

        Args:
            model: The specific GPT model to use (e.g., "gpt-4", "gpt-3.5-turbo").
            api_key: OpenAI API key. If None, will attempt to load from environment.
            organization: OpenAI organization ID. Only required for organizational accounts.
                For personal accounts, leave this as None.
                If None, will attempt to load from environment variable "OPENAI_ORGANIZATION".
            max_tokens: Maximum number of tokens to generate.
            temperature: Controls randomness in the output (0.0 to 1.0).
            default_system_prompt: Optional default system prompt to use if none is provided.

        Raises:
            ConfigurationError: If the API key cannot be loaded.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.default_system_prompt = default_system_prompt or "You are a helpful assistant."

        # Load API key if not provided
        self.api_key = api_key or get_api_key("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")

        # Load organization if not provided
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")

        # Initialize client (only include organization if it has a value)
        client_kwargs = {"api_key": self.api_key}
        if self.organization:
            client_kwargs["organization"] = self.organization

        self.client = AsyncOpenAI(**client_kwargs)

        logger.debug(f"Initialized GPTModel with model: {model}")

    async def generate_response(
            self,
            prompt: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the GPT model.

        Args:
            prompt: The text prompt to send to the model.
            history: Optional conversation history as a list of message dictionaries.
            config: Optional configuration parameters for this specific request.

        Returns:
            A string containing the model's response.

        Raises:
            LLMAPIError: If there's an error with the OpenAI API call.
        """
        try:
            # Get request configuration by merging default and request-specific configs
            request_config = self._prepare_request_config(config)

            # Prepare messages for the API call
            messages = self._prepare_messages(prompt, history, request_config)

            logger.debug(f"Sending request to OpenAI API with {len(messages)} messages")

            # Make the API call
            response = await self.client.chat.completions.create(
                model=request_config.get("model", self.model),
                messages=messages,
                max_tokens=request_config.get("max_tokens", self.max_tokens),
                temperature=request_config.get("temperature", self.temperature),
            )

            # Extract and return the response text
            response_text = response.choices[0].message.content
            return response_text

        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {str(e)}")
            error_message = str(e)

            # Provide more helpful error messages for common authentication issues
            if "mismatched_organization" in error_message:
                help_message = (
                    "The OpenAI-Organization header doesn't match your API key. "
                    "For personal accounts, remove the organization parameter entirely. "
                    "For organization accounts, ensure the organization ID is correct."
                )
                raise LLMAPIError(f"OpenAI authentication error: {help_message}")
            elif "invalid_api_key" in error_message:
                raise LLMAPIError("Invalid OpenAI API key. Please check your API key and try again.")
            else:
                raise LLMAPIError(f"OpenAI authentication error: {error_message}")

        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {str(e)}")
            raise LLMAPIError("OpenAI rate limit exceeded. Please try again later or reduce request frequency.")

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMAPIError(f"Error calling OpenAI API: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error when calling OpenAI API: {str(e)}")
            raise LLMAPIError(f"Unexpected error: {str(e)}")

    def _prepare_request_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare the request configuration by merging default and request-specific configs.

        Args:
            config: Request-specific configuration parameters.

        Returns:
            A dictionary with the merged configuration.
        """
        request_config = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system_prompt": self.default_system_prompt,
        }

        if config:
            request_config.update(config)

        return request_config

    def _prepare_messages(
            self,
            prompt: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Dict[str, Any] = None,
    ) -> List[Dict[str, str]]:
        """
        Prepare the messages for the OpenAI API call.

        Args:
            prompt: The text prompt to send.
            history: Optional conversation history.
            config: Configuration parameters including system prompt.

        Returns:
            A list of message dictionaries formatted for the OpenAI API.
        """
        messages = []

        # Add system message if available
        system_prompt = config.get("system_prompt", self.default_system_prompt)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history if available
        if history:
            messages.extend(history)

        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})

        return messages
