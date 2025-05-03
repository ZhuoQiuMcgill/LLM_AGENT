import os
import logging
from typing import Dict, List, Optional, Any

# Use original imports
import openai
from openai import AsyncOpenAI, OpenAIError, AuthenticationError, RateLimitError, BadRequestError, APIError
from ..llm.base import LLMInterface
from ..exceptions import LLMAPIError, ConfigurationError
from ..utils.helpers import get_api_key

logger = logging.getLogger(__name__)


# Restore inheritance from LLMInterface
class GPTModel(LLMInterface):
    """
    Implementation of LLMInterface for OpenAI's GPT models.

    This class handles authentication, API calls, and response parsing for
    OpenAI's GPT models. It dynamically selects the correct token limit
    parameter ('max_tokens' or 'max_completion_tokens') based on the model.
    """

    # Define prefixes for models known or suspected to use max_completion_tokens
    MODELS_USING_MAX_COMPLETION_TOKENS_PREFIXES = ("o4-", "gpt-4o-")

    def __init__(
            self,
            model: str = "gpt-4o",  # Default to a newer model
            api_key: Optional[str] = None,
            organization: Optional[str] = None,
            # Keep generic name for constructor, map later
            max_output_tokens: int = 1500,
            temperature: float = 0.7,
            default_system_prompt: Optional[str] = None,
    ):
        """
        Initialize the GPT model.

        Args:
            model: The specific GPT model ID to use (e.g., "gpt-4o", "o4-mini-...", "gpt-3.5-turbo").
            api_key: OpenAI API key. If None, will attempt to load using get_api_key.
            organization: OpenAI organization ID. Only required for organizational accounts.
                If None, will attempt to load from environment variable "OPENAI_ORGANIZATION".
            max_output_tokens: Maximum number of tokens to generate in the completion.
                               This value will be mapped to either 'max_tokens' or
                               'max_completion_tokens' depending on the model used.
            temperature: Controls randomness in the output (0.0 to 2.0).
            default_system_prompt: Optional default system prompt to use if none is provided.

        Raises:
            ConfigurationError: If the API key cannot be loaded or client initialization fails.
        """
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.default_system_prompt = default_system_prompt or "You are a helpful assistant."

        # Restore original API key loading
        self.api_key = api_key or get_api_key("OPENAI_API_KEY")
        if not self.api_key:
            # Use original ConfigurationError
            raise ConfigurationError(
                "OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")

        # Load organization if not provided (original logic)
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")

        # Initialize async client (only include organization if it has a value)
        client_kwargs = {"api_key": self.api_key}
        if self.organization:
            client_kwargs["organization"] = self.organization

        try:
            self.client = AsyncOpenAI(**client_kwargs)
            logger.info(f"AsyncOpenAI client initialized successfully for default model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            # Use original ConfigurationError
            raise ConfigurationError(f"Failed to initialize AsyncOpenAI client: {e}") from e

    async def generate_response(
            self,
            prompt: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the GPT model using the chat completions endpoint.

        Args:
            prompt: The text prompt to send to the model.
            history: Optional conversation history as a list of message dictionaries.
            config: Optional configuration parameters for this specific request
                    (e.g., overriding 'model', 'max_output_tokens', 'temperature').

        Returns:
            A string containing the model's primary text response.

        Raises:
            LLMAPIError: If there's an error with the OpenAI API call or response parsing.
        """
        current_model = self.model  # Initialize with default
        try:
            # Get request configuration by merging default and request-specific configs
            request_config = self._prepare_request_config(config)
            current_model = request_config.get("model", self.model)  # Update current_model
            # Use the generic max_output_tokens from config or default
            current_max_output_tokens = request_config.get("max_output_tokens", self.max_output_tokens)
            current_temperature = request_config.get("temperature", self.temperature)

            # Prepare messages for the API call
            messages = self._prepare_messages(prompt, history, request_config)

            logger.debug(f"Sending request to OpenAI API ({current_model}) with {len(messages)} messages.")

            # Create base API parameters
            api_params = {
                "model": current_model,
                "messages": messages,
                "temperature": current_temperature,
            }

            # --- Dynamically set the correct token limit parameter ---
            if current_model.startswith(self.MODELS_USING_MAX_COMPLETION_TOKENS_PREFIXES):
                api_params["max_completion_tokens"] = current_max_output_tokens
                logger.debug(f"Using 'max_completion_tokens': {current_max_output_tokens} for model {current_model}")
            else:
                api_params["max_tokens"] = current_max_output_tokens
                logger.debug(f"Using 'max_tokens': {current_max_output_tokens} for model {current_model}")
            # --- End dynamic parameter setting ---

            # Add other relevant parameters from config if needed
            if 'top_p' in request_config:
                api_params['top_p'] = request_config['top_p']
            if 'frequency_penalty' in request_config:
                api_params['frequency_penalty'] = request_config['frequency_penalty']
            if 'presence_penalty' in request_config:
                api_params['presence_penalty'] = request_config['presence_penalty']
            if 'stop' in request_config:
                api_params['stop'] = request_config['stop']

            # Make the API call using the chat completions endpoint
            response = await self.client.chat.completions.create(**api_params)
            logger.debug(
                f"Received response from OpenAI API. Finish reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")

            # --- Robust Extraction of Text Content ---
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content is not None:
                    response_text = message.content.strip()
                    logger.debug(f"Extracted response text (length: {len(response_text)}).")
                    return response_text
                else:
                    logger.warning("OpenAI response message content is missing or None.")
                    return ""  # Return empty string if content is missing/None
            else:
                logger.warning("OpenAI response does not contain 'choices'.")
                # Use original LLMAPIError
                raise LLMAPIError("OpenAI response structure invalid: No choices found.")

        # --- Refined Error Handling using original custom exceptions ---
        except AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}", exc_info=True)
            error_body = e.body or {}
            error_code = error_body.get("code", "unknown")
            error_message = error_body.get("message", str(e))

            if error_code == "invalid_api_key":
                help_message = "Invalid OpenAI API key provided."
            elif error_code == "mismatched_organization":
                help_message = ("Organization ID mismatch. For personal keys, ensure no organization is set. "
                                "For organizational keys, verify the ID.")
            else:
                help_message = f"Authentication failed: {error_message}"
            # Use original LLMAPIError
            raise LLMAPIError(help_message) from e

        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {e}", exc_info=True)
            error_body = e.body or {}
            error_type = error_body.get("type", "unknown")
            error_message = error_body.get("message", str(e))

            if "insufficient_quota" in error_message or error_type == "insufficient_quota":
                help_message = ("OpenAI API quota exceeded. Check billing details and usage limits at "
                                "https://platform.openai.com/account/billing.")
            else:
                help_message = "OpenAI API rate limit hit. Please wait and retry, or reduce request frequency."
            # Use original LLMAPIError
            raise LLMAPIError(help_message) from e

        except BadRequestError as e:
            logger.error(f"OpenAI bad request error: {e}", exc_info=True)
            error_body = e.body or {}
            error_param = error_body.get("param")
            error_message = error_body.get("message", str(e))
            error_code = error_body.get("code")

            if error_param:
                help_message = f"Invalid parameter '{error_param}': {error_message}"
            else:
                help_message = f"Invalid request to OpenAI API: {error_message}"

            if error_code == "unsupported_parameter" and "'max_tokens'" in str(
                    error_param) and "max_completion_tokens" in error_message:
                help_message += (f" (Model '{current_model}' likely requires 'max_completion_tokens'. "
                                 f"Consider adding its prefix to MODELS_USING_MAX_COMPLETION_TOKENS_PREFIXES if needed.)")
            # Use original LLMAPIError
            raise LLMAPIError(help_message) from e

        except APIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            # Use original LLMAPIError
            raise LLMAPIError(f"OpenAI API returned an error: {e}") from e

        except OpenAIError as e:
            logger.error(f"General OpenAI error: {e}", exc_info=True)
            # Use original LLMAPIError
            raise LLMAPIError(f"An OpenAI error occurred: {e}") from e

        except Exception as e:
            # Catch unexpected errors and wrap in LLMAPIError as per original pattern
            logger.error(f"Unexpected error during OpenAI API call: {e}", exc_info=True)
            raise LLMAPIError(f"An unexpected error occurred: {e}") from e

    def _prepare_request_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare the request configuration by merging default and request-specific configs.

        Args:
            config: Request-specific configuration parameters.

        Returns:
            A dictionary with the merged configuration.
        """
        # Start with class defaults, using the generic token name
        request_config = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,  # Use generic name here
            "temperature": self.temperature,
            "system_prompt": self.default_system_prompt,
        }

        if config:
            # Override with request-specific config
            # Ensure the key used here matches the one expected from the caller ('max_output_tokens')
            for key, value in config.items():
                if value is not None:
                    request_config[key] = value

        return request_config

    def _prepare_messages(
            self,
            prompt: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Dict[str, Any] = None,
    ) -> List[Dict[str, str]]:
        """
        Prepare the messages list for the OpenAI API chat completions call.

        Args:
            prompt: The current text prompt from the user.
            history: Optional conversation history (list of {'role': str, 'content': str}).
            config: The prepared request configuration dictionary containing the system prompt.

        Returns:
            A list of message dictionaries formatted for the OpenAI API.
        """
        messages = []

        system_prompt = config.get("system_prompt")  # Use potentially updated system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
                else:
                    logger.warning(f"Skipping invalid history item: {msg}")

        messages.append({"role": "user", "content": prompt})

        return messages
