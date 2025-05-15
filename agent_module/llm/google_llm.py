"""
Google Gemini implementation of the LLM interface.
"""

import os
import logging
import importlib.util
import mimetypes
from typing import Dict, List, Optional, Any

# Check if google-generativeai package is installed
if importlib.util.find_spec("google.generativeai") is not None:
    import google.generativeai as genai
    from google.generativeai.types import AsyncGenerateContentResponse

    _HAS_GOOGLE_GENERATIVEAI = True
else:
    _HAS_GOOGLE_GENERATIVEAI = False

from ..llm.base import LLMInterface
from ..exceptions import LLMAPIError, ConfigurationError
from ..utils.helpers import get_api_key

logger = logging.getLogger(__name__)


class MissingDependencyError(ImportError):
    """Exception raised when a required dependency is missing."""
    pass


class GeminiModel(LLMInterface):
    """
    Implementation of LLMInterface for Google's Gemini models.

    This class handles authentication, API calls, and response parsing for
    Google's Gemini models.

    Note:
        This implementation requires the 'google-generativeai' package.
        Install it using: pip install google-generativeai
    """

    def __init__(
            self,
            model: str = "gemini-pro",
            api_key: Optional[str] = None,
            temperature: float = 0.7,
            max_output_tokens: int = 65536,
            top_p: float = 0.95,
            top_k: int = 40,
            default_system_prompt: Optional[str] = None,
    ):
        """
        Initialize the Gemini model.

        Args:
            model: The specific Gemini model to use (e.g., "gemini-pro").
            api_key: Google AI API key. If None, will attempt to load from environment.
            temperature: Controls randomness in the output (0.0 to 1.0).
            max_output_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            default_system_prompt: Optional default system prompt to use if none is provided.

        Raises:
            ConfigurationError: If the API key cannot be loaded.
            MissingDependencyError: If the 'google-generativeai' package is not installed.
        """
        if not _HAS_GOOGLE_GENERATIVEAI:
            raise MissingDependencyError(
                "The 'google-generativeai' package is required to use GeminiModel. "
                "Please install it using: pip install google-generativeai"
            )

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.default_system_prompt = default_system_prompt or "You are a helpful assistant."

        # Load API key if not provided
        self.api_key = api_key or get_api_key("GOOGLE_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "Google AI API key not found. Please provide it or set GOOGLE_API_KEY environment variable.")

        # Initialize the API
        genai.configure(api_key=self.api_key)

        logger.debug(f"Initialized GeminiModel with model: {model}")

    async def generate_response(
            self,
            prompt: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the Gemini model.

        Args:
            prompt: The text prompt to send to the model.
            history: Optional conversation history as a list of message dictionaries.
            config: Optional configuration parameters for this specific request.

        Returns:
            A string containing the model's response.

        Raises:
            LLMAPIError: If there's an error with the Google AI API call.
        """
        try:
            # Get request configuration by merging default and request-specific configs
            request_config = self._prepare_request_config(config)

            # Initialize model with configuration
            generation_config = {
                "temperature": request_config.get("temperature", self.temperature),
                "max_output_tokens": request_config.get("max_output_tokens", self.max_output_tokens),
                "top_p": request_config.get("top_p", self.top_p),
                "top_k": request_config.get("top_k", self.top_k),
            }

            # Reinitialize API configuration to ensure fresh state
            genai.configure(api_key=self.api_key)

            # Create a fresh model instance each time to avoid event loop issues
            model = genai.GenerativeModel(
                model_name=request_config.get("model", self.model),
                generation_config=generation_config,
            )

            # Prepare the chat session
            chat = model.start_chat(history=self._format_history_for_gemini(history))

            # Get system prompt (if any)
            system_prompt = request_config.get("system_prompt", self.default_system_prompt)

            # Add system prompt if needed (Gemini handles system prompts differently)
            if system_prompt and not history:
                # For the first message, we can include the system prompt
                prompt = f"{system_prompt}\n\n{prompt}"

            logger.debug(f"Sending request to Google AI API")

            # Make the API call
            response = await chat.send_message_async(prompt)

            # Extract and return the response text
            return response.text

        except Exception as e:
            logger.error(f"Google AI API error: {str(e)}")
            raise LLMAPIError(f"Error calling Google AI API: {str(e)}")

    async def process_with_image(
            self,
            text: str,
            image_path: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the Gemini model using both text and image inputs.

        Args:
            text: The text prompt to send to the model.
            image_path: Path to the image file to include with the prompt.
            history: Optional conversation history as a list of message dictionaries.
            config: Optional configuration parameters for this specific request.

        Returns:
            A string containing the model's response.

        Raises:
            LLMAPIError: If there's an error with the Google AI API call.
            MissingDependencyError: If the required packages are not installed.
        """
        try:
            if not _HAS_GOOGLE_GENERATIVEAI:
                raise MissingDependencyError(
                    "The 'google-generativeai' package is required to use GeminiModel with images. "
                    "Please install it using: pip install google-generativeai"
                )

            # Get request configuration
            request_config = self._prepare_request_config(config)

            # For image processing, use vision-capable model
            current_model = request_config.get("model", self.model)
            if "vision" not in current_model:
                current_model = "gemini-pro-vision"
                logger.info(f"Switched to vision-capable model: {current_model}")

            # Check if image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Initialize generation config
            generation_config = {
                "temperature": request_config.get("temperature", self.temperature),
                "max_output_tokens": request_config.get("max_output_tokens", self.max_output_tokens),
                "top_p": request_config.get("top_p", self.top_p),
                "top_k": request_config.get("top_k", self.top_k),
            }

            # Configure API
            genai.configure(api_key=self.api_key)

            # Create model instance
            model = genai.GenerativeModel(
                model_name=current_model,
                generation_config=generation_config,
            )

            # Get system prompt
            system_prompt = request_config.get("system_prompt", self.default_system_prompt)

            # Create final prompt text
            if system_prompt and not history:
                prompt_text = f"{system_prompt}\n\n{text}"
            else:
                prompt_text = text

            logger.debug(f"Sending request to Google AI API with text and image")

            # Try to load and process image using the appropriate API method
            try:
                # Method 1: Using built-in image loading
                if hasattr(genai.types, "Image") and hasattr(genai.types.Image, "from_file"):
                    image = genai.types.Image.from_file(image_path)
                    response = await model.generate_content_async([prompt_text, image])
                # Method 2: Using MIME type and binary data
                else:
                    mime_type, _ = mimetypes.guess_type(image_path)
                    if not mime_type or not mime_type.startswith('image/'):
                        mime_type = 'image/jpeg'  # Default

                    with open(image_path, "rb") as img_file:
                        image_data = img_file.read()

                    response = await model.generate_content_async([
                        {"text": prompt_text},
                        {"inline_data": {"mime_type": mime_type, "data": image_data}}
                    ])
            except Exception as e:
                # Fallback: Try using PIL if available
                try:
                    from PIL import Image
                    image = Image.open(image_path)
                    response = await model.generate_content_async([prompt_text, image])
                except ImportError:
                    logger.error("Failed to process image with Gemini API and PIL is not available")
                    raise LLMAPIError(f"Failed to process image with Gemini API: {e}")

            # Extract response text
            try:
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'parts') and response.parts:
                    return response.parts[0].text
                else:
                    logger.warning("Could not extract text from Google AI response")
                    return str(response)
            except Exception as e:
                logger.error(f"Failed to extract text from response: {e}")
                return ""

        except FileNotFoundError as e:
            logger.error(f"Image file error: {e}", exc_info=True)
            raise LLMAPIError(f"Image file error: {e}")
        except Exception as e:
            logger.error(f"Error processing image with Google AI: {e}", exc_info=True)
            raise LLMAPIError(f"Error processing image with Google AI: {e}")

    async def process_with_image_bin(
            self,
            text: str,
            image_data: bytes,
            mime_type: Optional[str] = None,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the Gemini model using both text and binary image data.

        This method is useful for processing images directly from memory without saving to disk,
        such as handling file uploads in web applications.

        Args:
            text: The text prompt to send to the model.
            image_data: Binary image data.
            mime_type: Optional MIME type of the image (e.g., "image/jpeg", "image/png").
                      If None, will attempt to detect or use a default.
            history: Optional conversation history as a list of message dictionaries.
            config: Optional configuration parameters for this specific request.

        Returns:
            A string containing the model's response.

        Raises:
            LLMAPIError: If there's an error with the Google AI API call.
            MissingDependencyError: If the required packages are not installed.
        """
        try:
            if not _HAS_GOOGLE_GENERATIVEAI:
                raise MissingDependencyError(
                    "The 'google-generativeai' package is required to use GeminiModel with images. "
                    "Please install it using: pip install google-generativeai"
                )

            # Validate image data
            if not image_data:
                raise ValueError("Image data is empty")

            # Get request configuration
            request_config = self._prepare_request_config(config)

            # For image processing, use vision-capable model
            current_model = request_config.get("model", self.model)
            if "vision" not in current_model:
                current_model = "gemini-pro-vision"
                logger.info(f"Switched to vision-capable model: {current_model}")

            # Use default MIME type if not provided
            if not mime_type:
                # Try to detect MIME type
                try:
                    import magic
                    mime_type = magic.Magic(mime=True).from_buffer(image_data)
                except ImportError:
                    # If python-magic is not available, use default
                    mime_type = 'image/jpeg'
                    logger.warning("Could not detect MIME type, using default: image/jpeg")

                # Ensure it's an image MIME type
                if not mime_type.startswith('image/'):
                    mime_type = 'image/jpeg'
                    logger.warning(f"Non-image MIME type detected, using default: {mime_type}")

            # Initialize generation config
            generation_config = {
                "temperature": request_config.get("temperature", self.temperature),
                "max_output_tokens": request_config.get("max_output_tokens", self.max_output_tokens),
                "top_p": request_config.get("top_p", self.top_p),
                "top_k": request_config.get("top_k", self.top_k),
            }

            # Configure API
            genai.configure(api_key=self.api_key)

            # Create model instance
            model = genai.GenerativeModel(
                model_name=current_model,
                generation_config=generation_config,
            )

            # Get system prompt
            system_prompt = request_config.get("system_prompt", self.default_system_prompt)

            # Create final prompt text
            if system_prompt and not history:
                prompt_text = f"{system_prompt}\n\n{text}"
            else:
                prompt_text = text

            logger.debug(f"Sending request to Google AI API with text and binary image data")

            # Try different methods to process the binary image data
            try:
                # Method 1: Using built-in image handling for binary data if available
                # (available in newer Google AI SDK versions)
                if hasattr(genai.types, "Image") and hasattr(genai.types.Image, "from_bytes"):
                    image = genai.types.Image.from_bytes(image_data)
                    response = await model.generate_content_async([prompt_text, image])
                # Method 2: Using inline data approach
                else:
                    response = await model.generate_content_async([
                        {"text": prompt_text},
                        {"inline_data": {"mime_type": mime_type, "data": image_data}}
                    ])
            except Exception as e:
                # Fallback: Try using PIL if available
                try:
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image_data))
                    response = await model.generate_content_async([prompt_text, image])
                except ImportError:
                    logger.error("Failed to process binary image with Gemini API and PIL is not available")
                    raise LLMAPIError(f"Failed to process binary image with Gemini API: {e}")

            # Extract response text
            try:
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'parts') and response.parts:
                    return response.parts[0].text
                else:
                    logger.warning("Could not extract text from Google AI response")
                    return str(response)
            except Exception as e:
                logger.error(f"Failed to extract text from response: {e}")
                return ""

        except ValueError as e:
            logger.error(f"Image data error: {e}", exc_info=True)
            raise LLMAPIError(f"Image data error: {e}")
        except Exception as e:
            logger.error(f"Error processing binary image with Google AI: {e}", exc_info=True)
            raise LLMAPIError(f"Error processing binary image with Google AI: {e}")

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
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "system_prompt": self.default_system_prompt,
        }

        if config:
            request_config.update(config)

        return request_config

    def _format_history_for_gemini(
            self,
            history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Format the conversation history for the Gemini API.

        Args:
            history: Conversation history in the standard format.

        Returns:
            A list of history entries formatted for the Gemini API.
        """
        if not history:
            return []

        # Convert standard history format to Gemini format
        # Standard format typically uses "role" and "content" keys
        # Gemini uses "role" and "parts" keys
        gemini_history = []

        for message in history:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Map OpenAI roles to Gemini roles
            if role == "system":
                # Skip system messages as they're handled differently
                continue
            elif role == "assistant":
                gemini_role = "model"
            else:
                gemini_role = "user"

            gemini_history.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

        return gemini_history

    def model_name(self):
        return self.model
