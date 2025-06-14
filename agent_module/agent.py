"""
Core Agent module that handles the interaction with LLMs.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple

# Use relative imports for better package structure
from .llm.base import LLMInterface
from .exceptions import AgentError

# Forward reference for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .history.memory import HistoryManager

logger = logging.getLogger(__name__)


class Agent:
    """
    Core Agent class that manages interactions with a chosen LLM.

    The Agent provides a unified interface for sending prompts to different LLMs,
    while maintaining conversation history and handling the necessary configurations.
    """

    def __init__(
            self,
            llm_interface: LLMInterface,
            history_manager: Optional['HistoryManager'] = None,
            name: str = "Assistant",
            system_prompt: Optional[str] = None,
            image_system_prompt: Optional[str] = None,
            vision_model_override: Optional[str] = None,
            max_image_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the Agent.

        Args:
            llm_interface: An implementation of LLMInterface to handle LLM interactions.
            history_manager: A HistoryManager instance to handle conversation history.
            name: A name for the Agent (used mainly for logging and identification).
            system_prompt: An optional system prompt to override the default one in the LLM.
            image_system_prompt: An optional system prompt specifically for image processing.
                                 If None, will use system_prompt for image requests as well.
            vision_model_override: Optional model name to use specifically for vision tasks.
                                   If provided, this model will be used instead of the default
                                   when processing images.
            max_image_size: Optional tuple (width, height) to resize images before processing.
                           If None, images will be processed at their original size.
        """
        self.llm = llm_interface

        # Use provided history manager or import and create a default one
        if history_manager is None:
            from .history.memory import InMemoryHistoryManager
            self.history_manager = InMemoryHistoryManager()
        else:
            self.history_manager = history_manager

        self.name = name
        self.system_prompt = system_prompt

        # Image-specific configuration
        self.image_system_prompt = image_system_prompt or system_prompt
        self.vision_model_override = vision_model_override
        self.max_image_size = max_image_size

        # Check if PIL is available for image resizing
        self._has_pil = False
        try:
            import PIL
            self._has_pil = True
            logger.debug("PIL is available for image processing")
        except ImportError:
            logger.debug("PIL is not available. Image resizing will be skipped.")

        logger.debug(f"Initialized Agent: {name}")

    async def process(self, input_text: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Process an input message through the Agent and generate a response.

        This method:
        1. Gets the conversation history from the history manager
        2. Combines the system prompt with the history and current input
        3. Sends this to the LLM for processing
        4. Updates the history with the new input and response
        5. Returns the LLM's response

        Args:
            input_text: The input text/prompt from the user.
            config: Optional configuration overrides for this specific request.

        Returns:
            The text response generated by the LLM.

        Raises:
            AgentError: If there's an error in processing the request.
        """
        try:
            # Get conversation history
            history = self.history_manager.get_history()

            # Prepare configuration
            request_config = self._prepare_request_config(config)

            # Generate response using the LLM
            response = await self.llm.generate_response(
                prompt=input_text,
                history=history,
                config=request_config
            )

            # Update history with the new input and response
            self.history_manager.add_user_message(input_text)
            self.history_manager.add_assistant_message(response)

            logger.debug(f"Agent {self.name} processed input and generated response")

            return response

        except Exception as e:
            logger.error(f"Error processing input with Agent {self.name}: {str(e)}")
            raise AgentError(f"Error processing input: {str(e)}") from e

    async def process_with_image(self, input_text: str, image_path: str,
                                 config: Optional[Dict[str, Any]] = None) -> str:
        """
        Process an input message and image through the Agent and generate a response.

        This method:
        1. Gets the conversation history from the history manager
        2. Validates the image file
        3. Combines the system prompt with the history, current text input, and image
        4. Sends this to the LLM for processing
        5. Updates the history with the new input and response
        6. Returns the LLM's response

        Args:
            input_text: The input text/prompt from the user.
            image_path: Path to the image file to include with the prompt.
            config: Optional configuration overrides for this specific request.

        Returns:
            The text response generated by the LLM.

        Raises:
            AgentError: If there's an error in processing the request.
            FileNotFoundError: If the image file doesn't exist.
        """
        try:
            # Validate image file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Get conversation history
            history = self.history_manager.get_history()

            # Prepare configuration for image processing
            request_config = self._prepare_image_request_config(config)

            # Resize image if needed and supported
            processed_image_path = image_path
            if self.max_image_size and self._has_pil:
                processed_image_path = self._resize_image_file(image_path)

            # Generate response using the LLM with image
            response = await self.llm.process_with_image(
                text=input_text,
                image_path=processed_image_path,
                history=history,
                config=request_config
            )

            # Update history with the new input and response
            # Note: We store only the text prompt in history, not the image reference
            self.history_manager.add_user_message(f"{input_text} [Image included]")
            self.history_manager.add_assistant_message(response)

            logger.debug(f"Agent {self.name} processed input with image and generated response")

            # Clean up temporary processed image if it's different from original
            if processed_image_path != image_path and os.path.exists(processed_image_path):
                try:
                    os.remove(processed_image_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary processed image: {e}")

            return response

        except FileNotFoundError as e:
            logger.error(f"Image file error: {str(e)}")
            raise AgentError(f"Image file error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error processing input with image with Agent {self.name}: {str(e)}")
            raise AgentError(f"Error processing input with image: {str(e)}") from e

    async def process_with_image_bin(
            self,
            input_text: str,
            image_data: bytes,
            mime_type: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process an input message and binary image data through the Agent and generate a response.

        This method is useful for processing images directly from memory without saving to disk,
        such as handling file uploads in web applications.

        Args:
            input_text: The input text/prompt from the user.
            image_data: Binary image data.
            mime_type: Optional MIME type of the image (e.g., "image/jpeg", "image/png").
                       If None, will attempt to detect or use a default.
            config: Optional configuration overrides for this specific request.

        Returns:
            The text response generated by the LLM.

        Raises:
            AgentError: If there's an error in processing the request.
            ValueError: If the image data is invalid.
        """
        try:
            # Validate image data
            if not image_data:
                raise ValueError("Image data is empty")

            # Get conversation history
            history = self.history_manager.get_history()

            # Prepare configuration for image processing
            request_config = self._prepare_image_request_config(config)

            # Resize image if needed and supported
            processed_image_data = image_data
            if self.max_image_size and self._has_pil:
                processed_image_data = self._resize_image_data(image_data)

            # Generate response using the LLM with binary image data
            response = await self.llm.process_with_image_bin(
                text=input_text,
                image_data=processed_image_data,
                mime_type=mime_type,
                history=history,
                config=request_config
            )

            # Update history with the new input and response
            self.history_manager.add_user_message(f"{input_text} [Image included]")
            self.history_manager.add_assistant_message(response)

            logger.debug(f"Agent {self.name} processed input with binary image data and generated response")

            return response

        except ValueError as e:
            logger.error(f"Image data error: {str(e)}")
            raise AgentError(f"Image data error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error processing input with binary image data with Agent {self.name}: {str(e)}")
            raise AgentError(f"Error processing input with binary image data: {str(e)}") from e

    async def run(self, input_text: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Alias for process() method to provide a more intuitive API.

        Args:
            input_text: The input text/prompt from the user.
            config: Optional configuration overrides for this specific request.

        Returns:
            The text response generated by the LLM.
        """
        return await self.process(input_text, config)

    async def connection_test(self, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Test the connection to the LLM by sending a simple test prompt.

        This method sends a test prompt to the LLM to verify that the connection
        is working properly. The test prompt includes the model name for verification.
        The conversation history is not updated with this test message.

        Args:
            config: Optional configuration overrides for this specific request.

        Returns:
            The text response generated by the LLM.

        Raises:
            AgentError: If there's an error in processing the request.
        """
        try:
            # Get the model name from the LLM implementation
            model_name = getattr(self.llm, "model", "unknown")

            # Create a test prompt
            test_prompt = f"This is a test message for {model_name}. If you receive and understand this message, please respond with: '{model_name} connection successful'"

            # Create a test-specific config that doesn't include the system prompt
            test_config = {} if config is None else config.copy()
            # Explicitly set system_prompt to None for connection tests
            test_config["system_prompt"] = None

            # Use an empty history list for connection tests
            # This prevents any previous conversation context from affecting the test
            empty_history = []

            # Generate response using the LLM
            response = await self.llm.generate_response(
                prompt=test_prompt,
                history=empty_history,
                config=test_config
            )

            logger.debug(f"Agent {self.name} performed connection test with model {model_name}")

            return response

        except Exception as e:
            logger.error(f"Error testing connection with Agent {self.name}: {str(e)}")
            raise AgentError(f"Error testing connection: {str(e)}") from e

    def _prepare_request_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare the request configuration.

        Args:
            config: Optional configuration overrides.

        Returns:
            A dictionary with the configuration for the LLM request.
        """
        request_config = {}

        # Add system prompt if available
        if self.system_prompt:
            request_config["system_prompt"] = self.system_prompt

        # Add any additional config parameters
        if config:
            request_config.update(config)

        return request_config

    def _prepare_image_request_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare the request configuration specifically for image processing.

        Args:
            config: Optional configuration overrides.

        Returns:
            A dictionary with the configuration for the LLM image processing request.
        """
        request_config = {}

        # Add image-specific system prompt if available
        if self.image_system_prompt:
            request_config["system_prompt"] = self.image_system_prompt

        # Override model if vision model is specified
        if self.vision_model_override:
            request_config["model"] = self.vision_model_override

        # Add any additional config parameters
        if config:
            request_config.update(config)

        return request_config

    def _resize_image_file(self, image_path: str) -> str:
        """
        Resize an image file to the maximum size specified in self.max_image_size.

        Args:
            image_path: Path to the image file to resize.

        Returns:
            Path to the resized image (which may be a temporary file).
        """
        if not self.max_image_size or not self._has_pil:
            return image_path

        try:
            from PIL import Image
            import io
            import tempfile

            # Open the image
            with Image.open(image_path) as img:
                # Check if resize is needed
                if img.width <= self.max_image_size[0] and img.height <= self.max_image_size[1]:
                    return image_path

                # Calculate new dimensions while preserving aspect ratio
                width, height = img.size
                ratio = min(self.max_image_size[0] / width, self.max_image_size[1] / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)

                # Resize the image
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)

                # Save to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                resized_img.save(temp_file.name, format='JPEG', quality=90)

                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

                return temp_file.name

        except Exception as e:
            logger.warning(f"Failed to resize image: {e}")
            return image_path

    def _resize_image_data(self, image_data: bytes) -> bytes:
        """
        Resize binary image data to the maximum size specified in self.max_image_size.

        Args:
            image_data: Binary image data to resize.

        Returns:
            Resized binary image data.
        """
        if not self.max_image_size or not self._has_pil:
            return image_data

        try:
            from PIL import Image
            import io

            # Open the image from binary data
            with Image.open(io.BytesIO(image_data)) as img:
                # Check if resize is needed
                if img.width <= self.max_image_size[0] and img.height <= self.max_image_size[1]:
                    return image_data

                # Calculate new dimensions while preserving aspect ratio
                width, height = img.size
                ratio = min(self.max_image_size[0] / width, self.max_image_size[1] / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)

                # Resize the image
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)

                # Convert back to binary data
                buffer = io.BytesIO()
                if img.format:
                    resized_img.save(buffer, format=img.format)
                else:
                    resized_img.save(buffer, format='JPEG')

                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

                return buffer.getvalue()

        except Exception as e:
            logger.warning(f"Failed to resize image data: {e}")
            return image_data

    def reset_history(self) -> None:
        """
        Reset the conversation history.
        """
        self.history_manager.clear()
        logger.debug(f"History reset for Agent {self.name}")

    def model_name(self):
        return self.llm.model_name()
