import os
import logging
import base64
from typing import Dict, List, Optional, Any

# Use original imports
import openai
from openai import (
    AsyncOpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
    BadRequestError,
    APIError,
)

from ..llm.base import LLMInterface
from ..exceptions import LLMAPIError, ConfigurationError
from ..utils.helpers import get_api_key

logger = logging.getLogger(__name__)


class GPTModel(LLMInterface):
    """Implementation of :class:`LLMInterface` for OpenAI GPT models.

    This class wraps OpenAI's *chat completions* endpoint, handling:
    * **Authentication** and async client creation
    * **Dynamic token‑limit parameter selection** (``max_tokens`` vs ``max_completion_tokens``)
    * **JSON‑mode** forcing for older models (``response_format={"type":"json_object"}``)
    * **Structured‑Output mode** (``response_format={"type":"json_schema", ...}``)
    * **Vision** requests (text + image / image‑binary)
    * Robust error handling mapped to the app's own exception hierarchy.
    """

    # Model‑prefixes that expect ``max_completion_tokens`` instead of ``max_tokens``
    MODELS_USING_MAX_COMPLETION_TOKENS_PREFIXES = ("o4-", "gpt-4o-", "o3-")

    # Models known to support **JSON mode** (``response_format.type == json_object``)
    JSON_SUPPORTED_MODELS = (
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
    )

    # Models that support **Structured Outputs** (json_schema / tool‑calling strict)
    # – At the time of writing the same set as JSON‑mode, but kept separate for future‑proofing.
    STRUCTURED_OUTPUT_SUPPORTED_MODELS = JSON_SUPPORTED_MODELS + ("o4-",)

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        max_output_tokens: int = 8192,
        temperature: float = 1.0,
        default_system_prompt: Optional[str] = None,
        force_json_response: bool = False,
        structured_output_schema: Optional[Dict[str, Any]] = None,
    ):
        """Create a new :class:`GPTModel` instance.

        Args
        ----
        model:  OpenAI model name (e.g. ``"gpt-4o"``, ``"o4-mini-2025-04-16"``).
        api_key:  Overrides ``OPENAI_API_KEY`` env‑var.
        organization:  Optional OpenAI *organization* id.
        max_output_tokens:  Default completion limit.
        temperature:  Sampling temperature ``[0.0 – 2.0]``.
        default_system_prompt:  Used when caller supplies no system prompt.
        force_json_response:  Enable legacy JSON‑mode. Ignored when *structured_output_schema* is provided.
        structured_output_schema:  JSON‑Schema dict that triggers **Structured Output** mode when not *None*.
        """

        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.default_system_prompt = (
            default_system_prompt or "You are a helpful assistant."
        )
        self.force_json_response = force_json_response
        self.structured_output_schema = structured_output_schema

        # --- Auth & client --------------------------------------------------
        self.api_key = api_key or get_api_key("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key not found. Provide it explicitly or set the OPENAI_API_KEY env‑var."
            )

        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")

        client_kwargs = {"api_key": self.api_key}
        if self.organization:
            client_kwargs["organization"] = self.organization

        try:
            self.client = AsyncOpenAI(**client_kwargs)
            logger.info("AsyncOpenAI client initialised (model=%s)", model)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to initialise AsyncOpenAI client")
            raise ConfigurationError(f"Failed to initialise OpenAI client: {exc}") from exc

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def model_name(self) -> str:  # noqa: D401 – keeps original external api
        return self.model

    # ---------------------------------------------------------------------
    # Async chat completions (text‑only)
    # ---------------------------------------------------------------------
    async def generate_response(
            self,
            prompt: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the assistant reply for *prompt* (optionally with *history*)."""

        request_config = self._prepare_request_config(config)
        current_model = request_config.get("model", self.model)
        current_max_tokens = request_config.get("max_output_tokens", self.max_output_tokens)
        current_temperature = request_config.get("temperature", self.temperature)
        force_json = request_config.get("force_json_response", self.force_json_response)
        schema = request_config.get("structured_output_schema", self.structured_output_schema)

        messages = self._prepare_messages(
            prompt,
            history,
            request_config,
            force_json=force_json,
        )

        # --- Build API params --------------------------------------------
        api_params: Dict[str, Any] = {
            "model": current_model,
            "messages": messages,
            "temperature": current_temperature,
        }

        # ----------------- Response‑format handling ----------------------
        if schema is not None and self._supports_structured_output(current_model):
            api_params["response_format"] = {
                "type": "json_schema",
                "json_schema": schema,
            }
            logger.debug("Structured‑output mode enabled for model %s", current_model)
        elif force_json and self._supports_json_response(current_model):
            api_params["response_format"] = {"type": "json_object"}
            logger.debug("Legacy JSON‑mode enabled for model %s", current_model)
        elif force_json:
            logger.warning(
                "Model %s does not support JSON mode – continuing without enforced JSON.",
                current_model,
            )

        # ----------------- Token‑limit parameter -------------------------
        if current_model.startswith(self.MODELS_USING_MAX_COMPLETION_TOKENS_PREFIXES):
            api_params["max_completion_tokens"] = current_max_tokens
        else:
            api_params["max_tokens"] = current_max_tokens

        # ----------------- Optional params ------------------------------
        for optional in ("top_p", "frequency_penalty", "presence_penalty", "stop"):
            if optional in request_config:
                api_params[optional] = request_config[optional]

        try:
            response = await self.client.chat.completions.create(**api_params)
            return self._extract_text_or_raise(response)
        except Exception as exc:
            self._handle_openai_exception(exc, current_model)

    # ---------------------------------------------------------------------
    # Vision – text + image‑file path
    # ---------------------------------------------------------------------
    async def process_with_image(
            self,
            text: str,
            image_path: str,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not os.path.exists(image_path):
            raise LLMAPIError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as fh:
            image_bytes = fh.read()

        return await self.process_with_image_bin(
            text=text,
            image_data=image_bytes,
            mime_type="image/jpeg",  # assume jpg – override by passing mime in *_bin*
            history=history,
            config=config,
        )

    # ---------------------------------------------------------------------
    # Vision – text + image bytes
    # ---------------------------------------------------------------------
    async def process_with_image_bin(
            self,
            text: str,
            image_data: bytes,
            mime_type: Optional[str] = None,
            history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not image_data:
            raise LLMAPIError("Image data is empty")

        request_config = self._prepare_request_config(config)
        current_model = request_config.get("model", self.model)
        current_temperature = request_config.get("temperature", self.temperature)
        force_json = request_config.get("force_json_response", self.force_json_response)
        schema = request_config.get("structured_output_schema", self.structured_output_schema)

        # Vision‑capable model safety hint (does not hard‑fail)
        if "vision" not in current_model and current_model != "gpt-4o":
            logger.warning(
                "Model %s may not support vision – consider 'gpt-4o' or a vision‑preview model.",
                current_model,
            )

        mime_type = mime_type or "image/jpeg"
        image_b64 = base64.b64encode(image_data).decode()

        # Messages ---------------------------------------------------------
        messages = []
        system_prompt = request_config.get("system_prompt")
        if system_prompt:
            if schema is None and force_json and self._supports_json_response(current_model):
                system_prompt = self._add_json_instruction(system_prompt)
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)

        user_text = text
        if schema is None and force_json and self._supports_json_response(current_model) and not system_prompt:
            user_text = self._add_json_instruction_to_text(text)

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        )

        # API params -------------------------------------------------------
        api_params: Dict[str, Any] = {
            "model": current_model,
            "messages": messages,
            "temperature": current_temperature,
        }

        if schema is not None and self._supports_structured_output(current_model):
            api_params["response_format"] = {
                "type": "json_schema",
                "json_schema": schema,
            }
        elif force_json and self._supports_json_response(current_model):
            api_params["response_format"] = {"type": "json_object"}

        if current_model.startswith(self.MODELS_USING_MAX_COMPLETION_TOKENS_PREFIXES):
            api_params["max_completion_tokens"] = request_config.get(
                "max_output_tokens", self.max_output_tokens
            )
        else:
            api_params["max_tokens"] = request_config.get(
                "max_output_tokens", self.max_output_tokens
            )

        for optional in ("top_p", "frequency_penalty", "presence_penalty", "stop"):
            if optional in request_config:
                api_params[optional] = request_config[optional]

        try:
            response = await self.client.chat.completions.create(**api_params)
            return self._extract_text_or_raise(response)
        except Exception as exc:
            self._handle_openai_exception(exc, current_model)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _prepare_request_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge instance defaults with per‑request *config*."""
        request_config: Dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "system_prompt": self.default_system_prompt,
            "force_json_response": self.force_json_response,
            "structured_output_schema": self.structured_output_schema,
        }
        if config:
            # Shallow update – callers should not nest
            request_config.update({k: v for k, v in config.items() if v is not None})
        return request_config

    def _prepare_messages(
            self,
            prompt: str,
            history: Optional[List[Dict[str, str]]],
            config: Dict[str, Any],
            *,
            force_json: bool,
    ) -> List[Dict[str, Any]]:
        """Return the *messages* array for the OpenAI request."""
        messages: List[Dict[str, Any]] = []
        system_prompt = config.get("system_prompt")
        schema = config.get("structured_output_schema")
        model = config.get("model", self.model)

        if system_prompt:
            if schema is None and force_json and self._supports_json_response(model):
                system_prompt = self._add_json_instruction(system_prompt)
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)

        user_prompt = prompt
        if schema is None and force_json and self._supports_json_response(model) and not system_prompt:
            user_prompt = self._add_json_instruction_to_text(prompt)

        messages.append({"role": "user", "content": user_prompt})
        return messages

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _supports_json_response(model: str) -> bool:
        return any(s in model for s in GPTModel.JSON_SUPPORTED_MODELS)

    @staticmethod
    def _supports_structured_output(model: str) -> bool:
        return any(s in model for s in GPTModel.STRUCTURED_OUTPUT_SUPPORTED_MODELS)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_json_instruction(prompt: str) -> str:
        if "json" not in prompt.lower():
            return prompt.rstrip() + " Always respond with valid JSON format."
        return prompt

    @staticmethod
    def _add_json_instruction_to_text(text: str) -> str:
        if "json" not in text.lower():
            return text.rstrip() + " Please respond in valid JSON format."
        return text

    # ------------------------------------------------------------------
    # Response + error helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text_or_raise(response: Any) -> str:
        if response.choices and response.choices[0].message and response.choices[0].message.content is not None:
            return response.choices[0].message.content.strip()
        raise LLMAPIError("OpenAI response missing content")

    def _handle_openai_exception(self, exc: Exception, model: str):  # noqa: C901 – long but flat
        """Translate OpenAI errors to domain‑specific :class:`LLMAPIError`."""
        if isinstance(exc, AuthenticationError):
            error_body = getattr(exc, "body", {}) or {}
            code = error_body.get("code", "unknown")
            if code == "invalid_api_key":
                raise LLMAPIError("Invalid OpenAI API key.") from exc
            if code == "mismatched_organization":
                raise LLMAPIError("Organization ID mismatch between key and client.") from exc
            raise LLMAPIError(f"Authentication failed: {exc}") from exc

        if isinstance(exc, RateLimitError):
            body = getattr(exc, "body", {}) or {}
            if body.get("type") == "insufficient_quota" or "insufficient_quota" in str(exc):
                raise LLMAPIError(
                    "OpenAI quota exceeded – check https://platform.openai.com/account/billing."
                ) from exc
            raise LLMAPIError("OpenAI rate limit hit – retry later.") from exc

        if isinstance(exc, BadRequestError):
            body = getattr(exc, "body", {}) or {}
            param = body.get("param")
            msg = body.get("message", str(exc))
            if (
                    body.get("code") == "unsupported_parameter"
                    and param == "max_tokens"
                    and "max_completion_tokens" in msg
            ):
                msg += (
                    f" (Model '{model}' expects 'max_completion_tokens' – add its prefix to"
                    " MODELS_USING_MAX_COMPLETION_TOKENS_PREFIXES if needed.)"
                )
            raise LLMAPIError(f"Bad request: {msg}") from exc

        if isinstance(exc, APIError):
            raise LLMAPIError(f"OpenAI API error: {exc}") from exc

        if isinstance(exc, OpenAIError):
            raise LLMAPIError(f"OpenAI error: {exc}") from exc

        # Catch‑all – re‑raise as LLMAPIError to keep external interface stable
        logger.exception("Unexpected error during OpenAI call")
        raise LLMAPIError(f"Unexpected error: {exc}") from exc
