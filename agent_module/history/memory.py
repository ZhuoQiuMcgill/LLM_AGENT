"""
History management for the Agent module.

This module provides classes for storing and managing conversation history.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class HistoryManager:
    """
    Abstract base class for history management.

    HistoryManager defines the interface for managing conversation history.
    Concrete implementations can use different storage backends.
    """

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the history.

        Args:
            content: The message content.
        """
        raise NotImplementedError("Subclasses must implement add_user_message")

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the history.

        Args:
            content: The message content.
        """
        raise NotImplementedError("Subclasses must implement add_assistant_message")

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the history.

        Args:
            content: The message content.
        """
        raise NotImplementedError("Subclasses must implement add_system_message")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            A list of message dictionaries representing the conversation history.
        """
        raise NotImplementedError("Subclasses must implement get_history")

    def clear(self) -> None:
        """
        Clear the conversation history.
        """
        raise NotImplementedError("Subclasses must implement clear")


class InMemoryHistoryManager(HistoryManager):
    """
    In-memory implementation of HistoryManager.

    This implementation stores conversation history in memory as a list of messages.
    """

    def __init__(self, max_messages: Optional[int] = None):
        """
        Initialize the in-memory history manager.

        Args:
            max_messages: Optional maximum number of messages to store.
                If provided, the history will be truncated to this length.
        """
        self.history: List[Dict[str, str]] = []
        self.max_messages = max_messages
        logger.debug(f"Initialized InMemoryHistoryManager with max_messages={max_messages}")

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the history.

        Args:
            content: The message content.
        """
        self.history.append({"role": "user", "content": content})
        self._truncate_if_needed()

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the history.

        Args:
            content: The message content.
        """
        self.history.append({"role": "assistant", "content": content})
        self._truncate_if_needed()

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the history.

        Args:
            content: The message content.
        """
        # System messages typically go at the beginning of the history
        self.history.insert(0, {"role": "system", "content": content})
        self._truncate_if_needed()

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            A list of message dictionaries representing the conversation history.
        """
        return self.history.copy()

    def clear(self) -> None:
        """
        Clear the conversation history.
        """
        self.history = []
        logger.debug("History cleared")

    def _truncate_if_needed(self) -> None:
        """
        Truncate the history to the maximum length if needed.
        """
        if self.max_messages and len(self.history) > self.max_messages:
            # Keep the most recent messages, but preserve any system messages at the beginning
            system_messages = [msg for msg in self.history if msg["role"] == "system"]
            non_system_messages = [msg for msg in self.history if msg["role"] != "system"]

            # Truncate non-system messages
            truncated_non_system = non_system_messages[-(self.max_messages - len(system_messages)):]

            # Combine system messages with truncated non-system messages
            self.history = system_messages + truncated_non_system

            logger.debug(f"History truncated to {len(self.history)} messages")
