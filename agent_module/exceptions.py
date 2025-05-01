"""
Custom exceptions for the agent module.
"""


class AgentError(Exception):
    """Base exception class for all Agent-related errors."""
    pass


class LLMAPIError(AgentError):
    """Exception raised when there's an error with the LLM API call."""
    pass


class ConfigurationError(AgentError):
    """Exception raised when there's a configuration issue."""
    pass


class HistoryError(AgentError):
    """Exception raised when there's an error with the conversation history."""
    pass